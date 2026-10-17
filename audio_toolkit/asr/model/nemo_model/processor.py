import torch
import torch.nn as nn


class DifferentiableAudioPreprocessor(nn.Module):
    def __init__(self, original_preprocessor):
        super().__init__()
        featurizer = original_preprocessor.featurizer
        self.dither = featurizer.dither
        self.preemph = featurizer.preemph
        self.mag_power = featurizer.mag_power
        self.sample_rate = featurizer.sample_rate
        self.normalize_type = featurizer.normalize

        self.n_fft = featurizer.n_fft
        self.win_length = featurizer.win_length
        self.hop_length = featurizer.hop_length

        self.pad_to = featurizer.pad_to
        self.exact_pad = featurizer.exact_pad
        self.pad_value = featurizer.pad_value
        self.max_length = featurizer.max_length
        self.stft_pad_amount = featurizer.stft_pad_amount

        self.frame_splicing = featurizer.frame_splicing
        self.log_zero_guard_type = featurizer.log_zero_guard_type
        self.log_zero_guard_value = featurizer.log_zero_guard_value

        if hasattr(featurizer, "fb"):
            fb = featurizer.fb
        else:
            fb = featurizer.filter_banks

        self.register_buffer("mel_basis", fb.clone().detach())

        self.register_buffer("window", featurizer.window.clone().detach())

    def apply_dither(self, signal):
        if self.dither > 0 and self.training:
            noise = torch.randn_like(signal) * self.dither
            signal = signal + noise
        return signal

    def preemphasis(self, x, length):
        if self.preemph is None or self.preemph == 0:
            return x
        timemask = torch.arange(x.size(1), device=x.device).expand(
            x.size(0), x.size(1)
        ) < length.unsqueeze(1)
        x_pre = torch.cat(
            [x[:, :1], x[:, 1:] - self.preemph * x[:, :-1]], dim=1
        )
        x_pre = x_pre.masked_fill(~timemask, 0.0)
        return x_pre

    def splice_frames(self, x, frame_splicing):
        seq = [x]
        for n in range(1, frame_splicing):
            seq.append(torch.cat([x[:, :, :n], x[:, :, n:]], dim=2))
        return torch.cat(seq, dim=1)

    def get_seq_len(self, seq_len):
        pad_amount = (
            self.stft_pad_amount * 2
            if self.stft_pad_amount is not None
            else self.n_fft
        )
        seq_len = torch.floor_divide(
            seq_len + pad_amount - self.n_fft,
            self.hop_length,
        )
        return seq_len.to(dtype=torch.long)

    def forward(self, x, seq_len):
        seq_len_time = seq_len
        seq_len_unfixed = self.get_seq_len(seq_len)
        seq_len = torch.where(
            seq_len == 0,
            torch.zeros_like(seq_len_unfixed),
            seq_len_unfixed,
        )

        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        if self.stft_pad_amount is not None:
            x = torch.nn.functional.pad(
                x.unsqueeze(1), (self.stft_pad_amount, self.stft_pad_amount), "constant"
            ).squeeze(1)

        x = self.apply_dither(x)
        x = self.preemphasis(x, seq_len_time)
        with torch.amp.autocast(x.device.type, enabled=False):
            x = torch.stft(
                x,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window.to(dtype=torch.float, device=x.device),
                center=False if self.exact_pad else True,
                return_complex=True,
                pad_mode="constant",
            )

        x = torch.abs(x)
        x = x.pow(self.mag_power)
        with torch.amp.autocast(x.device.type, enabled=False):
            x = torch.matmul(self.mel_basis.to(x.dtype), x)

        if self.log_zero_guard_type == "add":
            x = x + self.log_zero_guard_value
        elif self.log_zero_guard_type == "clamp":
            x = torch.clamp(x, min=self.log_zero_guard_value)
        else:
            raise ValueError(f"Unsupported log_zero_guard_type: {self.log_zero_guard_type}")
        x = torch.log(x)

        if self.frame_splicing > 1:
            x = self.splice_frames(x, self.frame_splicing)

        if self.normalize_type == "per_feature":
            mask = (
                torch.arange(x.size(2), device=x.device)[None, None, :]
                < seq_len[:, None, None]
            ).float()
            mean = (x * mask).sum(dim=2, keepdim=True) / seq_len[:, None, None]
            var = ((x - mean) ** 2 * mask).sum(
                dim=2, keepdim=True
            ) / (seq_len[:, None, None] - 1.0)
            std = torch.sqrt(var) + 1e-5
            x = (x - mean) / std

        max_len = x.size(-1)
        mask = torch.arange(max_len, device=x.device)
        mask = mask.repeat(x.size(0), 1) >= seq_len.unsqueeze(1)
        x = x.masked_fill(
            mask.unsqueeze(1).type(torch.bool).to(device=x.device),
            self.pad_value,
        )

        if self.pad_to > 0:
            pad_amt = x.size(-1) % self.pad_to
            if pad_amt != 0:
                x = nn.functional.pad(
                    x,
                    (0, self.pad_to - pad_amt),
                    value=self.pad_value,
                )
        elif self.pad_to == "max":
            x = nn.functional.pad(
                x,
                (0, self.max_length - x.size(-1)),
                value=self.pad_value,
            )

        return x, seq_len
