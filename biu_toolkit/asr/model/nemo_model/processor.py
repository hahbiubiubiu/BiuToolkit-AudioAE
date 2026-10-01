import torch
import torchaudio
import torch.nn as nn
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MyAudioToMelSpectrogramPreprocessor(nn.Module):
    def __init__(self, original_preprocessor):
        super().__init__()
        featurizer = original_preprocessor.featurizer
        self.dither = getattr(featurizer, 'dither')
        self.preemph = getattr(featurizer, 'preemph')
        self.mag_power = getattr(featurizer, 'mag_power')
        self.use_grads = getattr(featurizer, 'use_grads')
        self.sample_rate = getattr(featurizer, 'sample_rate')
        self.normalize_type = getattr(featurizer, 'normalize')

        self.n_fft = getattr(featurizer, 'n_fft')
        self.win_length = getattr(featurizer, 'win_length')
        self.hop_length = getattr(featurizer, 'hop_length')

        self.pad_to = getattr(featurizer, 'pad_to')
        self.exact_pad = getattr(featurizer, 'exact_pad')
        self.pad_value = getattr(featurizer, 'pad_value')
        self.max_length = getattr(featurizer, 'max_length')
        self.stft_pad_amount = getattr(featurizer, 'stft_pad_amount')
        
        self.frame_splicing = getattr(featurizer, 'frame_splicing')
        self.log_zero_guard_type = getattr(featurizer, 'log_zero_guard_type')
        self.log_zero_guard_value = getattr(featurizer, 'log_zero_guard_value')

        if hasattr(featurizer, 'fb'):
            fb = featurizer.fb
        elif hasattr(featurizer, 'filter_banks'):
            fb = featurizer.filter_banks
            
        self.register_buffer('mel_basis', fb.clone().detach())

        window = getattr(featurizer, 'window')
        self.register_buffer('window', window.clone().detach())

    def print_details(self):
        # detaill print
        print("=" * 20)
        print(f"dither: {self.dither}")
        print(f"preemph: {self.preemph}")
        print(f"mag_power: {self.mag_power}")
        print(f"sample_rate: {self.sample_rate}")
        print(f"normalize_type: {self.normalize_type}")
        print(f"Mel Basis Shape: {self.mel_basis.shape}")

        print(f"n_fft: {self.n_fft}")
        print(f"win_length: {self.win_length}")
        print(f"hop_length: {self.hop_length}")

        print(f"pad_to: {self.pad_to}")
        print(f"exact_pad: {self.exact_pad}")
        print(f"pad_value: {self.pad_value}")
        print(f"max_length: {self.max_length}")
        print(f"stft_pad_amount: {self.stft_pad_amount}")

        print(f"frame_splicing: {self.frame_splicing}")
        print(f"log_zero_guard_type: {self.log_zero_guard_type}")
        print(f"log_zero_guard_value: {self.log_zero_guard_value}")
        print("=" * 20)

    def apply_dither(self, signal):
        if self.dither > 0 and self.training:
            noise = torch.randn_like(signal) * self.dither
            signal = signal + noise
        return signal

    def preemphasis(self, x, length):
        if self.preemph is None or self.preemph == 0:
            return x
        timemask = torch.arange(x.size(1), device=x.device).expand(x.size(0), x.size(1)) < length.unsqueeze(1)
        x_pre = torch.cat(
            [x[:, :1], x[:, 1:] - self.preemph * x[:, :-1]],
            dim=1
        )
        x_pre = x_pre.masked_fill(~timemask, 0.0)
        return x_pre

    def splice_frames(self, x, frame_splicing):
        seq = [x]
        for n in range(1, frame_splicing):
            seq.append(torch.cat([x[:, :, :n], x[:, :, n:]], dim=2))
        return torch.cat(seq, dim=1)

    def get_seq_len(self, seq_len):
        pad_amount = self.stft_pad_amount * 2 if self.stft_pad_amount is not None else self.n_fft // 2 * 2
        seq_len = torch.floor_divide((seq_len + pad_amount - self.n_fft), self.hop_length)
        return seq_len.to(dtype=torch.long)

    def forward(self, x, seq_len):
        seq_len_time = seq_len
        seq_len_unfixed = self.get_seq_len(seq_len)
        seq_len = torch.where(seq_len == 0, torch.zeros_like(seq_len_unfixed), seq_len_unfixed)
        
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        
        if self.stft_pad_amount is not None:
            x = torch.nn.functional.pad(
                x.unsqueeze(1), (self.stft_pad_amount, self.stft_pad_amount), "constant"
            ).squeeze(1)
        
        x = self.apply_dither(x)
        x = self.preemphasis(x, seq_len_time)
        # print(f"Preemph Signal Shape: {x.shape}")
        with torch.amp.autocast(x.device.type, enabled=False):
            x = torch.stft(
                x,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                window=self.window.to(dtype=torch.float, device=x.device),
                center=False if self.exact_pad else True,
                return_complex=True,
                pad_mode='constant',
            )

        x = torch.abs(x)
        x = x.pow(self.mag_power)
        # print(f"Power Spec Shape: {x.shape}")

        with torch.amp.autocast(x.device.type, enabled=False):
            x = torch.matmul(self.mel_basis.to(x.dtype), x)
        # print(f"Mel Spec Shape: {x.shape}")

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
            var = ((x - mean) ** 2 * mask).sum(dim=2, keepdim=True) / (seq_len[:, None, None] - 1.0)
            std = torch.sqrt(var) + 1e-5
            x = (x - mean) / std

        max_len = x.size(-1)
        mask = torch.arange(max_len, device=x.device)
        mask = mask.repeat(x.size(0), 1) >= seq_len.unsqueeze(1)
        x = x.masked_fill(
            mask.unsqueeze(1).type(torch.bool).to(device=x.device), 
            self.pad_value
        )

        if self.pad_to > 0:
            pad_amt = x.size(-1) % self.pad_to
            if pad_amt != 0:
                x = nn.functional.pad(x, (0, self.pad_to - pad_amt), value=self.pad_value)
        elif self.pad_to == "max":
            x = nn.functional.pad(x, (0, self.max_length - x.size(-1)), value=self.pad_value)
        
        return x, seq_len