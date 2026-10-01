import torch
import torchaudio
    
class SRM():
    def __init__(self, model_name="Wav2vec2", model_size="base", device='cuda'):
        SRM_BUNDLES = {
            "base": {
                "WavLM": torchaudio.pipelines.WAVLM_BASE,
                "HuBERT": torchaudio.pipelines.HUBERT_BASE,
                "Wav2vec2": torchaudio.pipelines.WAV2VEC2_BASE,
            },
            "large": {
                "WavLM": torchaudio.pipelines.WAVLM_LARGE,
                "HuBERT": torchaudio.pipelines.HUBERT_LARGE,
                "Wav2vec2": torchaudio.pipelines.WAV2VEC2_LARGE,
                "HuBERT_asr": torchaudio.pipelines.HUBERT_ASR_LARGE,
                "Wav2vec2_asr": torchaudio.pipelines.WAV2VEC2_ASR_LARGE_LV60K_960H,
            },
            "xlarge": {
                "HuBERT_xlarge": torchaudio.pipelines.HUBERT_ASR_XLARGE,
            }
        }

        self.device = device
        self.model_name = model_name
        self.model_size = model_size
        bundle = SRM_BUNDLES[model_size][model_name]

        print(f"Loading {model_name}_{model_size} - {type(bundle).__name__} on {device}...")
        self.model = bundle.get_model().to(device)
        self.sample_rate = bundle.sample_rate
        self.model.eval()

        self._init_conv_length_params()

    def _init_conv_length_params(self):
        model = self.model.model if hasattr(self.model, "model") else self.model

        kernel_sizes = []
        strides = []
        paddings = []
        dilations = []

        for conv_layer in model.feature_extractor.conv_layers:
            conv = conv_layer.conv

            kernel_sizes.append(conv.kernel_size[0])
            strides.append(conv.stride[0])
            paddings.append(conv.padding[0])
            dilations.append(conv.dilation[0])

        self.conv_kernel_sizes = torch.tensor(
            kernel_sizes,
            dtype=torch.long,
            device=self.device
        )
        self.conv_strides = torch.tensor(
            strides,
            dtype=torch.long,
            device=self.device
        )
        self.conv_paddings = torch.tensor(
            paddings,
            dtype=torch.long,
            device=self.device
        )
        self.conv_dilations = torch.tensor(
            dilations,
            dtype=torch.long,
            device=self.device
        )

    def _get_feature_lengths(self, input_lengths):
        lengths = input_lengths.to(self.device).clone()

        for kernel_size, stride, padding, dilation in zip(
            self.conv_kernel_sizes,
            self.conv_strides,
            self.conv_paddings,
            self.conv_dilations
        ):
            lengths = torch.div(
                lengths + 2 * padding - dilation * (kernel_size - 1) - 1,
                stride,
                rounding_mode="floor"
            ) + 1

            lengths = torch.clamp(lengths, min=0)

        return lengths

    def trim_features_by_length(self, features, feature_lengths):
        """
        features:
            list[Tensor], 每层 [B, T_max, D]

        feature_lengths:
            Tensor, [B]

        return:
            trimmed_features:
                list[list[Tensor]]
                trimmed_features[layer_idx][batch_idx] -> [T_i, D]
        """

        trimmed_features = []

        for feat in features:
            layer_features = []

            for i, length in enumerate(feature_lengths):
                layer_features.append(feat[i, :length])

            trimmed_features.append(layer_features)

        return trimmed_features

    def get_features_batch(self, waveforms, sample_rate, num_layers=None):
        """
        waveforms:
            list[Tensor]
                each item can be [T] or [1, T]

        return:
            features:
                list[list[Tensor]]
                features[layer_idx][batch_idx] -> [T_i, D]
            feature_lengths:
                Tensor, shape [B]
        """

        if not isinstance(waveforms, list):
            raise ValueError("waveforms should be a list of tensors")
        processed = []
        for wav in waveforms:
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)  # [1, T]
            if wav.dim() != 2 or wav.size(0) != 1:
                raise ValueError("Each waveform in list should have shape [T] or [1, T]")
            if sample_rate != self.sample_rate:
                wav = torchaudio.functional.resample(
                    wav, sample_rate, self.sample_rate
                )
            processed.append(wav.squeeze(0))  # [T]

        lengths = torch.tensor(
            [wav.size(0) for wav in processed],
            dtype=torch.long,
            device=self.device
        )
        if self.model_name == "WavLM":
            # torchaudio's WavLM attention currently rejects attention masks,
            # and its convolutional front-end normalization is also affected
            # by raw waveform padding. Exact equivalence with single-item
            # inference therefore requires identical waveform lengths. Other
            # lengths run as independent sub-batches and are restored to input
            # order.
            feature_lengths = self._get_feature_lengths(lengths)
            groups = {}
            for index, length in enumerate(lengths.detach().cpu().tolist()):
                groups.setdefault(int(length), []).append(index)
            output = None
            for _, indices in groups.items():
                feature_length = int(feature_lengths[indices[0]].item())
                grouped = [processed[index] for index in indices]
                grouped_waveforms = torch.nn.utils.rnn.pad_sequence(
                    grouped, batch_first=True).to(self.device)
                group_features, _ = self.model.extract_features(
                    grouped_waveforms, num_layers=num_layers)
                if output is None:
                    output = [[None] * len(processed) for _ in group_features]
                for layer_index, feature in enumerate(group_features):
                    if feature.size(1) != feature_length:
                        raise RuntimeError(
                            "WavLM feature-length grouping produced an unexpected length")
                    for local_index, source_index in enumerate(indices):
                        output[layer_index][source_index] = feature[local_index]
            return output, feature_lengths

        same_length = torch.all(lengths == lengths[0]).item()
        if same_length:
            waveforms = torch.stack(processed, dim=0)  # [B, T]
        else:
            waveforms = torch.nn.utils.rnn.pad_sequence(
                processed,
                batch_first=True
            )  # [B, T_max]

        waveforms = waveforms.to(self.device)
        if self.model_name in ["HuBERT_asr", "Wav2vec2_asr", "HuBERT", "Wav2vec2"]:
            features, feature_lengths = self.model.extract_features(
                waveforms,
                lengths=lengths,
                num_layers=num_layers
            )
        else:
            raise ValueError(f"Unsupported model_name: {self.model_name}")
        features = self.trim_features_by_length(features, feature_lengths)
        return features, feature_lengths


    def get_features(self, waveform, sample_rate, num_layers=None):
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        features, _ = self.model.extract_features(
            waveform.to(self.device),
            num_layers=num_layers
        )
        return features
