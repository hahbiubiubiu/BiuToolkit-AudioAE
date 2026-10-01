import torch
import torchaudio

from .asr_factory import load_asr_model, get_model_type
from .asr_factory import get_model_list as get_model_list_tmp
from .processor import MyAudioToMelSpectrogramPreprocessor

class AsrModel:
    def __init__(self, model_name, device="cuda", load_best=False):
        self.model = load_asr_model(
            model_name,
            device=device,
            load_best=load_best,
        )
        self.model_type = get_model_type(model_name)
        self.preprocessor = MyAudioToMelSpectrogramPreprocessor(self.model.preprocessor).to(device)
        self.preprocessor.eval()

    def text2label(self, text, device="cuda"):
        # 根据模型类型选择不同的 tokenizer
        if self.model_type == "ctc":
            # CTC 模型使用 decoder.vocabulary
            if hasattr(self.model, 'tokenizer'):
                token_ids = [self.model.tokenizer.text_to_ids(text)]
            elif hasattr(self.model.decoder, 'vocabulary'):
                # 使用字符级 vocabulary
                vocabulary = self.model.decoder.vocabulary
                token_ids = [[vocabulary.index(c) if c in vocabulary else 0 for c in text.lower()]]
            else:
                raise AttributeError("Cannot find tokenizer or vocabulary in the model")
        else:
            # RNNT 模型通常有 tokenizer
            token_ids = [self.model.tokenizer.text_to_ids(text)]
        
        labels = torch.tensor(token_ids, dtype=torch.int32).to(device)
        length = torch.tensor([len(ids) for ids in token_ids], dtype=torch.int32).to(device)
        return labels, length

    def calc_loss(self, audio, label, label_len):
        audio_len = torch.tensor([audio.size(1)], dtype=torch.int32).to(audio.device)
        feature, feature_len = self.preprocessor(x=audio, seq_len=audio_len)
        encoded, encoded_len = self.model.encoder(audio_signal=feature, length=feature_len)

        if self.model_type == "ctc":
            log_probs = self.model.decoder(encoder_output=encoded)
            log_probs = log_probs.transpose(0, 1)
            log_probs = torch.nn.functional.log_softmax(log_probs, dim=2)
            loss = torch.nn.functional.ctc_loss(
                log_probs=log_probs,
                targets=label,
                input_lengths=encoded_len,
                target_lengths=label_len,
                blank=self.model.decoder.num_classes_with_blank - 1,
                reduction="mean",
            )
            
        elif self.model_type == "rnnt":
            encoded_len = encoded_len.to(dtype=torch.int32)
            decoded_out, _, _ = self.model.decoder(targets=label, target_length=label_len)
            self.model.joint._fuse_loss_wer = False
            joint_out = self.model.joint(encoder_outputs=encoded, decoder_outputs=decoded_out)
            max_len_in_batch = encoded_len.max().item()
            if joint_out.size(1) > max_len_in_batch:
                joint_out = joint_out[:, :max_len_in_batch, :, :]
            blank_id = self.model.decoder.blank_idx 
            loss = torchaudio.functional.rnnt_loss(
                logits=joint_out,
                targets=label,
                logit_lengths=encoded_len,
                target_lengths=label_len,
                blank=blank_id,
                reduction="mean"
            )

        elif self.model_type == "tdt":
            encoded_len = encoded_len.to(dtype=torch.int32)
            decoded_out, _, _ = self.model.decoder(targets=label, target_length=label_len)
            self.model.joint._fuse_loss_wer = False
            joint_out = self.model.joint(encoder_outputs=encoded, decoder_outputs=decoded_out)
            loss = self.model.loss(
                log_probs=joint_out, 
                targets=label, 
                input_lengths=encoded_len, 
                target_lengths=label_len
            )

        elif self.model_type == "hybrid":
            encoded_len = encoded_len.to(dtype=torch.int32)
            decoded_out, _, _ = self.model.decoder(targets=label, target_length=label_len)
            self.model.joint._fuse_loss_wer = False
            joint_out = self.model.joint(encoder_outputs=encoded, decoder_outputs=decoded_out)
            max_len_in_batch = encoded_len.max().item()
            if joint_out.size(1) > max_len_in_batch:
                joint_out = joint_out[:, :max_len_in_batch, :, :]
            blank_id = self.model.decoder.blank_idx
            loss_rnnt = torchaudio.functional.rnnt_loss(
                logits=joint_out,
                targets=label,
                logit_lengths=encoded_len,
                target_lengths=label_len,
                blank=blank_id,
                reduction="mean"
            )

            ctc_logits = self.model.ctc_decoder(encoder_output=encoded)
            ctc_log_probs = torch.nn.functional.log_softmax(ctc_logits, dim=-1).transpose(0, 1)
            ctc_blank_id = self.model.ctc_decoder.num_classes_with_blank - 1
            loss_ctc = torch.nn.functional.ctc_loss(
                log_probs=ctc_log_probs,
                targets=label,
                input_lengths=encoded_len,
                target_lengths=label_len,
                blank=ctc_blank_id,
                reduction="mean",
                zero_infinity=True
            )

            # 获取 CTC 权重的配置，默认通常是 0.3 或 0.1
            ctc_weight = self.model.cfg.aux_ctc.ctc_loss_weight
            loss = loss_rnnt + (ctc_weight * loss_ctc)

        return loss
    
    def calc_gradient(self, audio, label, label_len):
        """
        计算目标文本相对于输入音频的梯度
        
        使用 torch.autograd.grad() 而不是 backward()，
        这样即使模型在 eval() 模式下也能正常工作
        
        Args:
            audio: 输入音频张量，需要设置 requires_grad=True
            label: 目标文本标签
            label_len: 标签长度
            
        Returns:
            gradient: 损失相对于音频的梯度
        """
        loss = self.calc_loss(audio, label, label_len)
        gradient = torch.autograd.grad(loss, audio)[0]
        return gradient, loss

    def test_processor(self, audio):
        audio_len = torch.tensor([audio.size(1)], dtype=torch.int32).to(audio.device)
        feature_ori, feature_len_ori = self.model.preprocessor(input_signal=audio, length=audio_len)
        feature_new, feature_len_new = self.preprocessor(x=audio, seq_len=audio_len)
        is_close = torch.allclose(feature_ori, feature_new)
        
        if not is_close:
            # 打印详细差异信息
            diff = (feature_ori - feature_new).abs()
            print(f"Original feature shape: {feature_ori.shape}, New feature shape: {feature_new.shape}")
            print(f"Are they close? {is_close}")
            print(f"Max difference: {diff.max().item():.6f}")
            print(f"Mean difference: {diff.mean().item():.6f}")
            print(f"Original range: [{feature_ori.min():.4f}, {feature_ori.max():.4f}]")
            print(f"New range: [{feature_new.min():.4f}, {feature_new.max():.4f}]")
            
            # 检查是否是因为 dither (随机噪声)
            if hasattr(self.preprocessor, 'dither') and self.preprocessor.dither > 0:
                print(f"⚠️ Dither is enabled ({self.preprocessor.dither}), which adds random noise during training")
            
            # 检查是否在训练模式
            if self.preprocessor.training:
                print(f"⚠️ Preprocessor is in training mode (use .eval() to disable dither)")
        else:
            print(f"Original feature shape: {feature_ori.shape}, New feature shape: {feature_new.shape}, Are they close? {is_close}")

    def transcribe(self, audio):
        audio_len = torch.tensor([audio.size(1)], dtype=torch.int32).to(audio.device)
        if self.model_type == "ctc":
            log_probs, encoded_len, greedy_predictions = self.model(
                input_signal=audio, 
                input_signal_length=audio_len
            )
            text = self.model.decoding.ctc_decoder_predictions_tensor(
                greedy_predictions, encoded_len
            )[0].text
        
        elif self.model_type in ["rnnt", "tdt", "hybrid"]:
            processed, p_len = self.model.preprocessor(input_signal=audio, length=audio_len)
            encoded, encoded_len = self.model.encoder(audio_signal=processed, length=p_len)
            text = self.model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=encoded, 
                encoded_lengths=encoded_len
            )[0].text
        return text
    
    def transcribe_with_file(self, audio_path):
        audio, sr = torchaudio.load(audio_path)
        text = self.transcribe(audio.to(next(self.model.parameters()).device))
        return text

def get_model_list():
    return get_model_list_tmp()


def asr_model(model_name, device='cuda', eval_mode=True):
    return AsrModel(model_name, device=device, load_best=eval_mode)

def asr_func(model_name, device='cuda', by_file=True, eval_mode=True):
    model = asr_model(model_name=model_name, device=device, eval_mode=eval_mode)
    if by_file:
        return lambda wav_path: model.transcribe_with_file(wav_path)
    else:
        return lambda audio: model.transcribe(audio)