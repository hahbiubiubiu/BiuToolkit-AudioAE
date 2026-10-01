# biu-toolkit

`biu-toolkit` 实现了统一的 ASR 模型调用、NeMo ASR 可微损失/梯度计算，以及 SRM 语音表征提取。

## 已实现的 ASR 模型

- Whisper：`tiny`、`base`、`small`、`medium`、`large-v3`
- NeMo：Jasper、QuartzNet、Citrinet、ContextNet、Squeezeformer、Conformer CTC/RNNT、FastConformer CTC/RNNT/Hybrid、Parakeet CTC/RNNT/TDT、Canary、Canary-Qwen
- Qwen：Qwen2-Audio、Qwen3-ASR

模型默认下载或读取自：

```text
biu_toolkit/model_cache/
```

也可以通过环境变量覆盖：

```bash
export BIU_TOOLKIT_MODEL_DIR=/path/to/model_cache
```

## ASR 调用示例

统一使用 `get_func()` 获取转写函数：

```python
from biu_toolkit.asr.model import get_func

# Whisper
whisper_asr = get_func(
    "whisper_large-v3",
    device="cuda:0",
    by_file=True,
)
text = whisper_asr("example.wav")

# NeMo
nemo_asr = get_func(
    "nemo_quartznet",
    device="cuda:0",
    by_file=True,
)
text = nemo_asr("example.wav")

# Qwen3-ASR
qwen_asr = get_func(
    "qwen_qwen3_asr",
    device="cuda:0",
    by_file=True,
)
text = qwen_asr("example.wav")
```

## NeMo ASR 损失与音频梯度

NeMo 封装支持 CTC、RNNT、TDT 和 Hybrid 模型，可计算目标文本损失及损失对输入音频的梯度：

```python
import torchaudio

from biu_toolkit.asr.model import get_model

device = "cuda:0"
model = get_model(
    "nemo_quartznet",
    device=device,
    eval_mode=False,
)

audio, sample_rate = torchaudio.load("example.wav")
if sample_rate != 16000:
    audio = torchaudio.functional.resample(audio, sample_rate, 16000)
audio = audio.to(device).requires_grad_(True)

label, label_len = model.text2label("target transcription", device=device)
gradient, loss = model.calc_gradient(audio, label, label_len)

print(loss.item())
print(gradient.shape)
```

如果只需要损失值：

```python
loss = model.calc_loss(audio, label, label_len)
```

## SRM 模型调用

SRM 支持 WavLM、HuBERT 和 Wav2Vec2 的 `base`、`large` 等配置，用于提取中间层语音表征：

```python
import torchaudio

from biu_toolkit.srm import SRM

audio, sample_rate = torchaudio.load("example.wav")

srm = SRM(
    model_name="Wav2vec2",
    model_size="base",
    device="cuda:0",
)

features = srm.get_features(
    audio,
    sample_rate=sample_rate,
)
```
