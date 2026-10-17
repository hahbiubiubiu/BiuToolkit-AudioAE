# audio_toolkit

`audio_toolkit` 实现了 ASR、Large Audio Model 和语音表征模型的统一调用。
支持 Python 3.10 及以上版本。

## 安装

基础安装包含 PyTorch、Torchaudio 和通用音频处理依赖：

```bash
pip install .
```

按功能安装可选依赖，避免为不使用的模型下载大型依赖：

```bash
pip install ".[whisper]"       # OpenAI Whisper
pip install ".[nemo]"          # NVIDIA NeMo ASR
pip install ".[qwen]"          # Qwen2-Audio 与 Qwen3-ASR
pip install ".[api]"           # Alibaba 与 Tencent 云端 ASR
pip install ".[metrics]"       # CER、WER、STOI 与分段 SNR
pip install ".[all,dev]"       # 全部功能与开发工具
```

`metrics` 中的 PySEPM 直接安装自其 GitHub 源码，因此需要可用的 Git 和网络。
Tencent 实时识别还需要官方 `tencentcloud-speech-sdk-python` 源码；仓库开发环境
默认从 `audio_toolkit/asr/api/tencent/service/` 查找，独立安装后可通过
`TENCENT_SPEECH_SDK_PATH` 指向 SDK 根目录。该第三方源码不会进入 wheel。

## 已实现的 ASR 模型

- Whisper：`tiny`、`base`、`small`、`medium`、`large-v3`
- NeMo：Jasper、QuartzNet、Citrinet、ContextNet、Squeezeformer、
  Conformer CTC/RNNT、FastConformer CTC/RNNT/Hybrid、Parakeet CTC/RNNT/TDT

模型默认下载或读取自：

```text
audio_toolkit/model_cache/
```

也可以通过环境变量覆盖：

```bash
export AUDIO_TOOLKIT_MODEL_DIR=/path/to/model_cache
```

## ASR 调用示例

统一使用 `get_func()` 获取转写函数：

```python
from audio_toolkit.asr.model import get_func

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

```

## 云端 ASR API

Alibaba 和 Tencent 实现相同的 `ASRAPI` 接口，可识别音频文件或内存数据：

```python
from audio_toolkit.asr.api import AlibabaASR, TencentASR

alibaba = AlibabaASR()
tencent = TencentASR(env_file="config/tencent.env")

text = alibaba.transcribe_file("example.wav")
text = tencent.transcribe_data(waveform, sample_rate=16_000)
```

未传入 `env_file` 时，API 默认读取当前执行目录下的 `.env`。Alibaba 和
Tencent 均可在构造时指定其他配置文件；直接传入的凭据优先于环境变量。

Tencent 会按照 flash、sentence、realtime、task 的顺序自动切换服务，并在
账户鉴权失败后切换至下一个已配置账户。

## Large Audio Model 调用示例

Qwen2-Audio 与 Qwen3-ASR 位于独立的 `audio_toolkit.lam` 包：

```python
from audio_toolkit.lam import get_func

qwen = get_func(
    "qwen3_asr",
    device="cuda:0",
    by_file=True,
)
text = qwen("example.wav")
```

## ASR 损失与梯度

`get_model()` 对 NeMo 和 Whisper 返回统一的 ASR 模型对象。NeMo 支持 CTC、
RNNT、TDT 和 Hybrid；两种后端实现相同的转写、损失及梯度接口：

```python
import torchaudio

from audio_toolkit.asr.model import get_model

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

gradient, loss = model.compute_audio_gradient(
    audio,
    "target transcription",
)

print(loss.item())
print(gradient.shape)
```

如果只需要损失值，或需要损失对 ASR 模型参数的梯度：

```python
loss = model.compute_loss(audio, "target transcription")
parameter_gradients, loss = model.compute_model_gradients(
    audio,
    "target transcription",
)
```

统一接口由 `ASRModel` 定义，包含 `transcribe()`、`transcribe_file()`、
`compute_loss()`、`compute_audio_gradient()` 和 `compute_model_gradients()`。

## SRM 模型调用

SRM 支持 WavLM、HuBERT 和 Wav2Vec2 的 `base`、`large` 等配置，用于提取中间层语音表征：

```python
import torchaudio

from audio_toolkit.srm import SRM

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
