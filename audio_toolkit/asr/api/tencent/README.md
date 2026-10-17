# Tencent ASR

`TencentASR` implements the shared `ASRAPI` interface:

```python
from audio_toolkit.asr.api.tencent import TencentASR

asr = TencentASR(env_file="config/tencent.env")
text = asr.transcribe_file("example.wav")
text = asr.transcribe_data(waveform, sample_rate=16_000)
```

`env_file` 默认为当前执行目录下的 `.env`。

`asr.py` handles account and service switching. The four Tencent protocol
implementations and the realtime speech SDK live under `service/`.

The realtime service searches `service/tencentcloud-speech-sdk-python` first.
For an installed wheel, download Tencent's speech SDK separately and set
`TENCENT_SPEECH_SDK_PATH` to its root directory.

## Accounts

Multiple accounts are loaded in numeric order:

```bash
ACCOUNT_SECRET_ID_1=xxx
ACCOUNT_SECRET_KEY_1=xxx
ACCOUNT_APP_ID_1=xxx
ACCOUNT_SECRET_ID_2=xxx
ACCOUNT_SECRET_KEY_2=xxx
ACCOUNT_APP_ID_2=xxx
```

A single account can use `TENCENTCLOUD_SECRET_ID`,
`TENCENTCLOUD_SECRET_KEY`, and `APPID`, or be passed directly:

```python
asr = TencentASR(
    secret_id="xxx",
    secret_key="xxx",
    appid="xxx",
)
```

## Automatic switching

Services run in this order: flash, sentence, realtime, task.

- Resource exhaustion disables the current service for the current account.
- Authentication or account failure disables the current account.
- Rate limits, network failures, timeouts, and server failures retry the current
  service before switching to the next service.
- Input errors and unexpected implementation errors are raised immediately.

Retry behavior is configured once on the client:

```python
asr = TencentASR(
    max_retries=2,
    retry_base_sleep=0.35,
    retry_max_sleep=2.0,
)
```

Switch state is observable and resettable:

```python
print(asr.get_switch_status())
asr.reset_switch_state(reload_accounts=True)
```
