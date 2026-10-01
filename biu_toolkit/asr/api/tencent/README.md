# Tencent ASR Refactored Interface

替换同名文件后，上层只需要调用：

```python
from asr_func.tencent.tencent_asr import asr_by_file, asr_by_bytes, asr_by_data

text = asr_by_file("xxx.wav")
# or
text = asr_by_data(wav_tensor, sample_rate=16000)
```

## 环境变量

多账户推荐：

```bash
ACCOUNT_SECRET_ID_1=xxx
ACCOUNT_SECRET_KEY_1=xxx
ACCOUNT_APP_ID_1=xxx
ACCOUNT_SECRET_ID_2=xxx
ACCOUNT_SECRET_KEY_2=xxx
ACCOUNT_APP_ID_2=xxx
```

单账户兼容：

```bash
TENCENTCLOUD_SECRET_ID=xxx
TENCENTCLOUD_SECRET_KEY=xxx
APPID=xxx
```

## 四个底层 API

每个文件都暴露同样的标准接口：

```python
asr_by_bytes(audio_data: bytes) -> str
asr_by_file(audio_file: str) -> str
```

底层 API 只做一件事：成功返回文本，失败抛出 `TencentASRError` 或其子类。账户/API fallback 只放在 `tencent_asr.py`。

## fallback 策略

- `TencentASRResourceExhaustedError(4004)`：禁用当前账户下当前 API，切换下一个 API。
- `TencentASRAuthError(4002/4003/4005)`：禁用当前账户，切换下一个账户。
- `TencentASRServerError/TencentASRNetworkError/TencentASRTimeoutError/4006`：切换下一个 API。
- `4001/4007` 等确定性输入错误：直接抛出，不重试。

可查看状态：

```python
from asr_func.tencent.tencent_asr import get_fallback_status, reset_fallback_state
print(get_fallback_status())
reset_fallback_state(reload_accounts=True)
```


## v3 fallback / retry 策略

`tencent_asr.py` 现在对临时错误会先重试当前 API，再切换 API：

- `TencentASRResourceExhaustedError(4004)`：禁用当前账户下当前 API，切换下一个 API。
- `TencentASRAuthError(4002/4003/4005)`：禁用当前账户，切换下一个账户。
- `TencentASRNetworkError / TencentASRTimeoutError / TencentASRServerError / 4006`：先重试当前 API；重试次数耗尽后，切换下一个 API，但不禁用该 API。
- `4001/4007` 等确定性输入错误：直接抛出，不重试、不切换 API。

可通过以下参数调整重试：

```python
text = asr_by_file(
    "test.wav",
    max_retries_per_api=2,      # 每个 API 失败后额外重试 2 次；总尝试次数 = 1 + 2
    retry_base_sleep=0.35,      # 首次退避时间
    retry_max_sleep=2.0,        # 最大退避时间
)
```
