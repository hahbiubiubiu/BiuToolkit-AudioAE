from __future__ import annotations

import http.client
import json
import os
import time
from urllib.parse import urlencode

from audio_toolkit.asr.api.base import ASRAPI


class AlibabaASRError(RuntimeError):
    def __init__(self, http_status: int, status: object, message: object):
        self.http_status = http_status
        self.status = status
        self.message = str(message or "")
        super().__init__(
            f"HTTP {http_status}, status={status}, message={self.message!r}"
        )


class AlibabaASR(ASRAPI):
    """Alibaba Cloud streaming ASR client."""

    def __init__(
        self,
        app_key: str | None = None,
        access_key_id: str | None = None,
        access_key_secret: str | None = None,
        *,
        region: str = "cn-shanghai",
        sample_rate: int = 16_000,
        max_attempts: int = 5,
        env_file: str | os.PathLike[str] | None = None,
        load_env: bool = True,
    ):
        super().__init__(env_file=env_file, load_env=load_env)

        self.app_key = app_key or os.getenv("APP_KEY")
        self.access_key_id = access_key_id or os.getenv("ACCESS_KEY_ID")
        self.access_key_secret = access_key_secret or os.getenv(
            "ACCESS_KEY_SECRET"
        )
        self.region = region
        self.sample_rate = sample_rate
        self.max_attempts = max(1, int(max_attempts))
        self._token: str | None = None
        self._token_expire_time = 0

    def _get_token(self) -> str:
        if not all(
            (self.app_key, self.access_key_id, self.access_key_secret)
        ):
            raise ValueError("Alibaba ASR credentials are not configured")

        from aliyunsdkcore.client import AcsClient
        from aliyunsdkcore.request import CommonRequest

        client = AcsClient(
            self.access_key_id,
            self.access_key_secret,
            self.region,
        )
        request = CommonRequest()
        request.set_method("POST")
        request.set_domain(f"nls-meta.{self.region}.aliyuncs.com")
        request.set_version("2019-02-28")
        request.set_action_name("CreateToken")
        token = json.loads(client.do_action_with_exception(request))["Token"]
        self._token = token["Id"]
        self._token_expire_time = token["ExpireTime"]
        return self._token

    def _valid_token(self) -> str:
        if self._token is None or time.time() >= self._token_expire_time - 60:
            return self._get_token()
        return self._token

    def _transcribe_bytes(
        self,
        audio_data: bytes,
        sample_rate: int | None,
    ) -> str:
        token = self._valid_token()
        host = f"nls-gateway.{self.region}.aliyuncs.com"
        params = urlencode(
            {
                "appkey": self.app_key,
                "format": "wav",
                "sample_rate": sample_rate or self.sample_rate,
                "enable_punctuation_prediction": "false",
                "enable_inverse_text_normalization": "true",
                "enable_voice_detection": "false",
            }
        )
        path = f"/stream/v1/asr?{params}"

        for attempt in range(self.max_attempts):
            try:
                status, body = self._request(host, path, token, audio_data)
            except (OSError, http.client.HTTPException) as exc:
                if attempt + 1 == self.max_attempts:
                    raise RuntimeError("Alibaba ASR request failed") from exc
                time.sleep(min(2**attempt, 4))
                continue
            result = json.loads(body)
            result_status = result.get("status")

            if status == 200 and result_status == 20000000:
                return result.get("result", "")

            retryable = status == 429 or status >= 500
            retryable = retryable or result_status == 40000005
            if retryable and attempt + 1 < self.max_attempts:
                time.sleep(min(2**attempt, 4))
                continue

            raise AlibabaASRError(
                http_status=status,
                status=result_status,
                message=result.get("message"),
            )

    @staticmethod
    def _request(
        host: str,
        path: str,
        token: str,
        audio_data: bytes,
    ) -> tuple[int, bytes]:
        connection = http.client.HTTPSConnection(host, timeout=10)
        try:
            connection.request(
                "POST",
                path,
                body=audio_data,
                headers={
                    "X-NLS-Token": token,
                    "Content-Type": "application/octet-stream",
                    "Content-Length": str(len(audio_data)),
                },
            )
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()
