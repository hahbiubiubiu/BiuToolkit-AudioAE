# -*- coding: utf-8 -*-
"""Tencent realtime websocket ASR wrapper."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

from .exception import TencentASRError, raise_normalized_asr_error, read_audio_file, validate_audio_bytes

logging.getLogger("websocket").setLevel(logging.CRITICAL)
logging.getLogger("tencent_speech").setLevel(logging.ERROR)

SERVICE_NAME = "realtime"
ENGINE_MODEL_TYPE = "16k_en"
SLICE_SIZE = 6400
SEND_INTERVAL = 0.02

def _candidate_sdk_paths() -> list[Path]:
    paths: list[Path] = []
    if sdk_path := os.getenv("TENCENT_SPEECH_SDK_PATH"):
        paths.append(Path(sdk_path).expanduser())

    base_dir = Path(__file__).resolve().parent
    paths.extend(
        [
            base_dir / "tencentcloud-speech-sdk-python",
            base_dir / "tencentcloud_speech_sdk",
        ]
    )
    return paths


def _add_sdk_to_sys_path() -> None:
    for sdk_path in _candidate_sdk_paths():
        if sdk_path.exists():
            sdk_path_str = str(sdk_path)
            if sdk_path_str not in sys.path:
                sys.path.append(sdk_path_str)
            return

try:
    from common import credential  # type: ignore
    from asr import speech_recognizer  # type: ignore
except Exception:
    _add_sdk_to_sys_path()
    try:
        from common import credential  # type: ignore
        from asr import speech_recognizer  # type: ignore
    except Exception as e:
        credential = None  # type: ignore
        speech_recognizer = None  # type: ignore
        _SDK_IMPORT_ERROR = e
    else:
        _SDK_IMPORT_ERROR = None
else:
    _SDK_IMPORT_ERROR = None


def _load_local_env_if_needed() -> None:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env", override=False)


def _credential(*, secret_id: str | None = None, secret_key: str | None = None, appid: str | None = None) -> tuple[str, str, str]:
    if not (appid and secret_id and secret_key):
        _load_local_env_if_needed()
    appid = appid or os.getenv("APPID")
    secret_id = secret_id or os.getenv("TENCENTCLOUD_SECRET_ID")
    secret_key = secret_key or os.getenv("TENCENTCLOUD_SECRET_KEY")
    if not all([appid, secret_id, secret_key]):
        raise_normalized_asr_error(4002, "缺少 APPID/TENCENTCLOUD_SECRET_ID/TENCENTCLOUD_SECRET_KEY", service=SERVICE_NAME)
    return appid, secret_id, secret_key


class _Listener(speech_recognizer.SpeechRecognitionListener if speech_recognizer else object):  # type: ignore
    def __init__(self):
        self.result_parts: list[str] = []
        self.error: tuple[object, object] | None = None

    def _check_error(self, response):
        if not isinstance(response, dict):
            return
        code = response.get("code", 0)
        if "error" in response or (code not in (0, None, "0")):
            self.error = (code, json.dumps(response, ensure_ascii=False))

    def on_recognition_start(self, response):
        self._check_error(response)

    def on_sentence_begin(self, response):
        self._check_error(response)

    def on_recognition_result_change(self, response):
        self._check_error(response)

    def on_sentence_end(self, response):
        self._check_error(response)
        if isinstance(response, dict):
            text = response.get("result", {}).get("voice_text_str")
            if text:
                self.result_parts.append(text)

    def on_recognition_complete(self, response):
        self._check_error(response)

    def on_fail(self, response):
        self.error = (None, json.dumps(response, ensure_ascii=False))


def asr_by_bytes(audio_data: bytes, *, secret_id: str | None = None, secret_key: str | None = None, appid: str | None = None) -> str:
    validate_audio_bytes(audio_data)
    if _SDK_IMPORT_ERROR is not None or credential is None or speech_recognizer is None:
        raise TencentASRError(4001, f"实时识别 SDK 未正确加载: {_SDK_IMPORT_ERROR}", service=SERVICE_NAME)

    appid, secret_id, secret_key = _credential(secret_id=secret_id, secret_key=secret_key, appid=appid)
    listener = _Listener()
    recognizer = None
    try:
        cred = credential.Credential(secret_id, secret_key)
        recognizer = speech_recognizer.SpeechRecognizer(appid, cred, ENGINE_MODEL_TYPE, listener)
        recognizer.set_filter_modal(1)
        recognizer.set_filter_punc(1)
        recognizer.set_filter_dirty(1)
        recognizer.set_need_vad(1)
        recognizer.set_voice_format(1)
        recognizer.set_word_info(1)
        recognizer.set_convert_num_mode(1)
        recognizer.start()

        for offset in range(0, len(audio_data), SLICE_SIZE):
            if listener.error:
                raise_normalized_asr_error(listener.error[0], listener.error[1], service=SERVICE_NAME)
            recognizer.write(audio_data[offset: offset + SLICE_SIZE])
            time.sleep(SEND_INTERVAL)

        recognizer.stop()
        if listener.error:
            raise_normalized_asr_error(listener.error[0], listener.error[1], service=SERVICE_NAME)
        return " ".join(listener.result_parts).strip()
    except TencentASRError:
        raise
    except Exception as e:
        raise_normalized_asr_error(None, str(e), service=SERVICE_NAME)
    finally:
        if recognizer is not None:
            try:
                recognizer.stop()
            except Exception:
                pass


def asr_by_file(audio_file: str, *, secret_id: str | None = None, secret_key: str | None = None, appid: str | None = None) -> str:
    return asr_by_bytes(read_audio_file(audio_file), secret_id=secret_id, secret_key=secret_key, appid=appid)
