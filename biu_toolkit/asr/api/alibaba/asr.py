import io
import os
import time
import json
import torch
import http.client
import numpy as np
from pathlib import Path
import scipy.io.wavfile as wavfile
from urllib.parse import urlencode
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

APP_KEY = os.getenv("APP_KEY")
ACCESS_KEY_ID = os.getenv("ACCESS_KEY_ID")
ACCESS_KEY_SECRET = os.getenv("ACCESS_KEY_SECRET")

class AlibabaASR:
    def __init__(self, 
        app_key=APP_KEY, access_key_id=ACCESS_KEY_ID, access_key_secret=ACCESS_KEY_SECRET, 
        region="cn-shanghai"
    ):
        self.app_key = app_key
        self.ak_id = access_key_id
        self.ak_secret = access_key_secret
        self.region = region
        
        self._cached_token = None
        self._cached_expire_time = 0

    def _get_token(self):
        """内部方法：请求新的阿里云鉴权 Token"""
        client = AcsClient(self.ak_id, self.ak_secret, self.region)
        request = CommonRequest()
        request.set_method('POST')
        request.set_domain(f'nls-meta.{self.region}.aliyuncs.com')
        request.set_version('2019-02-28')
        request.set_action_name('CreateToken')
        
        response = client.do_action_with_exception(request)
        token_data = json.loads(response)
        
        self._cached_token = token_data['Token']['Id']
        self._cached_expire_time = token_data['Token']['ExpireTime']
        return self._cached_token

    def _ensure_valid_token(self):
        """确保 Token 处于有效期内，提前 60 秒刷新"""
        current_time = int(time.time())
        if self._cached_token is None or current_time >= self._cached_expire_time - 60:
            self._get_token()
        return self._cached_token

    @staticmethod
    def _numpy_to_wav_bytes(audio_np, sr=16000):
        """将 Numpy 数组在内存中直接转换为 16-bit PCM WAV 字节流"""
        audio_int16 = np.int16(np.clip(audio_np, -1.0, 1.0) * 32767)
        buffer = io.BytesIO()
        wavfile.write(buffer, sr, audio_int16)
        return buffer.getvalue()

    def transcribe(self, 
        x_cand: [np.ndarray, torch.Tensor, io.BytesIO, bytes, str], 
        sample_rate=16000
    ):
        """
        核心对外接口：接收 Numpy 音频数组，返回识别出的文本。
        包含防止 QPS 超限的退避重试机制。
        """
        if isinstance(x_cand, np.ndarray):
            audio_bytes = self._numpy_to_wav_bytes(x_cand, sr=sample_rate)
        elif isinstance(x_cand, torch.Tensor):
            audio_bytes = self._numpy_to_wav_bytes(x_cand.cpu().numpy(), sr=sample_rate)
        elif isinstance(x_cand, io.BytesIO):
            audio_bytes = x_cand.getvalue()
        elif isinstance(x_cand, bytes):
            audio_bytes = x_cand
        elif isinstance(x_cand, str) and os.path.isfile(x_cand):
            with open(x_cand, 'rb') as f:
                audio_bytes = f.read()
        else:
            raise ValueError("Unsupported input type for x_cand")

        token = self._ensure_valid_token()
        host = f"nls-gateway.{self.region}.aliyuncs.com"
        params = {
            'appkey': self.app_key,
            'format': 'wav',
            'sample_rate': sample_rate,
            'enable_punctuation_prediction': 'false', 
            'enable_inverse_text_normalization': 'true',
            'enable_voice_detection': 'false'
        }
        url_path = f"/stream/v1/asr?{urlencode(params)}"
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                conn = http.client.HTTPSConnection(host, timeout=10)
                headers = {
                    'X-NLS-Token': token,
                    'Content-Type': 'application/octet-stream',
                    'Content-Length': str(len(audio_bytes))
                }
                conn.request('POST', url_path, body=audio_bytes, headers=headers)
                response = conn.getresponse()
                body = response.read()
                conn.close()
                # print(f"Attempt {attempt + 1}: Response status: {response.status}, body: {body.decode('utf-8')}")
                if response.status == 200:
                    res_json = json.loads(body)
                    if res_json.get('status') == 20000000:
                        return res_json.get('result', '')
                    elif res_json.get('status') == 40000005:  # 触发 QPS 限流
                        time.sleep(1 + attempt * 0.5)
                        continue
                    else:
                        return ""
                elif response.status in [429, 403, 502]:
                    time.sleep(2 ** attempt)  # 指数退避长休眠
                    continue
                else:
                    time.sleep(0.5)
            except Exception:
                time.sleep(1)
        # return ""
        raise RuntimeError("Failed to transcribe after multiple attempts.")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Transcribe one audio file with Alibaba Cloud ASR")
    parser.add_argument("audio_file", help="Path to a WAV audio file")
    args = parser.parse_args()

    api = AlibabaASR()
    text = api.transcribe(args.audio_file)
    print(text)
