import os
import torch
import jiwer
import pysepm

def calc_cer(pred_text, target_text):
    predicted = ''.join(filter(lambda x: x.isalnum() or x.isspace(), pred_text.lower()))
    ground_truth = ''.join(filter(lambda x: x.isalnum() or x.isspace(), target_text.lower()))
    cer = jiwer.cer(ground_truth.strip(), predicted.strip())
    return cer

def calc_wer(pred_text, target_text):
    predicted = ''.join(filter(lambda x: x.isalnum() or x.isspace(), pred_text.lower()))
    ground_truth = ''.join(filter(lambda x: x.isalnum() or x.isspace(), target_text.lower()))
    wer = jiwer.wer(ground_truth.strip(), predicted.strip())
    return wer

def calc_stoi(adv_signal, ori_signal, sr=16000):
    stoi = pysepm.stoi(
        ori_signal.squeeze().detach().cpu().numpy(),
        adv_signal.squeeze().detach().cpu().numpy(),
        sr
    )
    return stoi

def calc_ssnr(adv_signal, ori_signal, sr=16000):
    ssnr = pysepm.SNRseg(
        ori_signal.squeeze().detach().cpu().numpy(),
        adv_signal.squeeze().detach().cpu().numpy(),
        sr
    )
    return ssnr

def calc_snr(adv_signal, ori_signal):
    noise = adv_signal - ori_signal
    signal_power = torch.mean(ori_signal ** 2)
    noise_power = torch.mean(noise ** 2)
    snr = 10 * torch.log10(signal_power / noise_power + 1e-8)
    return snr.item()

def adaptive_clip(delta, carrier, epsilon=0.20):
    limit = epsilon * torch.abs(carrier)
    return torch.max(torch.min(delta, limit), -limit)

def normalize_text(text):
    return ''.join(filter(lambda x: x.isalnum() or x.isspace(), text.lower())).strip()