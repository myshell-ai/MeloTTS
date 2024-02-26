import torch
import os
from . import utils
from cached_path import cached_path
DOWNLOAD_CKPT_URLS = {
    'EN': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/EN/checkpoint.pth',
    'EN_V2': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/EN_V2/checkpoint.pth',
    'FR': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/FR/checkpoint.pth',
    'JP': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/JP/checkpoint.pth',
    'ES': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/ES/checkpoint.pth',
    'ZH': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/ZH/checkpoint.pth',
    'KR': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/KR/checkpoint.pth',
}

DOWNLOAD_CONFIG_URLS = {
    'EN': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/EN/config.json',
    'EN_V2': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/EN_V2/config.json',
    'FR': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/FR/config.json',
    'JP': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/JP/config.json',
    'ES': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/ES/config.json',
    'ZH': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/ZH/config.json',
    'KR': 'https://myshell-public-repo-hosting.s3.amazonaws.com/openvoice/basespeakers/KR/config.json',
}

def load_or_download_config(locale):
    language = locale.split('-')[0].upper()
    assert language in DOWNLOAD_CONFIG_URLS
    config_path = cached_path(DOWNLOAD_CONFIG_URLS[language])
    return utils.get_hparams_from_file(config_path)

def load_or_download_model(locale, device):
    language = locale.split('-')[0].upper()
    assert language in DOWNLOAD_CKPT_URLS
    ckpt_path = cached_path(DOWNLOAD_CKPT_URLS[language])
    return torch.load(ckpt_path, map_location=device)
