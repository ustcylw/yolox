import os, sys
import torch



def cuda_available():
    return torch.cuda.is_available()


def cuda_count():
    return torch.cuda.device_count()


def get_device(device):
    if (isinstance(device, str) and device == 'cpu') or (isinstance(device, int) and device == -1):
        return torch.device(device='cpu')
    elif isinstance(device, int) and device > -1:
        return torch.device(device=device)
    elif isinstance(device, list):
        return [torch.device(device=device_id) for device_id in device]
    else:
        raise ValueError(f'device error !!! {device=}')
