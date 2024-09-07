# -*-coding:utf-8-*-
import torch
import numpy as np


def extract_ampl_phase(fft_im):
    fft_amp = torch.sqrt(fft_im[:, :, :, :, 0] ** 2 + fft_im[:, :, :, :, 1] ** 2)
    fft_pha = torch.atan2(fft_im[:, :, :, :, 1], fft_im[:, :, :, :, 0])
    return fft_amp, fft_pha


def low_freq_mutate(amp_x1, amp_x2, L=0.03, lam=0.5):
    # lam = np.random.uniform(0.5-alpha,0.5+alpha)
    # lam = torch.from_numpy(np.random.normal(0.5,alpha,size=amp_x1.size())).cuda()
    _, _, h, w = amp_x1.size()
    b = (np.floor(np.amin((h, w)) * L)).astype(int)  # get b

    amp_x1_clone = amp_x1.clone()
    amp_x1_clone[:, :, 0:b, 0:b] = (1 - lam) * amp_x1[:, :, 0:b, 0:b] + lam * amp_x2[:, :, 0:b, 0:b]  # top left
    amp_x1_clone[:, :, 0:b, w - b:w] = (1 - lam) * amp_x1[:, :, 0:b, w - b:w] + lam * amp_x2[:, :, 0:b,
                                                                                      w - b:w]  # top right
    amp_x1_clone[:, :, h - b:h, 0:b] = (1 - lam) * amp_x1[:, :, h - b:h, 0:b] + lam * amp_x2[:, :, h - b:h,
                                                                                      0:b]  # bottom left
    amp_x1_clone[:, :, h - b:h, w - b:w] = (1 - lam) * amp_x1[:, :, h - b:h, w - b:w] + lam * amp_x2[:, :, h - b:h,
                                                                                              w - b:w]  # bottom right

    return amp_x1_clone


def mixup(x1, x2):
    fft_x1 = torch.fft.fft2(x1.clone(), dim=(-2, -1))
    fft_x1 = torch.stack((fft_x1.real, fft_x1.imag), dim=-1)
    amp_x1, pha_x1 = extract_ampl_phase(fft_x1.clone())

    fft_x2 = torch.fft.fft2(x2.clone(), dim=(-2, -1))
    fft_x2 = torch.stack((fft_x2.real, fft_x2.imag), dim=-1)
    amp_x2, pha_x2 = extract_ampl_phase(fft_x2.clone())

    amp_x1_new = low_freq_mutate(amp_x1=amp_x1.clone(), amp_x2=amp_x2.clone(), L=0.03, lam=0.5)
    fft_clone = fft_x1.clone().cuda()
    fft_clone[:, :, :, :, 0] = torch.cos(pha_x1.clone()) * amp_x1_new.clone()
    fft_clone[:, :, :, :, 1] = torch.sin(pha_x1.clone()) * amp_x1_new.clone()

    # get the recomposed image: source content, target style
    # _, _, imgH, imgW = x1.size()
    amp_pha_unwrap = torch.fft.ifft2(torch.complex(fft_clone[:, :, :, :, 0], fft_clone[:, :, :, :, 1]),
                                     dim=(-2, -1)).float()
    return amp_pha_unwrap


def pha_unwrapping(x):
    fft_x = torch.fft.fft2(x.clone(), dim=(-2, -1))
    fft_x = torch.stack((fft_x.real, fft_x.imag), dim=-1)
    pha_x = torch.atan2(fft_x[:, :, :, :, 1], fft_x[:, :, :, :, 0])

    fft_clone = torch.zeros(fft_x.size(), dtype=torch.float).cuda()
    fft_clone[:, :, :, :, 0] = torch.cos(pha_x.clone())
    fft_clone[:, :, :, :, 1] = torch.sin(pha_x.clone())

    # get the recomposed image: source content, target style
    pha_unwrap = torch.fft.ifft2(torch.complex(fft_clone[:, :, :, :, 0], fft_clone[:, :, :, :, 1]),
                                 dim=(-2, -1)).float()

    return pha_unwrap
