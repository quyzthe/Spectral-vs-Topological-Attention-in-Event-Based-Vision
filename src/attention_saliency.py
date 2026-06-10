# attention_saliency.py
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.color import rgb2lab

# -----------------------------
# Device
# -----------------------------
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class AttentionModuleLevel(nn.Module):
    def __init__(self, num_ori=4):
        super().__init__()
        self.num_ori = num_ori

        self.lif = nn.Identity()

    def forward(self, inp):

        x = inp[:,0:1]
        y = inp[:,1:2]
        out = (x.abs() + y.abs()) / 2  # [B,1,H,W]
        return out

class AttentionModule(nn.Module):
    def __init__(self, pyramid_levels=1, num_ori=4):
        super().__init__()
        self.levels = nn.ModuleList([AttentionModuleLevel(num_ori=num_ori) for _ in range(pyramid_levels)])

    def forward(self, inp):

        h, w = inp.shape[-2:]
        scale = 128 / max(h, w)
        new_h, new_w = max(1,int(h*scale)), max(1,int(w*scale))
        inp_small = F.interpolate(inp, size=(new_h,new_w), mode='bilinear', align_corners=False)

        out = 0
        for level in self.levels:
            level_out = level(inp_small)
            out += F.interpolate(level_out, size=(h,w), mode='bilinear', align_corners=False)

        out = out / (out.max() + 1e-8)
        return out.squeeze().cpu().detach().numpy()

# Global attention net
attention_net = AttentionModule().to(device)

# -----------------------------
# Spectral Residual Saliency
# -----------------------------
def spectral_residual_saliency(img, width=128):
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img_gray.shape
    new_h = int(width * h / w)
    img_resized = cv2.resize(img_gray, (width, new_h))

    c = cv2.dft(np.float32(img_resized), flags=cv2.DFT_COMPLEX_OUTPUT)
    mag = np.sqrt(c[:,:,0]**2 + c[:,:,1]**2)

    log_mag = np.log(mag + 1e-8)
    avg_log_mag = cv2.boxFilter(log_mag, -1, (3,3))
    spectral_residual = np.exp(log_mag - avg_log_mag)

    c[:,:,0] = c[:,:,0] * spectral_residual / (mag + 1e-8)
    c[:,:,1] = c[:,:,1] * spectral_residual / (mag + 1e-8)

    c_inv = cv2.dft(c, flags=(cv2.DFT_INVERSE | cv2.DFT_SCALE))
    sal_map = c_inv[:,:,0]**2 + c_inv[:,:,1]**2
    sal_map = cv2.GaussianBlur(sal_map, (9,9), 3)
    sal_map = cv2.normalize(sal_map, None, 0., 1., cv2.NORM_MINMAX)
    return sal_map.astype(np.float32)

def spectral_residual_saliency_nn(img, width=128):
    img_small = cv2.resize(img, (128, int(128*img.shape[0]/img.shape[1])))
    gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_tensor = torch.tensor(np.stack([grad_x, grad_y], axis=0)[np.newaxis,...], dtype=torch.float32).to(device)/255.0

    attn_map_small = attention_net(grad_tensor)
    attn_map = cv2.resize(attn_map_small, (img.shape[1], img.shape[0]))
    img_weighted = (img.astype(np.float32) * attn_map[..., np.newaxis]).astype(np.uint8)
    return spectral_residual_saliency(img_weighted, width=width)

# -----------------------------
# BMS Saliency
# -----------------------------
N_THRESHOLDS = 10
def activate_boolean_map(bool_map):
    activation = np.array(bool_map, dtype=np.uint8)
    ffill_mask = np.zeros((activation.shape[0]+2, activation.shape[1]+2), dtype=np.uint8)
    for i in range(activation.shape[0]):
        for j in [0, activation.shape[1]-1]:
            if activation[i,j]:
                cv2.floodFill(activation, ffill_mask, (j,i), 0)
    for i in [0, activation.shape[0]-1]:
        for j in range(activation.shape[1]):
            if activation[i,j]:
                cv2.floodFill(activation, ffill_mask, (j,i), 0)
    return activation

def compute_bms_saliency(img):
    img_lab = rgb2lab(img)
    img_lab -= img_lab.min()
    img_lab /= img_lab.max()
    thresholds = np.arange(0,1,1.0/N_THRESHOLDS)[1:]

    bool_maps = []
    img_lab_T = img_lab.transpose(2,0,1)
    for thresh in thresholds:
        bool_maps.extend(list(img_lab_T > thresh))

    attn_map = np.zeros(img_lab.shape[:2], dtype=np.float32)
    for bmap in bool_maps:
        attn_map += activate_boolean_map(bmap)
    attn_map /= N_THRESHOLDS
    attn_map = cv2.GaussianBlur(attn_map, (0,0), 3)
    norm = np.sqrt((attn_map**2).sum())
    attn_map /= (norm + 1e-8)
    attn_map /= attn_map.max() / 255
    return attn_map.astype(np.uint8)

def compute_bms_saliency_nn(img):
    img_small = cv2.resize(img, (128, int(128*img.shape[0]/img.shape[1])))
    gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_tensor = torch.tensor(np.stack([grad_x, grad_y], axis=0)[np.newaxis,...], dtype=torch.float32).to(device)/255.0

    attn_map_small = attention_net(grad_tensor)
    attn_map = cv2.resize(attn_map_small, (img.shape[1], img.shape[0]))
    img_weighted = (img.astype(np.float32) * attn_map[..., np.newaxis]).astype(np.uint8)
    return compute_bms_saliency(img_weighted)
