# saliency.py
import cv2
import numpy as np
from skimage.color import rgb2lab, gray2rgb
from skimage.io import imread
from skimage.transform import rescale

# ------------------------
# Spectral Residual Saliency
# ------------------------
def spectral_residual_saliency(img, width=128):
    """
    Compute saliency map using Spectral Residual method.
    Input:
        img: RGB image as numpy array
        width: resize width for computation
    Output:
        saliency_map: float32 numpy array normalized [0,1]
    """
    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    h, w = img_gray.shape
    new_h = int(width * h / w)
    img_resized = cv2.resize(img_gray, (width, new_h))

    # Fourier Transform
    c = cv2.dft(np.float32(img_resized), flags=cv2.DFT_COMPLEX_OUTPUT)
    mag = np.sqrt(c[:,:,0]**2 + c[:,:,1]**2)

    # Spectral Residual
    log_mag = np.log(mag + 1e-8)
    avg_log_mag = cv2.boxFilter(log_mag, -1, (3,3))
    spectral_residual = np.exp(log_mag - avg_log_mag)

    # Apply residual to original complex
    c[:,:,0] = c[:,:,0] * spectral_residual / (mag + 1e-8)
    c[:,:,1] = c[:,:,1] * spectral_residual / (mag + 1e-8)

    # Inverse Fourier
    c_inv = cv2.dft(c, flags=(cv2.DFT_INVERSE | cv2.DFT_SCALE))
    sal_map = c_inv[:,:,0]**2 + c_inv[:,:,1]**2

    # Gaussian smoothing and normalization
    sal_map = cv2.GaussianBlur(sal_map, (9,9), 3)
    sal_map = cv2.normalize(sal_map, None, 0., 1., cv2.NORM_MINMAX)

    return sal_map


# ------------------------
# BMS Saliency
# ------------------------
N_THRESHOLDS = 10

def activate_boolean_map(bool_map):
    activation = np.array(bool_map, dtype=np.uint8)
    mask_shape = (bool_map.shape[0] + 2, bool_map.shape[1] + 2)
    ffill_mask = np.zeros(mask_shape, dtype=np.uint8)

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
    """
    Compute Boolean Map Saliency (BMS) for RGB image.
    Input:
        img: RGB image as numpy array
    Output:
        attn_map: uint8 saliency map [0,255]
    """
    img_lab = rgb2lab(img)
    img_lab -= img_lab.min()
    img_lab /= img_lab.max()
    thresholds = np.arange(0, 1, 1.0 / N_THRESHOLDS)[1:]

    bool_maps = []
    for thresh in thresholds:
        img_lab_T = img_lab.transpose(2,0,1)
        img_thresh = (img_lab_T > thresh)
        bool_maps.extend(list(img_thresh))

    attn_map = np.zeros(img_lab.shape[:2], dtype=np.float32)
    for bmap in bool_maps:
        attn_map += activate_boolean_map(bmap)
    attn_map /= N_THRESHOLDS

    # Gaussian smoothing
    attn_map = cv2.GaussianBlur(attn_map, (0,0), 3)
    
    # Normalize to [0,255]
    norm = np.sqrt((attn_map**2).sum())
    attn_map /= (norm + 1e-8)
    attn_map /= attn_map.max() / 255

    return attn_map.astype(np.uint8)

def load_rgb_image(img_path, max_dim=320):
    img = imread(img_path)
    if img.ndim == 2:
        img = gray2rgb(img)
    elif img.shape[2] == 4:
        img = img[:,:,:3]
    upper_dim = max(img.shape[:2])
    if upper_dim > max_dim:
        img = rescale(img, max_dim/float(upper_dim), order=3, anti_aliasing=True, channel_axis=-1)
        img = (img * 255).astype(np.uint8)
    return img