# saliency.py
import cv2
import numpy as np
from skimage.color import rgb2lab, gray2rgb
from skimage.io import imread
from skimage.transform import rescale


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



def apply_center_surround(img, sigma_center=1.0, sigma_surround=3.0, strength=1.0):

    # Ensure image is float for calculation
    img_float = img.astype(np.float32) / 255.0
    
    # Calculate Center (Excitatory)
    center = cv2.GaussianBlur(img_float, (0, 0), sigma_center)
    
    # Calculate Surround (Inhibitory)
    surround = cv2.GaussianBlur(img_float, (0, 0), sigma_surround)

    dog = center - surround
    

    cs_img = dog * strength

    return cs_img

def spectral_residual_saliency_with_cs(img, width=128):

    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    img_cs = apply_center_surround(img_gray, sigma_center=2, sigma_surround=8)
   
    img_cs_norm = cv2.normalize(img_cs, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h, w = img_gray.shape
    new_h = int(width * h / w)
    img_resized = cv2.resize(img_cs_norm, (width, new_h)) # Dùng ảnh đã lọc CS

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


def compute_bms_saliency_with_cs(img):
    img_lab = rgb2lab(img)
    
    for i in range(3): 
        channel = img_lab[:,:,i]
        ch_norm = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
        
        cs_response = apply_center_surround(ch_norm, sigma_center=3, sigma_surround=10)
        

        img_lab[:,:,i] += cs_response * 5.0
    # ---------------------------------------------

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

    attn_map = cv2.GaussianBlur(attn_map, (0,0), 3)
    
    norm = np.sqrt((attn_map**2).sum())
    attn_map /= (norm + 1e-8)
    attn_map /= attn_map.max() / 255

    return attn_map.astype(np.uint8)



import cv2
import numpy as np
from skimage.color import rgb2lab, gray2rgb
from skimage.io import imread
from skimage.transform import rescale


def apply_center_surround_filter(img, sigma_center=1.0, sigma_surround=3.0, strength=1.0):
    """
    Apply Center-Surround filter (DoG) to enhance edges and local contrast.
    This mimics retinal ganglion cells.
    
    Input:
        img: Grayscale or single channel image (numpy array), uint8 or float.
    Output:
        cs_img: Filtered image, normalized to [0, 1].
    """
    if img.dtype != np.float32:
        img_float = img.astype(np.float32) / 255.0
    else:
        img_float = img

    # Center
    center = cv2.GaussianBlur(img_float, (0, 0), sigma_center)
    
    # Surround
    surround = cv2.GaussianBlur(img_float, (0, 0), sigma_surround)
    
    # DoG = Center - Surround
    dog = (center - surround) * strength
    cs_img = np.abs(dog)
    
    return np.clip(cs_img, 0, 1)


def apply_hollow_filling(saliency_map, kernel_size=5):
    """
    Apply Morphological Closing and Filling to fill hollow regions
    typically caused by edge-only data (e.g., event cameras).
    
    Input:
        saliency_map: Saliency map (float 0-1 or uint8 0-255)
    Output:
        filled_map: Processed map with filled regions.
    """
    if saliency_map.dtype != np.uint8:
        s_uint8 = (saliency_map * 255).astype(np.uint8)
    else:
        s_uint8 = saliency_map.copy()

    # 1. Morphological Closing: Dilate then Erode to close gaps in contours
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(s_uint8, cv2.MORPH_CLOSE, kernel)
    
    # 2. Contour Filling: Find contours and fill them
    # Threshold to get binary mask of potential objects
    # Adjust threshold (e.g., 50) based on your data sensitivity
    _, binary = cv2.threshold(closed, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filled = np.zeros_like(s_uint8)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    
    result_uint8 = cv2.max(s_uint8, filled)
    
    if saliency_map.dtype != np.uint8:
        return result_uint8.astype(np.float32) / 255.0
    return result_uint8


def spectral_residual_saliency_v2(img, width=128, enable_cs=True, enable_fill=True):
    """
    Compute saliency map using Spectral Residual method with optional enhancements.
    
    Args:
        img: RGB image (numpy array).
        width: Calculation width.
        enable_cs: Enable Center-Surround filtering (pre-processing).
        enable_fill: Enable morphological filling for hollow objects (post-processing).
    """
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img

    if enable_cs:

        cs_response = apply_center_surround_filter(img_gray, sigma_center=2, sigma_surround=8, strength=2.0)
 
        img_input = (cs_response * 255).astype(np.uint8)
    else:
        img_input = img_gray

    h, w = img_input.shape
    new_h = int(width * h / w)
    img_resized = cv2.resize(img_input, (width, new_h))

    # Fourier Transform
    c = cv2.dft(np.float32(img_resized), flags=cv2.DFT_COMPLEX_OUTPUT)
    mag = np.sqrt(c[:,:,0]**2 + c[:,:,1]**2)

    # Spectral Residual Calculation
    log_mag = np.log(mag + 1e-8)
    avg_log_mag = cv2.boxFilter(log_mag, -1, (3,3))
    spectral_residual = np.exp(log_mag - avg_log_mag)

    c[:,:,0] = c[:,:,0] * spectral_residual / (mag + 1e-8)
    c[:,:,1] = c[:,:,1] * spectral_residual / (mag + 1e-8)

    # Inverse Fourier
    c_inv = cv2.dft(c, flags=(cv2.DFT_INVERSE | cv2.DFT_SCALE))
    sal_map = c_inv[:,:,0]**2 + c_inv[:,:,1]**2

    sal_map = cv2.GaussianBlur(sal_map, (9,9), 3)
    sal_map = cv2.normalize(sal_map, None, 0., 1., cv2.NORM_MINMAX)

    if enable_fill:
        sal_map = apply_hollow_filling(sal_map, kernel_size=5)

    return sal_map



N_THRESHOLDS = 10

def _activate_boolean_map_helper(bool_map):
    """Internal helper for BMS to process boolean maps."""
    activation = np.array(bool_map, dtype=np.uint8)
    mask_shape = (bool_map.shape[0] + 2, bool_map.shape[1] + 2)
    ffill_mask = np.zeros(mask_shape, dtype=np.uint8)

    for j in range(activation.shape[1]):
        if activation[0, j]:
            cv2.floodFill(activation, ffill_mask, (j, 0), 0)
        if activation[activation.shape[0]-1, j]:
            cv2.floodFill(activation, ffill_mask, (j, activation.shape[0]-1), 0)
    
    for i in range(activation.shape[0]):
        if activation[i, 0]:
            cv2.floodFill(activation, ffill_mask, (0, i), 0)
        if activation[i, activation.shape[1]-1]:
            cv2.floodFill(activation, ffill_mask, (activation.shape[1]-1, i), 0)
            
    return activation


def compute_bms_saliency_v2(img, enable_cs=True, enable_fill=True):
    """
    Compute Boolean Map Saliency (BMS) with optional enhancements.
    
    Args:
        img: RGB image (numpy array).
        enable_cs: Enable Center-Surround filtering on Lab channels.
        enable_fill: Enable morphological filling for hollow objects.
    """
    img_lab = rgb2lab(img)

    if enable_cs:

        for i in range(3):

            ch_norm = cv2.normalize(img_lab[:,:,i], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Apply filter
            cs_resp = apply_center_surround_filter(ch_norm, sigma_center=2, sigma_surround=6, strength=1.5)
            

            img_lab[:,:,i] += cs_resp * 10.0 

    img_lab -= img_lab.min()
    img_lab /= (img_lab.max() + 1e-8)
    
    thresholds = np.arange(0, 1, 1.0 / N_THRESHOLDS)[1:]
    bool_maps = []

    for thresh in thresholds:
        img_lab_T = img_lab.transpose(2,0,1)
        img_thresh = (img_lab_T > thresh)
        bool_maps.extend(list(img_thresh))

    attn_map = np.zeros(img_lab.shape[:2], dtype=np.float32)
    for bmap in bool_maps:
        attn_map += _activate_boolean_map_helper(bmap)
    
    attn_map /= N_THRESHOLDS

    attn_map = cv2.GaussianBlur(attn_map, (0,0), 3)
    
    norm_val = np.sqrt((attn_map**2).sum())
    attn_map /= (norm_val + 1e-8)
    
    max_val = attn_map.max()
    if max_val > 0:
        attn_map /= max_val
        
    attn_map_uint8 = (attn_map * 255).astype(np.uint8)

    if enable_fill:
        attn_map_uint8 = apply_hollow_filling(attn_map_uint8, kernel_size=7)

    return attn_map_uint8

