import os
import re
import numpy as np
import cv2
import matplotlib.pyplot as plt


def build_event_frame(x, y, p, H, W):
    frame = np.zeros((H, W), dtype=np.int32)
    np.add.at(frame, (y, x), p)
    return frame


def extract_frame_number(filename):
    """
    Extract 3-digit frame index from filename.
    Example:
        Diving-Side_001_013.png -> 013
    """
    match = re.search(r'_(\d+)\.', filename)
    if match:
        return match.group(1).zfill(3)
    return None


def generate_event_frames(
    x, y, p, t,
    rgb_dir,
    BIN_US=100_000_000,
    n=1,
    plot=False
):

    image_dir = os.path.join(rgb_dir, "images")
    fixation_dir = os.path.join(rgb_dir, "fixation")
    maps_dir = os.path.join(rgb_dir, "maps")

    rgb_files = sorted([f for f in os.listdir(image_dir)
                        if f.lower().endswith((".png", ".jpg", ".jpeg"))])

    fixation_files = sorted(os.listdir(fixation_dir))
    maps_files = sorted(os.listdir(maps_dir))

    # -------- Create lookup dict by frame index --------
    fixation_dict = {}
    for f in fixation_files:
        idx = extract_frame_number(f)
        if idx:
            fixation_dict[idx] = f

    maps_dict = {}
    for f in maps_files:
        idx = extract_frame_number(f)
        if idx:
            maps_dict[idx] = f

    t_start = 0
    t_end = t.max()
    bins = np.arange(t_start, t_end + BIN_US, BIN_US)

    H = y.max() + 1
    W = x.max() + 1

    event_frames = []
    rgb_selected = []
    fixation_selected = []
    maps_selected = []

    i = 0
    while i < len(bins) - 1:

        t0 = bins[i]
        t1 = bins[min(i+n, len(bins)-1)]

        mask = (t >= t0) & (t < t1)
        if mask.sum() < 20:
            i += n
            continue

        event_frame = build_event_frame(x[mask], y[mask], p[mask], H, W)
        event_frames.append(event_frame)

        rgb_idx = min(i, len(rgb_files)-1)
        rgb_file = rgb_files[rgb_idx]

        # -------- Extract frame number from RGB --------
        frame_number = extract_frame_number(rgb_file)

        # -------- Load RGB --------
        rgb_path = os.path.join(image_dir, rgb_file)
        rgb_img = cv2.imread(rgb_path)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_selected.append(rgb_img)

        # -------- Load Fixation --------
        fix_img = None
        if frame_number in fixation_dict:
            fix_path = os.path.join(fixation_dir, fixation_dict[frame_number])
            fix_img = cv2.imread(fix_path, 0)

        fixation_selected.append(fix_img)

        # -------- Load GT Map --------
        sal_map = None
        if frame_number in maps_dict:
            map_path = os.path.join(maps_dir, maps_dict[frame_number])
            sal_map = cv2.imread(map_path, 0)

        maps_selected.append(sal_map)

        # -------- Plot --------
        if plot:
            fig, axs = plt.subplots(1, 4, figsize=(16, 4))

            axs[0].imshow(event_frame, cmap="gray")
            axs[0].set_title("Event")

            axs[1].imshow(rgb_img)
            axs[1].set_title("RGB")

            if fix_img is not None:
                axs[2].imshow(fix_img, cmap="gray")
                axs[2].set_title("Fixation")

            if sal_map is not None:
                axs[3].imshow(sal_map, cmap="gray")
                axs[3].set_title("GT Map")

            for ax in axs:
                ax.axis("off")

            plt.tight_layout()
            plt.show()

        i += n

    return event_frames, rgb_selected, fixation_selected, maps_selected
