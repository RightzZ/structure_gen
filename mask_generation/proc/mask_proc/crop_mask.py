import cv2
import glob
import numpy as np
import os
from tqdm import tqdm
import argparse


def crop_mask(img_path, save_dir):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_size = img.shape[0]

    # separate 0 or 255
    _, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    # calculate the coordinate the value of which is 255
    (y, x) = np.where(img % 2 == 1)

    x, y = np.array(x), np.array(y)
    if len(x) == 0 or len(y) == 0:
        return

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    h, w = y_max - y_min, x_max - x_min

    c_x, c_y = int((x_min + x_max) / 2), int((y_min + y_max) / 2)
    crop_size = int(max(h, w) / 2) + 1

    # t:top, b:bottom, l:left, r:right
    t, b, l, r = max(c_y - crop_size, 0), min(c_y + crop_size + 1, img_size), max(c_x - crop_size, 0), min(
        c_x + crop_size + 1, img_size)

    # crop the image and save it with its coordinate and orignal size (npy)
    cropped_img = img[t:b, l:r]

    os.makedirs(save_dir, exist_ok=True)
    img_name = os.path.basename(img_path)
    cv2.imwrite(os.path.join(save_dir, img_name), cropped_img)


    bb = np.array([t, b, l, r, cropped_img.shape[0]])
    npy_name = os.path.splitext(img_name)[0] + ".npy"
    np.save(os.path.join(save_dir, npy_name), bb)


def process_data_dir(data_dir):
    mask_leaf_dir = os.path.join(data_dir, 'mask_leaf')
    cropped_dir = os.path.join(data_dir, 'cropped_mask')

    if not os.path.exists(mask_leaf_dir):
        print(f"❌ mask_leaf not found in {data_dir}")
        return

    for sub_root, dirs, files in os.walk(mask_leaf_dir):
        if not any(f.endswith('.png') for f in files):
            continue

        rel_path = os.path.relpath(sub_root, mask_leaf_dir)
        save_dir = os.path.join(cropped_dir, rel_path)

        img_paths = sorted(glob.glob(os.path.join(sub_root, 'leaf_*.png')))
        for img_path in img_paths:
            crop_mask(img_path, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    args = parser.parse_args()

    process_data_dir(args.data_dir)