import argparse
import cv2
import glob
import numpy as np
import os
import subprocess


def bit_reverse(reverse_src_path, reverse_tgt_path, text):
    plant_idx_list = os.listdir(reverse_src_path)
    for plant_idx in plant_idx_list:
        if not os.path.exists(f'{reverse_tgt_path}/{plant_idx}'):
            os.mkdir(f'{reverse_tgt_path}/{plant_idx}')
        render_dirs = sorted([d for d in os.listdir(f'{reverse_src_path}/{plant_idx}') if d.startswith("render_")])
        for render_dir in render_dirs:
            tgt_path = f'{reverse_tgt_path}/{plant_idx}/{render_dir}'
            if not os.path.exists(tgt_path):
                os.mkdir(tgt_path)
            bit_img_path_list = sorted(glob.glob(f'{reverse_src_path}/{plant_idx}/{render_dir}/{text}*.png'))
            for bit_img_path in bit_img_path_list:
                bit_img = cv2.imread(bit_img_path, cv2.IMREAD_GRAYSCALE)
                mask_img = cv2.bitwise_not(bit_img)
                _, mask_img = cv2.threshold(mask_img, 200, 255, cv2.THRESH_BINARY)
                cv2.imwrite(f'{tgt_path}/{os.path.basename(bit_img_path)}', mask_img)


def crop(crop_src_path, crop_tgt_path):
    plant_idx_list = os.listdir(crop_src_path)
    for plant_idx in plant_idx_list:
        if not os.path.exists(f'{crop_tgt_path}/{plant_idx}'):
            os.mkdir(f'{crop_tgt_path}/{plant_idx}')
        render_dirs = sorted([d for d in os.listdir(f'{crop_src_path}/{plant_idx}') if d.startswith("render_")])
        for render_dir in render_dirs:
            tgt_path = f'{crop_tgt_path}/{plant_idx}/{render_dir}'
            if not os.path.exists(tgt_path):
                os.mkdir(tgt_path)
            mask_img_path_list = sorted(glob.glob(f'{crop_src_path}/{plant_idx}/{render_dir}/leaf*.png'))
            for mask_img_path in mask_img_path_list:
                crop_mask(mask_img_path, tgt_path)


def crop_mask(mask_img_path, tgt_path):
    mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
    mask_img_size = mask_img.shape[0]

    (y, x) = np.where(mask_img % 2 == 1)
    # x, y = np.array(x), np.array(y)
    if len(x) == 0 or len(y) == 0:
        return

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    h, w = y_max - y_min, x_max - x_min

    c_x, c_y = int((x_min + x_max) / 2), int((y_min + y_max) / 2)
    crop_size = int(max(h, w) / 2) + 1

    t, b, l, r = max(c_y - crop_size, 0), min(c_y + crop_size + 1, mask_img_size), max(c_x - crop_size, 0), min(
        c_x + crop_size + 1, mask_img_size)

    cropped_mask = mask_img[t:b, l:r]

    cv2.imwrite(f'{tgt_path}/{os.path.basename(mask_img_path)}', cropped_mask)
    bb = np.array([t, b, l, r, cropped_mask.shape[0]])
    np.save(f'{tgt_path}/{os.path.splitext(os.path.basename(mask_img_path))[0]}.npy', bb)


def count_max_leaf_num(count_src_path):
    plant_idx_list = os.listdir(count_src_path)
    max_num = 0

    for plant_idx in plant_idx_list:
        mask_path = f'{count_src_path}/{plant_idx}'
        render_dirs = sorted([d for d in os.listdir(mask_path) if d.startswith("render_")])
        for render_dir in render_dirs:
            leaf_num = len(glob.glob(f'{mask_path}/{render_dir}/leaf*.png'))
            if leaf_num > max_num:
                max_num = leaf_num

    return max_num


def make_gt_data(gt_src_path, output_dir, img_shape, max_num, species):
    save_dir = os.path.join(output_dir, f'p_{species}', 'train', 'segment')
    os.makedirs(save_dir, exist_ok=True)

    plant_idx_list = os.listdir(gt_src_path)
    img_id = 1

    for plant_idx in plant_idx_list:
        render_times_path = f'{gt_src_path}/{plant_idx}'
        render_dirs = sorted([d for d in os.listdir(render_times_path) if d.startswith("render_")])
        for render_dir in render_dirs:
            mask_img_path_list = glob.glob(f'{render_times_path}/{render_dir}/leaf*.png')

            gt_img = np.zeros(img_shape, dtype=np.uint8)
            h = 0
            h_add = 180 / max_num

            for mask_img_path in mask_img_path_list:
                mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
                if np.sum(mask_img) == 0:  # if the leaf is perfectly occluded
                    continue
                (y, x) = np.where(mask_img % 2 == 1)
                mask_img = np.repeat(np.expand_dims(mask_img, 2), 3, 2)
                mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2HSV)
                for k in range(len(x)):
                    mask_img[y[k]][x[k]] = (h, 255, 255)
                paste_img = cv2.cvtColor(mask_img, cv2.COLOR_HSV2RGB)
                gt_img = cv2.bitwise_or(gt_img, paste_img)
                h = h + h_add

            save_path = os.path.join(save_dir, f'{str(img_id).zfill(6)}.png')
            cv2.imwrite(save_path, gt_img)
            img_id += 1


def main(args):

    root_path = f'{args.data_dir}/{args.species}'
    shape = args.image_shape

    img_shape = (shape, shape, 3)

    print("📦 Process1: binary image => mask image")
    reverse_src_path = f'{root_path}/binary'
    if args.species in ['hawthorn', 'amodal', 'plant'] :
        for text in ['leaf', 'branch']:
            reverse_tgt_path = f'{root_path}/mask_{text}'
            if not os.path.exists(reverse_tgt_path):
                os.mkdir(reverse_tgt_path)
            bit_reverse(reverse_src_path, reverse_tgt_path, text)
    else:
        text = 'leaf'
        reverse_tgt_path = f'{root_path}/mask_{text}'
        if not os.path.exists(reverse_tgt_path):
            os.mkdir(reverse_tgt_path)
        bit_reverse(reverse_src_path, reverse_tgt_path, text)

    print("📦 Process2: cropping mask image")
    crop_src_path = f'{root_path}/mask_leaf'
    crop_tgt_path = f'{root_path}/cropped_mask'
    if not os.path.exists(crop_tgt_path):
        os.mkdir(crop_tgt_path)
    crop(crop_src_path, crop_tgt_path)

    print("📦 Process3: create GT mask image")
    count_src_path = crop_tgt_path
    gt_src_path = crop_src_path

    max_num = count_max_leaf_num(count_src_path)
    # print(max_num)
    # # max_num = 7
    make_gt_data(gt_src_path, args.output_dir, img_shape, max_num, args.species)


def get_args():
    parser = argparse.ArgumentParser(description='proc data made by Blender')
    parser.add_argument("--data_dir", type=str, required=True, help='directory where you put your generated data')
    parser.add_argument('--species', type=str, required=True, help='plant species name')
    parser.add_argument("--output_dir", type=str, required=True, help='output directory')
    parser.add_argument("--image_shape", type=int, required=True, help='the shape of the image in dataset')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)