import cv2
import glob
import argparse
import os
from tqdm import tqdm


def bitwise_not_img(b_path, root_tgt_path, text, render_num, base_dir):
    for i in range(render_num):
        render_dir = f'render_{str(i).zfill(4)}'
        tgt_dir = f'{root_tgt_path}/{render_dir}'
        os.mkdir(tgt_dir)
        img_path_list = sorted(glob.glob(f'{base_dir}/{b_path}/{render_dir}/{text}*.png'))
        for img_path in img_path_list:
            b_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            inverted_b_img = cv2.bitwise_not(b_img)
            _, inverted_b_img = cv2.threshold(inverted_b_img, 200, 255, cv2.THRESH_BINARY)
            cv2.imwrite(f'{tgt_dir}/{os.path.basename(img_path)}', inverted_b_img)


def main(base_binary_dir, text):
    binary_img_list = sorted(os.listdir(base_binary_dir))
    for b_path in tqdm(binary_img_list):
        render_num = len(os.listdir(os.path.join(base_binary_dir, b_path)))
        root_tgt_path = os.path.join(base_binary_dir.replace("binary", f"mask_{text}"), b_path)
        os.makedirs(root_tgt_path, exist_ok=True)
        bitwise_not_img(b_path, root_tgt_path, text, render_num, base_binary_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    args = parser.parse_args()

    data_path = args.data_dir
    text_list = ['leaf', 'branch']

    binary_dir = os.path.join(data_path, 'binary')

    if os.path.isdir(binary_dir):
        for text in text_list:
            main(binary_dir, text)

