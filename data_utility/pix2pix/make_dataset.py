import argparse
import cv2
import glob
import os
import torch
from tqdm import tqdm
from torchvision import transforms
import torchvision.transforms.functional as F
from typing import OrderedDict
import re
import sys

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import annotate, load_image, predict
from huggingface_hub import hf_hub_download
from mobile_sam import sam_model_registry, SamPredictor


def load_gdino_model_hf(repo_id, filename, ckpt_config_filename, device):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    arg = SLConfig.fromfile(cache_config_file)
    model = build_model(arg)
    arg.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location=device)
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print('Model loaded from {} \n => {}'.format(cache_file, log))
    _ = model.eval()
    return model


def process_grounding_dino(model, img_path, res, text, device):
    # print('processing with grounding dino...')
    
    box_threshold = 0.1
    text_threshold = 0.3

    transform = transforms.Resize((res, res))
    image_source, image = load_image(img_path)
    image_source, image = cv2.resize(image_source, (res, res)), transform(image)
    
    boxes, logits, phrases = predict(
        model = model,
        image = image,
        caption = text,
        box_threshold = box_threshold,
        text_threshold = text_threshold
    )

    return image_source, boxes


def process_segment_anything(model, image_source, bb, device):
    # print('processing with segment anything...')

    model.set_image(image_source)
    H, W, _ = image_source.shape
    
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(bb) * torch.Tensor([W, H, W, H])
    transformed_boxes = model.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)

    masks, _, _ = model.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes,
        multimask_output=False,
    )
    
    return masks.cpu()


def save_hconcat_mask_and_masked_img(masks, image_source, boxes, tgt_path, data_id):
    img_size = image_source.shape[0]
    tgt_size = 256
    # print(len(boxes))
    # print(len(masks))

    for leaf_id in range(len(masks)):
        box = boxes[leaf_id].cpu().numpy() * img_size
        
        crop_size = int(max(box[2], box[3]) / 2) + 1
        x1, x2 = max(int(box[0]) - crop_size, 0), min(int(box[0]) + crop_size, img_size)
        y1, y2 = max(int(box[1]) - crop_size, 0), min(int(box[1]) + crop_size, img_size)
        # print(x2-x1)
        # print(y2-y1)

        if (x2 - x1) == (y2 - y1):
            leaf_mask = masks[leaf_id][0].to(torch.uint8) * 255
            leaf_mask_aligned = leaf_mask[max(y1, 0):min(y2, img_size), max(x1, 0):min(x2, img_size)]
            leaf_mask_aligned = F.resize(img=leaf_mask_aligned.unsqueeze(0), size=(tgt_size, tgt_size), interpolation=transforms.InterpolationMode.NEAREST, antialias=False)
            leaf_mask_aligned = cv2.cvtColor(leaf_mask_aligned.squeeze(0).numpy(), cv2.COLOR_GRAY2RGB)
            # leaf_mask_aligned = cv2.resize(leaf_mask_aligned, (tgt_size, tgt_size), cv2.INTER_NEAREST)
            leaf_mask_aligned_90 = cv2.rotate(leaf_mask_aligned, cv2.ROTATE_90_CLOCKWISE)
            leaf_mask_aligned_180 = cv2.rotate(leaf_mask_aligned, cv2.ROTATE_180)
            leaf_mask_aligned_270 = cv2.rotate(leaf_mask_aligned, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # leaf_masked = cv2.bitwise_and(image_source, image_source, mask=leaf_mask.numpy())
            # leaf_masked_aligned = leaf_masked[max(y1, 0):min(y2, img_size), max(x1, 0):min(x2, img_size)]
            leaf_masked_aligned = image_source[y1:y2, x1:x2]
            leaf_masked_aligned = cv2.resize(leaf_masked_aligned, (tgt_size, tgt_size))
            leaf_masked_aligned = cv2.cvtColor(leaf_masked_aligned, cv2.COLOR_RGB2BGR)
            leaf_masked_aligned_90 = cv2.rotate(leaf_masked_aligned, cv2.ROTATE_90_CLOCKWISE)
            leaf_masked_aligned_180 = cv2.rotate(leaf_masked_aligned, cv2.ROTATE_180)
            leaf_masked_aligned_270 = cv2.rotate(leaf_masked_aligned, cv2.ROTATE_90_COUNTERCLOCKWISE)

            tgt_img = cv2.hconcat([leaf_mask_aligned, leaf_masked_aligned])
            cv2.imwrite(f'{tgt_path}/{str(data_id).zfill(6)}.png', tgt_img)
            data_id = data_id + 1
            tgt_img_90 = cv2.hconcat([leaf_mask_aligned_90, leaf_masked_aligned_90])
            cv2.imwrite(f'{tgt_path}/{str(data_id).zfill(6)}.png', tgt_img_90)
            data_id = data_id + 1
            tgt_img_180 = cv2.hconcat([leaf_mask_aligned_180, leaf_masked_aligned_180])
            cv2.imwrite(f'{tgt_path}/{str(data_id).zfill(6)}.png', tgt_img_180)
            data_id = data_id + 1
            tgt_img_270 = cv2.hconcat([leaf_mask_aligned_270, leaf_masked_aligned_270])
            cv2.imwrite(f'{tgt_path}/{str(data_id).zfill(6)}.png', tgt_img_270)
            data_id = data_id + 1
        
    return data_id
    


def main(args, img_files, tgt_path):
    # environment variables
    # img_path = './real_tree/aucuba/train/00000.png' # later, apapt args and glob
    device = torch.device(f'cuda:{str(args.device_id)}')
    data_id = args.resume

    # preparing grounding_dino model
    ckpt_repo_id = 'ShilongLiu/GroundingDINO'
    ckpt_filename = 'groundingdino_swinb_cogcoor.pth'
    ckpt_config_filename = 'GroundingDINO_SwinB.cfg.py'
    groundingdino_model = load_gdino_model_hf(ckpt_repo_id, ckpt_filename, ckpt_config_filename, device)
    
    # preparing mobile segment anything model
    model_type = 'vit_t'
    sam_checkpoint = './mobile_sam.pt'
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    sam_predictor = SamPredictor(sam)

    max_num = min(len(img_files), 100)

    with tqdm(range(0, max_num), file=sys.stdout, ncols=80) as pbar:
        for i in pbar:
            # process with grouding dino
            image_source, boxes = process_grounding_dino(model=groundingdino_model, img_path=img_files[i], res=args.img_res, text=args.text_prompt, device=device)
            # process with segment anything
            masks = process_segment_anything(model=sam_predictor, image_source=image_source, bb=boxes, device=device)
            # hconcat leaf_mask and leaf_masked and save the file as mode we specified
            data_id = save_hconcat_mask_and_masked_img(masks=masks, image_source=image_source, boxes=boxes, tgt_path=tgt_path, data_id=data_id)
            pbar.set_postfix(
                OrderedDict(
                    number_of_data=data_id,
                    image_shape=str(image_source.shape),
                    mask_shape=str(masks.shape)
                )
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='making leaf mask and masked img using Grounded-SAM')

    parser.add_argument('--img_res', default=1024, type=int, help='the resolution of input images')
    parser.add_argument('--text_prompt', type=str, required=True, help='object name you want to segment')
    parser.add_argument('--device_id', default=0, type=int, help='specify gpu id you want to use')
    parser.add_argument('--mode', default='train', help='[train | test]')
    parser.add_argument('--species', type=str, required=True, help='plant species name')
    # parser.add_argument('--branch', action='store_true')
    parser.add_argument('--resume', type=int, default=0, help='specify number if you already have detected img')
    parser.add_argument('--src_dir', type=str, required=True, help='Data Directory')

    args = parser.parse_args()

    all_files = glob.glob(f'{args.src_dir}/*')
    src_imgs = [f for f in all_files if re.search(r'\.(jpg|jpeg|png|bmp|tiff)$', f, re.IGNORECASE)]
    tgt_path = os.path.join(args.src_dir, args.species, args.mode)
    os.makedirs(tgt_path, exist_ok=True)

    

    main(args, img_files=src_imgs, tgt_path=tgt_path)





