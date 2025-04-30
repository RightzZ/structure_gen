# Model Training
This step allows you to generate dataset from real plant pictures, and train a Pix2Pix model with the generated dataset.

## Preparation
Please ensure that all your images are square and have the same resolution.
Also, please ensure that your images are correctly numbered.

```
<plant pictures directory>
|---000000.png # e.g. The resolution is 1024x1024
|---000001.png
|---...
```
## Usage

### Mask Generation
```
cd data_utility/pix2pix
python make_dataset.py \
--img_res <Resolution of images> \
--text_prompt <Usually 'leaf' or 'branch'> \
--device_id <The number of GPU you want to use> \
--mode <train or test> \
--species <The name of the species(whatever you want to name) \
--src_dir <Directory of the real plant pictures>
```

As results, you may find `train` and `test` folders in the directory.

### Training Model

Actually, this step follows [Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md) 
official project. We just use the recommended parameters like below
```
cd image_generation/pix2pix
python train.py \
--dataroot <Path to dataset> \
--name <Species name> \
--gpu_ids <GPU you want to use, you can write multiple gpus>
--model pix2pix \
--netG unet_256 \
--direction AtoB \
--lambda_L1 100 \
--norm batch \
--display_id 0 \ 
--preprocess none \ 
--no_flip \
--no_html \
```

After training, you can find results in `/image_generation/pix2pix/checkpoints/<name>`

You can finish these steps in GUI, but attention: in the step of Training Model, we only provides two parameters can be
modified: Dataset Directory(dataroot) and Model Name(name). So if you want to customize all parameters, please use CLI.