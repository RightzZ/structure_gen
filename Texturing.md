# Texturing
This step allows you to generate a series of realistic plant images by combining the previously generated plant masks
with the pre-trained plant models.

## Preparation
Please ensure that you have completed training the Pix2Pix model with real plant images, and that the plant masks have
been properly generated and processed in the preceding steps.

We have already trained a model for the hawthorn (車輪梅) species, which you can download and use.
Please put the model in `mask_generation/proc/texturing/pix2pix/ckpt/hawthorn`. Besides, if you want to use your own .pth
model, please put in the same directory with the species name.(e.g. `ckpt/komatsuna`)

Also, you should check your mask data. According to the earlier processing steps, your mask directory should include at
least the following subfolders:

```
<mask data directory>
|---<species>
|   |---img # Overall mask
|   |---cropped_mask # Mask for leaves
|   |---mask_branch # Mask for branches
```

Then, for background image, you should prepare background images (e.g. natural environment) as the same resolution as
masks. You should make sure that your background images are placed in the directory with sequential numbering.
```
<background image directory>
|---00000.png
|---00001.png
|---...
```
We provide a collection of background images that you can download as needed. We recommend you to put them in 
`data_utility/src/background`, for smoothly using GUI.

## Usage
```
cd mask_generation/proc/texturing
python texturing.py \
--gpu_id <GPU you want to use> \
--species <Species name, corresponds to /ckpt/(species)> \
--data_dir <Directory of masks> \
--bg_dir <Background image>
--img_res <Resolution of the image>
```
You can also use GUI to run this step.