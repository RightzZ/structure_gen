# Data Management

This step allows you to prganize the generated data and process the masks.

## Usage
### Dataset Classification

In the step Mask Generation, you should have generated a series of datasets.

```
<output directory>
|---data_00001_ara
|---data_00002_komatsuna
|---data_00003_plant
|---...
|---data_xxxxx_xxx
```

You can arrange the generated data into categories based on plant species.

```
<other directory>
|---amodal
|   |---binary
|   |---img
|   |---json
|   |---model
|---ara
|---komatsuna
|---plant
```

By using
```
cd data_utility/src/syn_plant
python data_classification.py \
--src dir <path/to/original/directory> \
--dst dir <path/to/destination>
```

### Mask&Dataset Processing

The masks produced in the earlier steps feature a white background and have not been further processed.  
In this step, the masks will be inverted, and each individual leaf mask will be cropped to a suitable size.  
Furthermore, the operation will produce ground truth masks, which can be used for a variety of applications.  
Note: The processing of the masks will be performed directly within the previously organized directory,
whereas for the generation of ground truth masks, you will need to specify a new output directory.

```
<in directory data_utility/src/syn_plant>
python proc_data.py \
--data_dir <path/to/classification/directory> \
--species <species you want to process> \
--output_dir <output/path/to/ground/truth/masks>
--image_shape <resolution of the mask image>
```
After this step, you will see the results as follows:

```
<directory>
|---<species>
|   |---cropped_mask
|   |---mask_branch
|   |---mask_leaf
|   |---amodal_seg # if the species is amodal
|   |---...
```

In `cropped_mask`, there are processed masks.

Also, for ground truth result, you can see

```
<directory>
|---p_<species>
|   |---train
|   |   |---segment
|   |   |   |---00001.png
|   |   |   |---...
```

Here are ground truth mask result for each data. 

You can also complete these steps through GUI.  
The paths used in GUI are set to default paths throughout the workflow, but you can modify them as needed.