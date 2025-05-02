# Mask Generation

This step allows you to generate models and mask data for various types of plants.

## Preparation

### Install Blender3.1 and lsystem
Here, we have already prepared Blender3.1 and add lsystem in it, also we adjust some code to generate plant masks
correctly. Please check Google drive link.


### Instell xvfb, libXi.so.6
Blender needs xvfb, libXi.so.6 and libXrender.so.1 to run. Please install xvfb by `apt install xvfb libxi6 libxrender1`.

## Usage
You can generate a series of plant masks and models through command-line commands.
```
cd mask_generation
python make_template_segdata.py \
--data_type <Plant species to generate: 'amodal', 'plant', 'komatsuna', 'ara'> \
--output_dir <path/to/output/directory> \
--data_num <Number of models to generate this time> \
--render_num <Render times of each model>
--resolution <Resolution of the generated mask image> \
--radius <Blender's camera orbit radius>
```

After generation, you may find results in the output directory like this form.
```
<output directory>
|---data_00001_ara
|   |---binary
|   |---img
|   |---json
|   |---model
|---data_00002_komatsuna
|---data_00003_plant
|---...
```
You can check out the result and select which you want to use.

Of course, you can also perform this entire workflow through the GUI interface. When you use GUI, for each plant species, 
parameters "Resolution" and "Camera orbit radius" will be automatically set to recommended values.

## Furthermore

More changes:\
You can modify `create_leaf_string` and `create_lsystem` functions in `make_template_segdata.py`
* The replacement rule of L-system (info is [here](http://algorithmicbotany.org/papers/abop/abop.pdf))
* How the leaves are attached to branch
    * Alternate (互生)  
      One leaf is attached to one node.
    * Opposite (対生)  
      Two leaves are attached to one node.
    * Decussate opposite (十字対生)  
      Two leaves are attached to one node,  
      but the direction in which the leaves are attached to the upper and lower nodes are 90 degrees different.
    * Verticillate (輪生)  
      Four leaves (in this project) are attached to one node.