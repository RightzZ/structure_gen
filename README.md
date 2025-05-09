# Structure-Aware Image Generation

## Introduction
This is a project about Plants & AI, featuring multiple functionalities.


Therefore, based on this paper, the project offers the following features:

- Model Training: Train a model for plant sample generation based on the Pix2Pix framework.
- Mask Generation: Generate a series of 3D plant models using Blender and L-system, along with corresponding asks for 
leaves and branches.
- Data Management: Manage the data generated by Mask Generation, and preprocess it for use in the texturing module.
- Texturing: Combine the generated plant masks with the trained model to produce a set of realistic plant images, which 
can be used as training data for other plant-related tasks.

## Environment
This project is developed and tested inside a Docker container.

Docker:([nvidia/cuda:11.6.2-cudnn8-devel-ubuntu20.04](https://hub.docker.com/layers/nvidia/cuda/11.6.2-cudnn8-devel-ubuntu20.04/images/sha256-4eeb683bf695d431ecba6c949b4ee86c1cff61c2786c4de93b8df095f0852b78?context=explore))  
Python:3.8  
PyTorch:1.13.1+cu116 and torchvision 0.14.1+cu116

## Dependency
This project contains the modifications of [Segment Anything](https://github.com/facebookresearch/segment-anything), [MaskDINO](https://github.com/IDEA-Research/MaskDINO), [Grounding-DINO](https://github.com/IDEA-Research/GroundingDINO), and [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) 
implementations. Please install all dependencies in your environment.

## Usage
We have built a GUI using Gradio, allowing you to perform each step of the workflow.
```
pip install gradio
python app.py
```
The GUI will run on the localhost:8888, you can change the port in app.py
```
if __name__ == '__main__':
    demo.launch(server_name="0.0.0.0", server_port=8888) # Change the port you want
```
Note: If you use GUI, we recommend keeping the default settings for all path-related parameters, as they have been 
adjusted to ensure seamless workflow between steps.

Alternatively, you can also execute each step via the command line. Detailed instructions for each step are provided in 
the corresponding README documents. 

## Dockerfile
Of course, if you want to automatically create a ready-to-use Docker container, you can use the configuration files 
included in the project.
```
cd Docker
chmod +x build.sh run.sh # add permissions
build.sh
run.sh
```
After this, a container will be automatically created with Miniconda installed inside. An environment named `sgen` will
be configured, and PyTorch, Gradio will be installed automatically. The local port 8888 will be bound to port 8888 inside
the container, allowing external access to the Gradio interface. Other dependencies still need to be installed manually.