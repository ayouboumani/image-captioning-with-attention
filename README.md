# Image captioning using region attention
This is a code in Pytorch used for a project in the course Object Recognition and Computer Vision MVA Class 2019/2020.
The project is about image captioning using region attention and focuses on the paper Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering.
## Getting Started


### Requirements

Python 3.6 

Pytorch 0.4 

CUDA 9.0 

and a GPU of course...


### Installing

1. Clone the code: 

```
git clone https://github.com/ayouboumani/image-captioning-with-attention.git
```

2.Create the environment for the Bottom-Up part:

```
conda create --name bottom_env
conda activate bottom_env
```
Next, go to the directory 'bottom-up' and run this command for installing python packages:

```
pip install -r requirements.txt
```
```
conda deactivate
```
3.Create the environment for the Top-Down part:

```
conda create --name top_env
conda activate top_env
```
Next, go to the directory 'top-down' and run:

```
pip install -r requirements.txt
```
```
conda deactivate
```

## Running the code

### 0. Pre-trained models

1. Bottom-Up: download from https://drive.google.com/file/d/10MBUgH_OygyEys59FNQ4qNGeQ9bl-ODb/view?usp=drivesdk
and put it in bottom-up/models/. 
This model is trained on Visual Genome.
2. Top-Down: download from https://drive.google.com/file/d/10atC8rY7PdhnKW08INO33mEXYUyQ6G0N/view
and put it in top-down/.
This model is trained on Microsoft COCO dataset.

### 1. Demo

In your working directory, create the folders: images/, features/ and captions/.
Put your test images in the corresponding folder.
1.Extract image features. First, activate the bottom env, go to the bottom directory and run either:
```
python extract_features.py --image_dir ../images --out_dir ../features
```
or if you want to view attention

### 2. Data preparation

### 3. Training

### 4. Evaluation

## Acknowledgments
This code is mainly a combination of existing repositories:
* https://github.com/poojahira/image-captioning-bottom-up-top-down
* https://github.com/violetteshev/bottom-up-features
* https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning

