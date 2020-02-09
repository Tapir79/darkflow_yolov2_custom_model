# Intro
* This project is for **Windows users**. For Linux there are plenty of ML tools and frameworks available but for Windows there are just a few.
* I cloned this amazing repo https://github.com/thtrieu/darkflow (GNU licence) that enables you to run YOLOv2 on Windows. Unfortunately YOLOv3 is not yet supported. Following the instructions of Mark Jay [https://www.youtube.com/channel/UC2W0aQEPNpU6XrkFCYifRFQ ](https://www.youtube.com/watch?v=PyjBd7IDYZs&list=PLX-LrBk6h3wSGvuTnxB2Kj358XfctL4BM) I was able to create a custom image labeler. 
The hardest part in every Python project is the environment setup and the dependency hell. Good luck with that!  
* If you want to run this locally with GPU acceleration (like you really should) you need a compatible NVIDIA graphics card. If you can find your graphics card here, you are good to go: https://en.wikipedia.org/wiki/CUDA


# Installation

### Get the repo
You have 3 options: 
* Fork the repo
* git clone https://github.com/Tapir79/darkflow_yolov2_custom_model.git 
* download and unzip it 

### For The GPU you need
1. CUDA 9.0 
https://developer.nvidia.com/cuda-90-download-archive
1. cuDNN v7.6.5 (November 5th, 2019), for CUDA 9.0
https://developer.nvidia.com/rdp/cudnn-download
1. Extract cuDNN. From all folders copy contents to corresponding CUDA folders. I.e. cuDNN/bin/ to CUDA/bin
1. Environment variables (if you can't find these, add them): 
  * CUDA_PATH: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0
  * CUDA_PATH_V9_0: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0
  * NVCUDASAMPLES9_0_ROOT: C:\ProgramData\NVIDIA Corporation\CUDA Samples\v9.0
  * PATH: 
    * C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\lib\x64
    * C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin
    * C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\libnvvp
    * C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\extras\CUPTI\libx64

### Anaconda environment 

1. Install Anaconda (choose to install conda in path)
https://www.anaconda.com/ 

2. Create Anaconda environment with Python 3.6 
  * IMPORTANT!!! All this is tested with these exact versions. Any other combination might not work. 
  * I shared the weights and the opencv version (especially the opencv version might be almost impossible to come by): https://drive.google.com/drive/folders/1hTaNZy0aW0bd3HQ8TrX2ZcVgtP3UnsQA?usp=sharing
  * The original project required Tensorflow-gpu 1.4.1 and CUDA 8. Pip didn't offer the Tensorflow-gpu 1.4.1 so I dowloaded the binary for Windows. The binary didn't work with other libraries even after several combination attempts and I gave up on it. I was able to make everything work with CUDA 9 and Tensorflow 1.12. The CPU Tensorflow 1.4.0 is still available through pip and works fine with CUDA 8. 
````
conda create -n darkflow pip python=3.6
conda activate darkflow
pip install Cython
pip install numpy==1.18.1
pip install matplotlib==2.1.2
pip install lxml
pip install tensorflow-gpu==1.12
pip install opencv_python-3.4.1+contrib-cp36-cp36m-win_amd64.whl
pip install tqdm
````
* Build the darkflow project
```
python setup.py build_ext --inplace
```

* (optional) If you ever need to remove the environment
``conda env remove -n darkflow``


### Testing that everything is working
In terminal run webcam test: 

````conda activate darkflow
python process_video.py
````

Get any mp4 -video of your choice. For demo choose a short and small one.  

This is how you create a labeled video:
````
python flow --model cfg/yolo.cfg --load darkflow/bin/yolov2.weights --demo IMG_0972.mp4 --saveVideo
````

# Custom object detection: 
We're not training everything from scratch. Instead we're using something called transfer learning. This is the case when you have a *small dataset and pretrained model with similar data*. 

* We take a pretrained model like yolov2 or yolov2-tiny. This is the **base model**
* We remove the **head** of the base model a.k.a the few first layers of the model
* We **freeze** the base model. This means that we don't touch the weights of the base model anymore. 
* We create our own custom head and attach it to the base model.
* The custom head is what we're training here with our custom images.

For a decent model you need at least 1000 good images and good labels per detected class. With decent we're talking about 70-80 % detection accuracy. 
* Images should be from different angles and you should have images of the detected object partly covered etc.
* You can also try to make your own image pyramids and twist the images before labeling. This is like manual convolution.  
* The more images you have the better model you get.   
* For every image you need an annotation. It is metadata saying that in these coordinates is my image and what it is representing

## Prepare the dataset

You can either use prelabeled dataset from GoogleApis or label your own images. 

### Option 1. With Googleapis OID ready annotated images
* For the data collection I used 
  https://github.com/EscVM/OIDv4_ToolKit
  that collects images from https://storage.googleapis.com/openimages/web/index.html
* I run 
  ````
  python main.py downloader --classes Scissors --type_csv train --limit 1000
  ````
* These images come with annotations but they are in darknet format
* When you download the images a folder called ``OID`` is created
* Copy the folder as it is inside the ``custom_model_training`` folder
* **IMPORTANT!!!** Image names cannot contain any special characters or hyphens. Just rename them like so 00001.JPG, 00002.JPG if they look  like ``ssrsg-35hety-ge5635.JPG ``
* Run ``oid_to_voc_xml.py`` 
* In ``OID\Dataset\train\Scissors`` you should have *.jpg images and *.xml annotations. 
* Move images to ``custom_model_training\images``
* Move annotations to 
  ``custom_model_training\annotations``

### Option 2. With plain images label manually
* Put all your images into ``custom_model_training\images`` folder
* cd into ``custom_model_training`` folder and run:  
  ````
  python draw_box.py
  ````
* Draw boxes around the objects you want to annotate
* press `` q`` to save and proceed to the next image
  
## Train model with tiny yolov2
### Cfg modification
* In cfg folder find ``tiny-yolo-voc.cfg`` , copy it and rename it
* !!! IMPORTANT - the original ``tiny-yolo-voc.cfg`` must be left as it is
* In the last layer ``[region]`` of the model the classes must be changed to the no of classes that you have. I had just 1. 
* In the ``[convolutional]`` layer right above the ``[region]`` layer you need change the filters. How to count the filters: 5 *(classes + 5). So filters with 1 class is 30. 
* A modified example file is named ``tiny-yolo-voc-c1.cgf``
  
### Download tiny yolo weights
This google drive has the darkflow yolov2 weights:
https://drive.google.com/drive/folders/0B1tW_VtY7onidEwyQ2FtQVplWEU
* Download: tiny-yolo-voc.weights
* Put them in the ``/bin`` -folder

### Modify the labels.txt
* Write the classes one on a row 
* I have just 1 class so my file has 1 row 
* scissors

### Train (finally!)
* In root folder.
* First time training with : ``python flow --model cfg/tiny-yolo-voc-1c.cfg --load bin/tiny-yolo-voc.weights --train --annotation custom_model_training/annotations --dataset custom_model_training/images --gpu 1.0 --epoch 150 ``

### Run your custom model
* Get yourself a pair of scissors or whatever you trained your model with. 
* Open process_video.py and replace the existing option map with the following: 
  * final-ckpt is in ckpt folder. There are many files called tiny-yolo-voc-1c-XXX.profile
  * Replace XXX with the value in the last file
  ````
  option = {
    'model': 'cfg/tiny-yolo-voc-1c.cfg',
    'load': XXX,
    'threshold': 0.1,
    'gpu': 0.9
  }
  ````
* run ``process_video.py``