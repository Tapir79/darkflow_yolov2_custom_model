## Intro

I cloned this amazing repo https://github.com/thtrieu/darkflow (GNU licence) that enables you to run YOLOv2 on Windows. Unfortunately YOLOv3 is not yet supported. Following the instructions of Mark Jay [https://www.youtube.com/channel/UC2W0aQEPNpU6XrkFCYifRFQ ](https://www.youtube.com/watch?v=PyjBd7IDYZs&list=PLX-LrBk6h3wSGvuTnxB2Kj358XfctL4BM) I was able to create a custom image labeler. 

### For The GPU you need
1. CUDA 9.0 
https://developer.nvidia.com/cuda-90-download-archive
1. cuDNN v7.6.5 (November 5th, 2019), for CUDA 9.0
https://developer.nvidia.com/rdp/cudnn-download
1. Extract cuDNN. From all folders copy contents to corresponding CUDA folders. I.e. cuDNN/bin/ to CUDA/bin
1. Environment variables: 
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
````
conda create -n darkflow pip python=3.6
conda activate darkflow
pip install Cython
pip install numpy==1.18.1
pip install tensorflow-gpu==1.12
pip install tensorflow_gpu-1.4.0-cp36-cp36m-win_amd64.whl
pip install opencv_python-3.4.1+contrib-cp36-cp36m-win_amd64.whl

python setup.py build_ext --inplace
````
1. (optional) If you ever need to remove the environment
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
For a decent model you need at least 1000 images per detected class
* Images should be from different angles and you should have images of the detected object partly covered etc.
* The more images you have the better model you get 
* For every image you need an annotation. It is metadata saying that in these coordinates is my image and what it is representing

## Collect data
* For the data collection I used 
  https://github.com/EscVM/OIDv4_ToolKit
  that collects images from https://storage.googleapis.com/openimages/web/index.html
* I run 
  ````
  python main.py downloader --classes Person --type_csv train --limit 1000
  ````
* You can also download the complete dataset here (needs a lot of disk space): https://storage.googleapis.com/openimages/web/download_v4.html
* These images come with annotations but I did my own
## Annotate data 
* In custom_model_project run: 
  ````
  ````

## Train model  