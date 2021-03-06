# CraftVision :eyeglasses::speech_balloon: 

> *Judge not, that you be not judged. For with the judgment you pronounce you will be judged, and with the measure you use it will be measured to you. [Matthew 7:1-2](https://www.biblegateway.com/passage/?search=Matthew+7:1-2)*


<p align="center">
  <img src="https://github.com/lukexyz/CraftVision/blob/master/output/assortment_0c.png?raw=true">
</p>

Ever accidentally enjoyed a sub-standard beverage? Ever wondered whether your premium IPA was fresh and delicious — or simply your brain trying to desperately justify the exuberant asking price? Well, fear no longer.

Now with the wisdom of the crowd, you too can calibrate a uniquely typical palette for craft brews, with _CraftVision_.  
`Identify craft beers that pass the Critic's judgement` ★★★★★

## Methodology

1. A first pass segments classes using the `YOLOv3` model.
2. Bottles are sent to a custom `ResNet34` classifier, fine-tuned on a collated label dataset (pre-trained on imagenet).
3. Ratings from `BeerAdvocate` are collected and displayed with a `OpenCV` overlay.

<br/>

<p align="center">
  <img src="https://github.com/lukexyz/CraftVision/blob/master/output/Artboard_2.png?raw=true">
</p>

<br/>

  → :notebook_with_decorative_cover: See [nb_controller.ipynb](/nb_controller.ipynb) for notebook 

  → :chart_with_downwards_trend: Training and hyperparameters are in [02_Dataset_and_training.ipynb](notebooks/02_Dataset_and_training.ipynb)

  → :bookmark_tabs: CraftVision/[detector.py](/detector.py) for inference code

<br/>

## Installation from AWS
`Deep Learning AMI (Ubuntu) Version 25.0`, GPU `p2.xlarge` for training :ballot_box_with_check:, `100 GB`

##### SSH into new linux box, activate pytorch conda environment
    $ ssh -i "<key>.pem" ubuntu@ec2-<public-ip>.us-east-2.compute.amazonaws.com
    $ source activate pytorch_p36
    
##### Clone and install requirements
    $ git clone https://github.com/lukexyz/CraftVision.git
    $ cd CraftVision
    $ sudo pip3 install -r requirements.txt

##### Download pretrained weights
    $ cd weights/
    $ bash download_weights.sh

## Run notebooks
    $ jupyter notebook --ip=0.0.0.0 --no-browser
    
Access notebooks through a browser with the aws public IPv4 address (found in ec2 instance console), not the private IP as shown in the notebook terminal. Do use the provided token from the terminal.

    http://<public IP>:8888/?token=<token>

<br/>

### Development Notes
1. :white_check_mark: Get Fastai custom dataset trained for labels
2. :white_check_mark: Get YOLOv3 implementation running
3. :white_check_mark: Extract bottle bounding box from YOLOv3 and send them to the brand classifier CNN
4. :white_check_mark: Run repo from `WSL` for development and use `aws` GPUs for training 
5. :white_check_mark: Run on saved video and tune up for real-time stream later
6. :white_check_mark: Improve HUD visuals when critic score is returned and display beverage info
    - Resize image output so text is readable
7. Increase number of brands → Run YoloV3 on training corpus to extract bottles before training
8. Refactor `detector` to optimise for stream capture
9. Upgrade segmentation model from YOLO to [detectron2](https://github.com/facebookresearch/detectron2)



<p align="center">
  <img src="https://github.com/lukexyz/CraftVision/blob/master/data/video/craft-vid-crop.gif?raw=true">
</p>

MIT License


##### Acknowledgements

* [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
* [Jeremy Howard - Practical Deep Learning for Coders, v3 - Lesson 2](https://course.fast.ai/)
