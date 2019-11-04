<p align="center">
  <img src="https://github.com/lukexyz/CraftVision/blob/master/output/assortment_0c.png?raw=true">
</p>

# :camera: CraftVision

Identify craft beers and dipslay their BeerAdvocate rating ★★★★☆

#### Method

1. A `YOLOv3` first pass is used to segment classes
2. Bottles and sent to a custom `ResNet` classifier, fine-tuned on a curated craft beer dataset (pre-trained on imagenet)
3. Identifies and displays `BeerAdvocate.com` ratings in an `OpenCV` overlay.

See CraftVision/nb_controller.ipynb for notebook
See CraftVision/detector.py for pytorch code

## Installation from AWS
`Deep Learning AMI (Ubuntu) Version 22.0`, GPU `p2.xlarge` for training :ballot_box_with_check:, `100 GB`

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

    http://<public IP>:8888/?token=<paste token here>


# Development Notes
1. :white_check_mark: Get Fastai custom dataset trained for labels
2. :white_check_mark: Get YOLOv3 implementation running
3. :white_check_mark: Extract bottle bounding boxing from YOLOv3 and send them to the brand classifier CNN
4. :white_check_mark: Run repo from `WSL` for development and use `aws` GPUs for training 
5. :white_check_mark: Run on saved video and tune up for real-time stream later
6. :white_check_mark: Improve HUD visuals when critic score is returned and displayed about beverage
    - Resize image output so text is readable

<p align="center">
  <img src="https://github.com/lukexyz/CraftVision/blob/master/data/video/craft-vid-crop.gif?raw=true">
</p>
