# CraftVision
Craft brewery identifier and critic.

1. :white_check_mark: Get Fastai custom dataset trained for labels
2. :white_check_mark: Get YOLOv3 implementation running
3. :white_check_mark: Extract bottle bounding boxing from YOLOv3 and send them to the brand classifier CNN
4. Improve HUD visuals when critic score is returned and displayed about beverage
4. Get this all running. Video first fine, the video stream


## Installation from aws
`Deep Learning AMI (Ubuntu) Version 22.0`, GPU `p2.xlarge` or larger, spot instance :ballot_box_with_check:, `100 GB`

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
