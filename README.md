# CraftVision
Craft brewery identifier and critic (AR)


## Installation from aws Deep Learning AMI

    $ source activate pytorch_p36
    
##### Clone and install requirements
    $ git clone https://github.com/eriklindernoren/PyTorch-YOLOv3
    $ cd PyTorch-YOLOv3/
    $ sudo pip3 install -r requirements.txt

##### Download pretrained weights
    $ cd weights/
    $ bash download_weights.sh

##### Install fastai
    $ conda install -c fastai fastai
    (optional extension for notebooks)
    $ conda install -c conda-forge jupyter_contrib_nbextensions
    
## Run notebooks
    $ jupyter notebook --ip=0.0.0.0 --no-browser
Access notebooks through a browser with the aws public IPv4 address (found in ec2 instance console), not the private IP as shown in the notebook terminal. Do use the provided token from the terminal.

    https://<public IP>:8888/?token=<paste token here>
