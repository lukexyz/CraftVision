"""
Module version of ./detect.py
"""

from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

from fastai.vision import load_learner, open_image, ImageBBox
from pathlib import Path
from data.ratings import *


def detector(opt, **kwargs):
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    """
    print("="*60)
    print(" "*20, "STARTING DETECTOR")
    print("="*60, '\n')
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    os.makedirs("output", exist_ok=True)

    # -------------------- LOAD CLASSIFICATION MODEL ------------------- #
    # Fastai craft brands
    c_path = Path('data/training')
    learn = load_learner(c_path)
    # ------------------------------------------------------------------ #
    

    # --------------------- LOAD SEGMENTAION MODEL --------------------- #
    # Pytorch yolov3
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    print(opt.model_def)
    # return None
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))
    model.eval()  # Set in evaluation mode
    # ------------------------------------------------------------------ #
    
    
    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    # Bounding-box colors
    cmap = plt.get_cmap("hsv")
    colors = [cmap(i) for i in np.linspace(0, 1, 10)]

    print("\nIterating images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):

        img = Image.open(path)
        print(f"({img_i}) Image: '{path}' {img.size}")

        # Configure output resolution
        dpi = 60
        height, width = img.size
        figsize = width / float(dpi), height / float(dpi)
        # Create plot
        img = np.array(img)
        # plt.figure(figsize=figsize)
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            
            # Sort boxes from left to right (x1)
            detections = sorted(detections, key=lambda x: int(x[0]))
            
            for det_i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
                if cls_pred != 39: continue  # Skip non-bottles

                # ========================== Classify Image ============================== #
                x1, y1, x2, y2 = [int(x) for x in [x1, y1, x2, y2]]
                crop_img = img[y1:y2, x1:x2]               
                
                # Convert cropped np arrary to fastai img for learner
                crop_img = Image.fromarray(crop_img)
                crop_img.save("data/training/temp/_.jpeg")
                crop_img = open_image("data/training/temp/_.jpeg")
                
                # Send to classifier
                b_pred, pred_idx, outputs = learn.predict(crop_img)
                # ======================================================================== #
    
                box_w = x2 - x1
                box_h = y2 - y1
                color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                color = (0.0, 0.2562523625023627, 1.0, 1.0) # blue
                c2 = (0.0, 1.0, 0.617278533938476, 1.0)     # cyan

                # Add label
                label_name = f"{str(b_pred).replace('_', ' ')}"
                rating = get_rating(label_name)
                title = f"{label_name}\n{rating}" 
                title = f'{label_name}\n"{rating[1]}" ({rating[3]} reviews)'

                # Create a Rectangle patch
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=4, edgecolor='blue', facecolor="none")
                bbox2 = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=c2, facecolor="none")

                # Add the bbox to the plot
                ax.add_patch(bbox)
                ax.add_patch(bbox2)

                plt.text(x1, y1,
                         s = title,
                         color="white",
                         verticalalignment="top",
                         bbox=dict(facecolor=color, edgecolor='blue', pad=3, alpha=0.9, boxstyle='round,pad=0.5'))
                
                # Review score label
                score_x = x1 + 10
                score_y2 = y2 - 40
                score_str = f"{rating[0]}%"

                plt.text(score_x, score_y2,
                         s = score_str,
                         color="white",
                         verticalalignment="top",
                         bbox=dict(facecolor=color, edgecolor='blue', pad=0, alpha=0.8, boxstyle='Circle,pad=0.1'))
                
                print(f"\t[Det {det_i}]\t+ Prediction: {classes[int(cls_pred)]} ({cls_conf.item():.3f}) {b_pred}")

        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        if opt.save_fig:
            filename = path.split("/")[-1].split(".")[0]
            plt.savefig(f"output/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        if opt.show_img:
            plt.show()
        #plt.close()

    return imgs, img_detections
