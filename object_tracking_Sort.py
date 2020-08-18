# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + colab={"base_uri": "https://localhost:8080/", "height": 204} colab_type="code" id="s8ROmOGzmgrE" outputId="7316e5d2-fad6-480f-cc14-04bf3f746175"
#downloading yolo weight
# !wget https://pjreddie.com/media/files/yolov3.weights -O config/yolov3.weights

# + colab={"base_uri": "https://localhost:8080/", "height": 71} colab_type="code" id="vU0PTXLvl05y" outputId="ce680ee0-aaff-4786-c9ef-50a72fb8fdda"
from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# initialize Sort object and video capture
from sort import *

import cv2
from IPython.display import clear_output

# + colab={"base_uri": "https://localhost:8080/", "height": 71} colab_type="code" id="3MDPMjBFl78p" outputId="4a3eae71-69b3-41f4-d9bd-fcfb702c5067"
config_path='config/yolov3.cfg'
weights_path='config/yolov3.weights'
class_path='config/coco.names'
img_size=416
conf_thres=0.8
nms_thres=0.4

# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_weights(weights_path)
model.cuda()
model.eval()
classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor


# + colab={} colab_type="code" id="hN6hEnofmEqG"
def detect_image(img):
    # scale and pad image
    ratio = min(img_size/img.size[0], img_size/img.size[1])
    imw = round(img.size[0] * ratio)
    imh = round(img.size[1] * ratio)
    img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
         transforms.Pad((max(int((imh-imw)/2),0), max(int((imw-imh)/2),0), max(int((imh-imw)/2),0), max(int((imw-imh)/2),0)),
                        (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 80, conf_thres, nms_thres)
    return detections[0]


# + colab={} colab_type="code" id="YppzrNHQ3vDq"
def convertMillis(millseconds):
    seconds, millseconds = divmod(millseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    day, hours = divmod(hours, 24)
    seconds = int(seconds + millseconds/10000)
    return f"{int(hours)}:{int(minutes)}:{int(seconds)}"


# + colab={"base_uri": "https://localhost:8080/", "height": 1000} colab_type="code" id="4EbhlQ5PunS0" outputId="77625bde-1043-4787-a5ba-71cf697d096b"
videopath = './video/traffic.mp4'

# %pylab inline 


cmap = plt.get_cmap('tab20b')
colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

# initialize Sort object and video capture

cap = cv2.VideoCapture(videopath)
mot_tracker = Sort() 

#while(True):
for ii in range(100):
    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
    time_report = convertMillis(timestamp)
    ret, frame = vid.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pilimg = Image.fromarray(frame)
    detections = detect_image(pilimg)

    img = np.array(pilimg)
    pad_x = max(img.shape[0] - img.shape[1], 0) * (img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (img_size / max(img.shape))
    unpad_h = img_size - pad_y
    unpad_w = img_size - pad_x
    if detections is not None:
        tracked_objects = mot_tracker.update(detections.cpu())

        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        for x1, y1, x2, y2, obj_id, cls_pred in tracked_objects:
            box_h = int(((y2 - y1) / unpad_h) * img.shape[0])
            box_w = int(((x2 - x1) / unpad_w) * img.shape[1])
            y1 = int(((y1 - pad_y // 2) / unpad_h) * img.shape[0])
            x1 = int(((x1 - pad_x // 2) / unpad_w) * img.shape[1])

            color = colors[int(obj_id) % len(colors)]
            color = [i * 255 for i in color]
            cls = classes[int(cls_pred)]
            cv2.rectangle(frame, (x1, y1), (x1+box_w, y1+box_h), color, 4)
            cv2.rectangle(frame, (x1, y1-35), (x1+len(cls)*19+60, y1), color, -1)
            cv2.putText(frame, cls + "-" + str(int(obj_id)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 3)

    fig=figure(figsize=(12, 8))
    title("Video Stream")
    imshow(frame)
    title(f"Video Stream: {time_report}")
    frame_img = Image.fromarray(frame)
    frame_img_resize = frame_img.resize((512, 512))
    frame_img_resize.save(f"images/traffic_frame_{ii}.jpg")
    show()

# + colab={} colab_type="code" id="UhtCAT4U18a7"
import imageio

jpg_dir = './images/'
images = []
for file_name in os.listdir(jpg_dir)[:50]:
    if file_name.endswith('.jpg'):
        file_path = os.path.join(jpg_dir, file_name)
        images.append(imageio.imread(file_path))

imageio.mimsave('./video/traffic_1.gif', images, fps=55)
