from PIL import Image
import platform
#import skimage.transform
import numpy as np
import random
import math
import glob
import io
import base64
#import cv2

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

import xml.etree.ElementTree as ET
import os
from os import listdir
from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import time

possible_labels = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
          'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
          'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

if T.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"

vgg16 = models.vgg16(pretrained = True).to(device)

img_transform = transforms.Compose([
         transforms.ToTensor(),
         transforms.Normalize(
         mean=[0.485, 0.456, 0.406],
         std=[0.229, 0.224, 0.225]
        )])

def vectorized_get_iou(a, b):
    ai = T.zeros_like(a).to(device)
    bi = T.zeros_like(b).to(device)

    ai[:, 0] = a[:, 0]
    ai[:, 1] = a[:, 1]
    ai[:, 2] = a[:, 0]+a[:, 3]
    ai[:, 3] = a[:, 1]+a[:, 2]

    bi[:, 0] = b[:, 0]
    bi[:, 1] = b[:, 1]
    bi[:, 2] = b[:, 0]+b[:, 3]
    bi[:, 3] = b[:, 1]+b[:, 2]

    x = T.max(T.stack((ai[:, 0], bi[:, 0])).T, dim=1)[0]
    y = T.max(T.stack((ai[:, 1], bi[:, 1])).T, dim=1)[0]
    w = T.min(T.stack((ai[:, 2], bi[:, 2])).T, dim=1)[0] - x
    h = T.min(T.stack((ai[:, 3], bi[:, 3])).T, dim=1)[0] - y

    defined = T.where(T.logical_and(w>=0, h>=0))[0]
    intersection = T.zeros(ai.shape[0]).to(device)
    intersection[defined] = w[defined].float() * h[defined].float()

    union = (a[:, 2]*a[:, 3]) + (b[:, 2]*b[:, 3]) - intersection
    return intersection/union

def calculate_IoU(bb1, bb2):
    bb1_size = (bb1[2]-bb1[0])*(bb1[3]-bb1[1])
    bb2_size = (bb2[2]-bb2[0])*(bb2[3]-bb2[1])
    
    Xs = [[1, bb1[0]], [1, bb1[2]], [2, bb2[0]], [2, bb2[2]]]
    Ys = [[1, bb1[1]], [1, bb1[3]], [2, bb2[1]], [2, bb2[3]]]
    
    Xs.sort(key = lambda x: x[1])
    Ys.sort(key = lambda x: x[1])

    if Xs[0][0] == Xs[1][0] or Ys[0][0] == Ys[1][0]:
        return 0
    
    x_overlap = Xs[2][1] - Xs[1][1]
    y_overlap = Ys[2][1] - Ys[1][1]

    intersection = x_overlap * y_overlap
    
    final = intersection/(bb1_size + bb2_size - intersection)
    
    return final

def transform_to_rchw(pred):
    #pred is transformed from [x1, y1, x2, y2] to [r, c, h, w]
    return [pred[0], pred[1], pred[3]-pred[1], pred[2]-pred[0]]

def transform_to_xyxy(pred):
    #pred is transformed from [r, c, h, w] to [x1, y1, x2, y2]
    return [pred[0], pred[1], pred[0]+pred[3], pred[1]+pred[2]]

def get_transformation(label, pred):
    #label and pred comes in the form [r, c, h, w]
    return [(label[0]-pred[0])/pred[3], (label[1]-pred[1])/pred[2], 
                      math.log(label[2]/pred[2]), math.log(label[3]/pred[3])]

def transform_pred(pred, transformation):
    #pred and transformation comes in the form [r, c, h, w]
    r = T.round(pred[3]*transformation[0] + pred[0])
    c = T.round(pred[2]*transformation[1] + pred[1])
    h = T.round(pred[2]*math.exp(transformation[2]))
    w = T.round(pred[3]*math.exp(transformation[3]))
    return r, c, h, w

def one_hot_encode_label(label, num_categories):
    final = T.zeros(label.shape[0], num_categories)
    new_label = T.cat((T.arange(label.shape[0]).unsqueeze(0).to(device), label.unsqueeze(0)))
    final[new_label[0], new_label[1]] = 1
    return final

def get_center_points(height, width):
    input = T.zeros(height, width)
    input[T.arange(1, height-1)] = 1
    input[:, 0] = 0
    input[:, width-1] = 0
    indices = T.where(input == 1)
    return T.stack((indices[0], indices[1]), dim=1)

def scale_center_points(indices, input_height, input_width, scaled_height, scaled_width):
    height_scale = scaled_height/input_height
    width_scale = scaled_width/input_width
    indices[:, 0] = T.round(indices[:, 0]*height_scale)+1
    indices[:, 1] = T.round(indices[:, 1]*width_scale)+1
    return indices

def vectorized_transform_bb(bbs, transformation):
    transformed = T.zeros_like(bbs).to(device)
    #r
    transformed[:, 0] = T.round(bbs[:, 3]*transformation[:, 0] + bbs[:, 0])
    #c
    transformed[:, 1] = T.round(bbs[:, 2]*transformation[:, 1] + bbs[:, 1])
    #
    transformed[:, 2] = T.round(bbs[:, 2]*T.exp(transformation[:, 2]))
    #w
    transformed[:, 3] = T.round(bbs[:, 3]*T.exp(transformation[:, 3]))
    
    return transformed

def vectorized_get_transformation(label, pred):
    #returns transformation from pred to label
    
    transformation = T.zeros_like(label).to(device)
    #r
    transformation[:, 0] = (label[:, 0]-pred[:, 0])/pred[:, 3]
    #c
    transformation[:, 1] = (label[:, 1]-pred[:, 1])/pred[:, 2]
    #h
    transformation[:, 2] = T.log(label[:, 2]/pred[:, 2])
    #w
    transformation[:, 3] = T.log(label[:, 3]/pred[:, 3])
    
    return transformation

def clip_bbs_to_fit_in_img(bbs, img_height, img_width):
    clipped_bbs = []
    for bb in bbs:
        old_r = bb[0]
        old_c = bb[1]
        old_h = bb[2]
        old_w = bb[3]
        if old_r < 0:
            r = 0
        else:
            r = old_r
        if old_c < 0:
            c = 0
        else:
            c = old_c
        if r + old_h > img_height:
            h = img_height-r
        else:
            h = old_h
        if c + old_w > img_width:
            w = img_width-c
        else:
            w = old_w
        clipped_bbs.append([r, c, h, w])
    return T.tensor(clipped_bbs)

def filter_bbs(bbs, img_height, img_width):
    filter = T.where(T.logical_and(T.logical_and(bbs[:, 0] >= 0, bbs[:, 1] >= 0), 
                          T.logical_and(bbs[:, 0] + bbs[:, 2] <= img_height, bbs[:, 1] + bbs[:, 3] <= img_width)))
    filtered_bbs = bbs[filter]
    return filtered_bbs

def non_max_suppression(bbs, scores):
    b = []
    for i, bb in enumerate(bbs):
        b.append([i, bb, scores[i]])
    b.sort(key = lambda x: x[2])

    d = []

    while len(b) != 0:
        highest_scored_bb = b.pop(0)
        d.append(highest_scored_bb)
        bbs_to_del = []
        if len(b) == 0:
            break
        for i, bb_data in enumerate(b):
            if calculate_IoU(highest_scored_bb[1], bb_data[1]) >= 0.3:
                bbs_to_del.append(bb_data)
        for data in bbs_to_del:
            b.remove(data)
    
    return d

class faster_rcnn_dataset(Dataset):
    def __init__(self, data_dict, filenames, image_folder_path):
        self.image_folder_path = image_folder_path
        self.data = {filename:data_dict[filename] for filename in filenames}
        self.keys = list(self.data.keys())
        random.shuffle(self.keys)
        self.height_scale = 600/224
        self.width_scale = 1000/224
        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, index):
        if T.is_tensor(index):
            idx = idx.tolist()
            
        filename = self.keys[index]
        data = self.data[filename]
        
        image_path = os.path.join(self.image_folder_path, filename)
        img = Image.open(image_path)
        reshape_tuple = (1000, 600) if img.size[0] > img.size[1] else (600, 1000)
        img = np.asarray(img.resize(reshape_tuple))
        img = img_transform(img)
        
        width_scale = reshape_tuple[0]/224
        height_scale = reshape_tuple[1]/224
    
        bbs = T.round(T.tensor([transform_to_rchw([x[0][0]*width_scale, x[0][1]*height_scale, x[0][2]*width_scale, x[0][3]*height_scale]) for x in data]))
        labels = T.tensor([possible_labels.index(x[1]) for x in data])
        
        return filename, img, bbs, labels, reshape_tuple

class vgg16_base(nn.Module):
    def __init__(self):
        super(vgg16_base, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding = (1, 1))
        self.conv1.weight.data = list(vgg16.children())[0][0].weight.data
        self.conv1.bias.data = list(vgg16.children())[0][0].bias.data
        
        self.conv2 = nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2.weight.data = list(vgg16.children())[0][2].weight.data
        self.conv2.bias.data = list(vgg16.children())[0][2].bias.data
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv3.weight.data = list(vgg16.children())[0][5].weight.data
        self.conv3.bias.data = list(vgg16.children())[0][5].bias.data
        
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4.weight.data = list(vgg16.children())[0][7].weight.data
        self.conv4.bias.data = list(vgg16.children())[0][7].bias.data
        
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv5.weight.data = list(vgg16.children())[0][10].weight.data
        self.conv5.bias.data = list(vgg16.children())[0][10].bias.data
        
        self.conv6 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6.weight.data = list(vgg16.children())[0][12].weight.data
        self.conv6.bias.data = list(vgg16.children())[0][12].bias.data
        
        self.conv7 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv7.weight.data = list(vgg16.children())[0][14].weight.data
        self.conv7.bias.data = list(vgg16.children())[0][14].bias.data
        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        self.conv8 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv8.weight.data = list(vgg16.children())[0][17].weight.data
        self.conv8.bias.data = list(vgg16.children())[0][17].bias.data
        
        self.conv9 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv9.weight.data = list(vgg16.children())[0][19].weight.data
        self.conv9.bias.data = list(vgg16.children())[0][19].bias.data
        
        self.conv10 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv10.weight.data = list(vgg16.children())[0][21].weight.data
        self.conv10.bias.data = list(vgg16.children())[0][21].bias.data
        
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        
        self.conv11 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv11.weight.data = list(vgg16.children())[0][24].weight.data
        self.conv11.bias.data = list(vgg16.children())[0][24].bias.data
        
        self.conv12 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv12.weight.data = list(vgg16.children())[0][26].weight.data
        self.conv12.bias.data = list(vgg16.children())[0][26].bias.data
        
        self.conv13 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv13.weight.data = list(vgg16.children())[0][28].weight.data
        self.conv13.bias.data = list(vgg16.children())[0][28].bias.data
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)
        
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool2(x)
        
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.pool3(x)
        
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))
        x = self.pool4(x)

        x = self.relu(self.conv11(x))
        x = self.relu(self.conv12(x))
        x = self.relu(self.conv13(x))
        
        return x

class Region_Proposal_Network(nn.Module):
    def __init__(self):
        super(Region_Proposal_Network, self).__init__()
        self.k = 9        
        self.scales = [128, 256, 512]
        self.aspect_ratios = [[1, 1], [1, 2], [2, 1]]
        
        self.conv1 = nn.Conv2d(512, 512, kernel_size = (3, 3))
        T.nn.init.normal_(self.conv1.weight.data, mean=0.0, std=0.01)
        T.nn.init.constant_(self.conv1.bias.data, 0)
        
        self.fcn_cls = nn.Linear(512, 2*self.k)
        T.nn.init.normal_(self.fcn_cls.weight.data, mean=0.0, std=0.01)
        T.nn.init.constant_(self.fcn_cls.bias.data, 0)
        
        self.fcn_bbox_reg = nn.Linear(512, 4*self.k)
        T.nn.init.normal_(self.fcn_bbox_reg.weight.data, mean=0.0, std=0.01)
        T.nn.init.constant_(self.fcn_bbox_reg.bias.data, 0)
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x, initial_img_height, initial_img_width):
        _, in_channels, in_height, in_width = x.shape
        center_points = get_center_points(in_height, in_width)
        scaled_center_points = scale_center_points(center_points, in_height, 
                                                   in_width, initial_img_height,
                                                   initial_img_width)
        
        box_sizes = [[num*scale for num in ratio] for scale in self.scales for ratio in self.aspect_ratios]
        anchors = T.tensor([[cp[0] - (size[0]/2), cp[1] - (size[1]/2), size[0], size[1]]for cp in scaled_center_points.tolist() for size in box_sizes])
        anchors = anchors.to(device)

        x = self.relu(self.conv1(x))
        x = T.t(T.flatten(x.squeeze(dim=0), start_dim=1))
        objectness = self.fcn_cls(x)
        bbox_reg = self.fcn_bbox_reg(x)
        
        objectness = T.reshape(objectness, (-1, 2))
        bbox_reg = T.reshape(bbox_reg, (-1, 4))
        
        pred_bbs = vectorized_transform_bb(anchors, bbox_reg)
        
        return objectness, bbox_reg, pred_bbs, anchors

def rpn_multitask_loss(pred_objectness, label_objectness, pred_regs, label_regs):
    label_objectness = label_objectness.to(device)
    label_regs = label_regs.to(device)

    cross_entropy = nn.CrossEntropyLoss()
    one_hot_label = one_hot_encode_label(label_objectness, 2).to(device)
    l_class = cross_entropy(pred_objectness, one_hot_label)
    
    difference = pred_regs - label_regs
    less_than_ones = T.where(difference < 1)
    greater_equal_to_ones = T.where(difference >= 1)
    difference[less_than_ones] = 0.5*T.square(difference[less_than_ones])
    difference[greater_equal_to_ones] = T.abs(difference[greater_equal_to_ones])-0.5
    zeros = T.where(label_objectness == 0)
    difference[zeros[0], :] = 0
    l_bb = T.sum(difference)/(label_objectness.shape[0]*4)
    
    return l_class + l_bb

def train(filename, img, ground_truth_bbs, img_size):
    img_width, img_height = img_size
    
    start = time.time()
    base_output = model_base(img)
    objectness, bbox_reg, pred_bbs, anchors = rpn(base_output, img_height, img_width)
    
    labels = T.tensor([-1 for _ in range(pred_bbs.detach().shape[0])]).to(device)
    
    all_ious = T.zeros(pred_bbs.detach().shape[0], ground_truth_bbs.shape[0]).to(device)
    for i, gt in enumerate(ground_truth_bbs):
        all_ious[:, i] = vectorized_get_iou(pred_bbs.detach(), T.tensor([gt.tolist() for _ in range(pred_bbs.detach().shape[0])]).to(device))
    
    cross_filter = T.where(T.logical_or(T.logical_or(pred_bbs.detach()[:, 0] < 0, pred_bbs.detach()[:, 1] < 0), 
                          T.logical_or(pred_bbs.detach()[:, 1] + pred_bbs.detach()[:, 2] > img_height, 
                                       pred_bbs.detach()[:, 0] + pred_bbs.detach()[:, 3] > img_width)))[0]
    crossings = cross_filter.shape[0]
    all_ious[cross_filter] = -1
        
    max_values, max_indices = T.max(all_ious, dim=1)
    highest_iou_bbs = ground_truth_bbs[max_indices]
    
    zero_indices = T.where(all_ious[T.arange(0, all_ious.shape[0]).to(device), max_indices] == 0)[0]
    negative_indices = T.where(all_ious[T.arange(0, all_ious.shape[0]).to(device), max_indices] == -1)[0]
    highest_iou_bbs[zero_indices] = T.tensor([-1, -1, -1, -1]).float().to(device)
    highest_iou_bbs[negative_indices] = T.tensor([-1, -1, -1, -1]).float().to(device)
    
    positives = T.where(all_ious[T.arange(0, all_ious.shape[0]).to(device), max_indices] > 0.7)[0]
    negatives = T.where(T.logical_and(all_ious[T.arange(0, all_ious.shape[0]).to(device),max_indices] < 0.3,
                                      all_ious[T.arange(0, all_ious.shape[0]).to(device),max_indices] > -1))[0]
    
    pos_indices = []
    neg_indices = []
    
    for i, gt in enumerate(ground_truth_bbs):
        if gt not in highest_iou_bbs.tolist():
            index = T.argmax(all_ious[:, i]).item()
            highest_iou_bbs[index] = gt
            pos_indices.append(index)
            if index not in positives.tolist():
                positives = positives.tolist()
                positives.append(index)
                positives = T.tensor(positives)
            if index in negatives.tolist():
                negatives = negatives.tolist()
                negatives.remove(index)
                negatives = T.tensor(negatives)
            
    labels[positives] = 1
    labels[negatives] = 0
    
    accuracy = T.count_nonzero(T.argmax(objectness.detach(), dim=1) == labels)/(positives.shape[0] + negatives.shape[0])
    predicted_as_object = T.where(T.argmax(objectness.detach(), dim=1) == 1)[0]
    non_crossing_rights = max_values[predicted_as_object][max_values[predicted_as_object] != -1]
    iou = T.sum(non_crossing_rights)/non_crossing_rights.shape[0]
    diff_bbs_predicted = T.unique(highest_iou_bbs[predicted_as_object], dim=0).tolist()
    diff_bbs_predicted.remove([-1, -1, -1, -1])
    #print(diff_bbs_predicted)
    
    x = lambda a : T.where(a>0.7)[0].shape[0]
    right_preds_per_bb = list(map(x, T.unbind(all_ious[predicted_as_object], 1)))
    
    anchor_to_gt_trans = T.tensor([get_transformation(x, anchors.detach()[i]) 
                          if x.tolist() != [-1, -1, -1, -1]
                          else [-1, -1, -1, -1] 
                          for i, x in enumerate(highest_iou_bbs)])
    
    num_positives = 128-len(pos_indices) if len(positives) >= 128-len(pos_indices) else len(positives)
    num_negatives = 256 - num_positives
    
    pos_indices += random.sample(positives.tolist(), num_positives)
    neg_indices += random.sample(negatives.tolist(), num_negatives)
    indices = pos_indices + neg_indices
    
    loss = rpn_multitask_loss(objectness[indices], labels[indices],
                              bbox_reg[indices], anchor_to_gt_trans[indices])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    end = time.time()
    print("model time", end-start)
    
    return pred_bbs, objectness, loss.item(), accuracy, iou, len(diff_bbs_predicted), right_preds_per_bb

train_info = []
def train_loop(epochs, lr_decrease_epoch = -1):
    train_gen = iter(train_dataloader)
    for epoch in range(epochs):
        if epoch == lr_decrease_epoch:
            for g in optimizer.param_groups:
                g['lr'] = g['lr']/10
            print()
            print("Lowered lr ********************")
            print()
        try:
            data = next(train_gen)
        except StopIteration:
            train_gen = iter(train_dataloader)
            data = next(train_gen)
            
        train_filename = data[0][0]
        train_img = data[1].to(device)
        train_bbs = data[2][0].to(device)
        train_labels = data[3][0].to(device)
        train_size = T.tensor(data[4]).tolist()
            
        t_pred_bbs, t_objectness, t_loss, t_accuracy, t_iou, t_num, t_right_per_bb= train(train_filename, train_img, train_bbs, train_size)
        train_info.append([t_loss, t_accuracy, t_iou, t_num, t_right_per_bb])
        
        print()
        print("Epoch {}".format(epoch))
        print("Train Loss: {}, train_accuracy: {}, train_iou {}, num: {}".format(train_info[-1][0], train_info[-1][1], train_info[-1][2], train_info[-1][3]))
        print("Right Predictions Per BB {}".format(train_info[-1][4]))

        if epoch%1000 == 0:
            print("Saving Model...")
            T.save(model_base.state_dict(), "faster_rcnn_model_base.pth")
            T.save(rpn.state_dict(), "faster_rcnn_rpn.pth")
        print()

        with open("faster_rcnn_log.txt","a+") as f:
            f.write("\n")
            f.write("Epoch {}\n".format(epoch))
            f.write("Train Loss: {}, train_accuracy: {}, train_iou {}, num: {} \n".format(train_info[-1][0], train_info[-1][1], train_info[-1][2], train_info[-1][3]))
            f.write("Right Predictions Per BB {} \n".format(train_info[-1][4]))
            f.write("\n")


if __name__ == "__main__":
    #Windows
    if platform.system() == "Windows":
        image_folder_path = "C:\\my_data\\Pascal_Voc\\VOCdevkit\\VOC2012\\JPEGImages"
    #Linux
    else:
        image_folder_path = "/home/ubuntu/Desktop/nathan/data/VOCdevkit/VOC2012/JPEGImages/"
        image_folder_path = "/home/ubuntu/Desktop/nathan/data/VOC2012/JPEGImages/"
    with open('ground_truths.txt') as f:
        truths = [string.split() for string in f.readlines()]
    data_dict = {}
    for line in truths:
        if line[0] not in data_dict:
            data_dict[line[0]] = [[[int(line[1]), int(line[2]), int(line[3]), int(line[4])], line[5]]]
        else:
            data_dict[line[0]].append([[int(line[1]), int(line[2]), int(line[3]), int(line[4])], line[5]])

    if os.path.exists("train_filenames.txt") == False:
        filenames = list(data_dict.keys())
        random.shuffle(filenames)
        train_test_split = 0.985
        split = round(len(filenames)*train_test_split)
        train_filenames = filenames[:split]
        val_filenames = filenames[split:]

        with open('train_filenames.txt', 'w') as f:
            for line in train_filenames:
                f.write(f"{line}\n")

        with open('val_filenames.txt', 'w') as f:
            for line in val_filenames:
                f.write(f"{line}\n")
    else:
        with open('train_filenames.txt') as f:
            train_filenames = [string[:-1] for string in f.readlines()]
        with open('val_filenames.txt') as f:
            val_filenames = [string[:-1] for string in f.readlines()]
            
    train_dataset = faster_rcnn_dataset(data_dict, train_filenames, image_folder_path)
    val_dataset = faster_rcnn_dataset(data_dict, val_filenames, image_folder_path)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=0)

    model_base = vgg16_base().to(device)
    model_base.train()
    rpn = Region_Proposal_Network().to(device)
    rpn.train()
    lr = 0.001
    epochs = 80000
    lr_decrease_epoch = 60000
            
    params = list(model_base.parameters()) + list(rpn.parameters())
    optimizer = optim.SGD(params, lr=lr, momentum=0.9, weight_decay = 0.0005)

    children = list(model_base.children())
    for child in children[:5]:
        if isinstance(child, nn.Conv2d):
            for param in child.parameters():
                param.requires_grad = False

    train_loop(epochs, lr_decrease_epoch)
