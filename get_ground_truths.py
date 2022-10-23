import xml.etree.ElementTree as ET
import os
from os import listdir
from os.path import join

from PIL import Image
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
%matplotlib inline

from selective_search import selective_search
import time

my_path = "C:\\my_data\\Pascal_Voc\\VOCdevkit\\VOC2012\\Annotations"
annotation_files = listdir(my_path)

for current_img_num, file in enumerate(annotation_files):
    tree = ET.parse(join(my_path, file))
    root = tree.getroot()
    filename = root.find("filename").text
    size = root.find("size")
    height = int(size.find("height").text)
    width = int(size.find("width").text)
    x_scale = width/224
    y_scale = height/224
    for item in root.findall("object"):
        name = item.find("name").text
        bndbox = item.find("bndbox")
        xmin = round(float(bndbox.find("xmin").text)/x_scale)
        ymin = round(float(bndbox.find("ymin").text)/y_scale)
        xmax = round(float(bndbox.find("xmax").text)/x_scale)
        ymax = round(float(bndbox.find("ymax").text)/y_scale)
        with open("ground_truths.txt","a+") as f:
            f.write("{} {} {} {} {} {}\n".format(filename, xmin, ymin, xmax, ymax, name))
