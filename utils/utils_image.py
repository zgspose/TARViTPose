#!/usr/bin/python
# -*- coding:utf8 -*-
import os
import cv2
import numpy as np
from .utils_folder import create_folder, folder_exists
import os.path as osp
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from turbojpeg import TurboJPEG

jpeg = TurboJPEG()

def read_image_turbojpeg(image_path):
    if not osp.exists(image_path):
        raise Exception(f"Failed to read image from path: {image_path}")
    with open(image_path, 'rb') as file:
        jpeg_buf = file.read()
    img = jpeg.decode(jpeg_buf)
    return img  # Already a numpy array


# def read_images_parallel(image_files):
#     with ThreadPoolExecutor() as executor:
#         return list(executor.map(read_image_pil, image_files))

def read_images_parallel(image_files, format='jpg'):
    if format == 'jpg':
        with ThreadPoolExecutor() as executor:
            return list(executor.map(read_image_turbojpeg, image_files))
    else:
        with ThreadPoolExecutor() as executor:
            return list(executor.map(read_image_pil, image_files))

def read_image(image_path):
    if not osp.exists(image_path):
        raise FileNotFoundError(f"Failed to read image from path: {image_path}")
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"Failed to decode image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

def read_image_pil(image_path):
    if not osp.exists(image_path):
        raise Exception("Failed to read image from path: {}".format(image_path))
    img = Image.open(image_path).convert('RGB')
    return np.array(img)

def save_image(image_save_path, image_data):
    create_folder(os.path.dirname(image_save_path))
    return cv2.imwrite(image_save_path, image_data)#, [100])
