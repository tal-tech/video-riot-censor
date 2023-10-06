#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import forge
import cv2
import numpy as np
import time


def _img_norm(img):

    mean = np.array([[123.675, 116.28, 103.53]])
    std = np.array([[58.395, 57.12, 57.375]])
    
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    
    return img


def handler(req):
  INPUT = req['rawimg'].as_ndarray()
  img_raw = cv2.imdecode(INPUT, cv2.IMREAD_COLOR)
  resize_width = 224
  resize_height = 224

  img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
  img_resize = cv2.resize(img, (resize_width, resize_height))

  img = _img_norm(img_resize.astype(np.float32))
  img = np.rollaxis(img, 2, 0)

  return {'preprocessed_img': img.astype('float32')}

forge.run(handler)

