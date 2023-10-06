#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import forge
import cv2
import numpy as np
import time


def handler(req):
  probs = req['det_probs'].as_ndarray()
  pred = int(np.argmax(probs, axis=1)[0])
  score = float(np.max(probs, axis=1)[0])
  if pred != 3 and score < 0.7:
    pred = 3
    score = 1
  result = np.array([pred], dtype='int32')
  prob = np.array([score], dtype='float32')

  return {'result': result, 'prob':prob}


forge.run(handler)


