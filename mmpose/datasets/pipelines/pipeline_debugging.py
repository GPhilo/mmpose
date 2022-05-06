from mmpose.datasets import PIPELINES
from PIL import Image
import cv2
import numpy as np

@PIPELINES.register_module()
class VisualizeKeypoints:
  def __init__(self,
               filename='dbg.png'):
    self.filename=filename
  def __call__(self, results):
    im = results['img'].copy()
    cv2.polylines(im, results['joints_3d'][None,:,:2].astype(np.int32), True, 255)
    Image.fromarray(im).save(self.filename)
    return results