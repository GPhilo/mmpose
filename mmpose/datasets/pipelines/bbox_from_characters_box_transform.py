from mmpose.datasets import PIPELINES
import numpy as np
import logging

# Training:
# 1) Get bbox from char box w/ random W|H scaling factors

# Val:
# 1) Get bbox from char box w/ fixed W|H scaling factors


TEST_BBOX_W_SCALE_FOR_CROP = 1.0
TEST_BBOX_H_SCALE_FOR_CROP = 2.0

def xyxy2xywh(bbox):
  """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
  evaluation.

  Args:
      bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
          ``xyxy`` order.

  Returns:
      list[float]: The converted bounding boxes, in ``xywh`` order.
  """

  _bbox = bbox.tolist()
  return [
    _bbox[0],
    _bbox[1],
    _bbox[2] - _bbox[0],
    _bbox[3] - _bbox[1],
  ]

def xywh2cs(box, ann_info, padding=1.25):
  """This encodes bbox(x,y,w,h) into (center, scale)

  Args:
      x, y, w, h (float): left, top, width and height
      padding (float): bounding box padding factor

  Returns:
      center (np.ndarray[float32](2,)): center of the bbox (x, y).
      scale (np.ndarray[float32](2,)): scale of the bbox w & h.
  """
  # NOTE: This adapts the bbox to KEEP THE ASPECT RATIO OF THE ORIGINAL IMAGE.
  x,y,w,h = box
  aspect_ratio = ann_info['image_size'][0] / ann_info['image_size'][1]
  center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

  if w > aspect_ratio * h:
    h = w * 1.0 / aspect_ratio
  elif w < aspect_ratio * h:
    w = h * aspect_ratio

  # pixel std is 200.0
  scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
  # padding to include proper amount of context
  scale = scale * padding

  return center, scale

def is_in_box(pts, bbox):
  '''
    pts: ndarray 1D or 2D, shape [N,2] or [2]
    bbox: xywh bbox
  '''
  if pts.ndim == 1:
    pts = pts[None,...]
  bbox = np.asarray(bbox)
  return ((pts >= bbox[:2]) & (pts < bbox[:2]+bbox[2:])).all(axis=-1).squeeze()


@PIPELINES.register_module()
class GenerateBBoxFromCharactersBox:
  """Replace the bounding box with one generated based on characters box.

  Required keys: 'char_bbox', 'joints_3d'

  Modifies key: 'bbox', 'center', 'scale'
  """
  def __init__(self, 
               is_training=False, 
               min_h_scale=1.3,
               max_h_scale=3.0,
               min_w_scale=0.8,
               max_w_scale=1.3,
               min_center_jitter=-0.05,
               max_center_jitter=0.05,
               keep_aspect_ratio=True,
               max_attepts=100
               ):
    self.is_training = is_training
    self.keep_aspect_ratio = keep_aspect_ratio
    self.min_h_scale = min_h_scale
    self.max_h_scale = max_h_scale
    self.min_w_scale = min_w_scale
    self.max_w_scale = max_w_scale
    self.min_center_jitter = min_center_jitter
    self.max_center_jitter = max_center_jitter
    self.max_attempts = max_attepts
    self._already_warned = set()
    pass

  def __call__(self, results):
    # 1) Get char bbox in relative coordinates
    # 2) Get char bbox center and wh
    # 3) Get the scale and jitter values
    # 4) Get the new scaled and jittered bbox
    # 5) Convert back to absolute coords
    # 6) Convert to center/scale
    # NOTE: Scale in the pipeline seems to be divided by a factor 200 everywhere (dunno why)

    if 'char_bbox' not in results:
      logging.warning(f"char_bbox annotation missing, using the whole image.")
      return results

    char_bbox = np.array(results['char_bbox']) # this is in ABSOLUTE coordinates. All the code below assumes RELATIVE ones.
    H,W,*_ = results['img'].shape
    char_bbox = char_bbox / np.array([W,H,W,H]) # now it's normalized

    center = (char_bbox[...,:2]+char_bbox[...,2:])/2
    wh = char_bbox[...,2:]-char_bbox[...,:2]

    # Repeat random box selection until we cover all annotated keypoints or we run out of attempts.
    bbox_covers_keypts = False
    ntries=0
    while ntries < self.max_attempts and not bbox_covers_keypts:
      # TODO: Check that axes are correct in the operations, since this was converted from a vectorized version of the code
      if self.is_training:
        crop_bbox_scale = np.ones_like(wh) + np.concatenate([ np.random.uniform(self.min_w_scale, self.max_w_scale, [1]), np.random.uniform(self.min_h_scale, self.max_h_scale, [1])], axis=-1)
        crop_center_jitter = np.random.uniform(self.min_center_jitter, self.max_center_jitter, center.shape)
      else:
        crop_bbox_scale =  np.ones_like(wh) + np.array([TEST_BBOX_W_SCALE_FOR_CROP, TEST_BBOX_H_SCALE_FOR_CROP], np.float32)
        crop_center_jitter = np.zeros_like(center)
      crop_center = center + crop_center_jitter
      crop_wh = wh * crop_bbox_scale
      new_bbox = xyxy2xywh(np.concatenate([crop_center-crop_wh/2, crop_center+crop_wh/2], axis=-1)*np.array([W,H,W,H])) # now it's not normalized anymore
      
      # Clip bbox to the image size
      new_bbox = [max(0,new_bbox[0]), max(0, new_bbox[1]), min(W, new_bbox[2]), min(H, new_bbox[3])]

      # Ensure box covers the GT keypoints (if they are defined)
      bbox_covers_keypts = (results['joints_3d']==0).all() or is_in_box(results['joints_3d'][...,:2], new_bbox).all()
      ntries += 1

    if not bbox_covers_keypts:
      # we ran out of tries, use the whole image
      if results['image_file'] not in self._already_warned:
        logging.warning(f"Failed to find a bbox that covers all annotated keypoints for img {results['image_file']}")
        self._already_warned.add(results['image_file'])
      new_bbox = [0,0,W,H]
      

    if self.keep_aspect_ratio:
      new_center, new_scale = xywh2cs(new_bbox, results['ann_info'], 1.0)
    else:
      new_center = crop_center*np.array([W,H])
      new_bb_arr = np.asarray(new_bbox).astype(np.float32)
      new_center = new_bb_arr[:2]+new_bb_arr[:2]/2
      new_scale = new_bb_arr[:2]/200
    results['bbox'] = new_bbox
    results['center'] = new_center
    results['scale'] = new_scale
    return results