from mmpose.datasets.builder import DATASETS
from ..top_down import TopDownCocoDataset
import numpy as np
import os.path as osp

@DATASETS.register_module()
class LPCornersDataset(TopDownCocoDataset):
  """License plate corner detection dataset"""
  
  # Taken from the superclass and adapted to also pass on the char_bbox annotation
  def _get_db(self):
    """Load dataset."""
    if (not self.test_mode) or self.use_gt_bbox:
      # use ground truth bbox
      gt_db = self._load_coco_keypoint_annotations()
    else:
      # use bbox from detection
      gt_db = self._load_coco_person_detection_results()
    return gt_db

  def _load_coco_keypoint_annotations(self):
    """Ground truth bbox and keypoints."""
    gt_db = []
    for img_id in self.img_ids:
      gt_db.extend(self._load_coco_keypoint_annotation_kernel(img_id))
    return gt_db

  def _load_coco_keypoint_annotation_kernel(self, img_id):
    """load annotation from COCOAPI.

    Note:
        bbox:[x1, y1, w, h]

    Args:
        img_id: coco image id

    Returns:
        dict: db entry
    """
    img_ann = self.coco.loadImgs(img_id)[0]
    width = img_ann['width']
    height = img_ann['height']
    num_joints = self.ann_info['num_joints']

    ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
    objs = self.coco.loadAnns(ann_ids)

    # sanitize bboxes
    valid_objs = []
    for obj in objs:
      if 'bbox' not in obj:
        continue
      x, y, w, h = obj['bbox']
      x1 = max(0, x)
      y1 = max(0, y)
      x2 = min(width - 1, x1 + max(0, w - 1))
      y2 = min(height - 1, y1 + max(0, h - 1))
      if ('area' not in obj or obj['area'] > 0) and x2 > x1 and y2 > y1:
        obj['clean_bbox'] = [x1, y1, x2 - x1, y2 - y1]
        valid_objs.append(obj)
    objs = valid_objs

    bbox_id = 0
    rec = []
    for obj in objs:
      if 'keypoints' not in obj:
        continue
      if max(obj['keypoints']) == 0:
        continue
      if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
        continue
      joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
      joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)

      keypoints = np.array(obj['keypoints']).reshape(-1, 3)
      joints_3d[:, :2] = keypoints[:, :2]
      joints_3d_visible[:, :2] = np.minimum(1, keypoints[:, 2:3])

      center, scale = self._xywh2cs(*obj['clean_bbox'][:4])

      image_file = osp.join(self.img_prefix, self.id2name[img_id])
      rec.append({
        'image_file': image_file,
        'center': center,
        'scale': scale,
        'bbox': obj['clean_bbox'][:4],
        'rotation': 0,
        'joints_3d': joints_3d,
        'joints_3d_visible': joints_3d_visible,
        'dataset': self.dataset_name,
        'bbox_score': 1,
        'bbox_id': bbox_id,
        'char_bbox': obj['char_bbox']
      })
      bbox_id = bbox_id + 1

    return rec