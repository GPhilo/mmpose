from mmpose.datasets import PIPELINES
from PIL import Image, ImageOps
import cv2
import numpy as np

def _get_max_preds(heatmaps):
  """Get keypoint predictions from score maps.

  Note:
    batch_size: N
    num_keypoints: K
    heatmap height: H
    heatmap width: W

  Args:
    heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

  Returns:
    tuple: A tuple containing aggregated results.

    - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
    - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
  """
  assert isinstance(heatmaps,
            np.ndarray), ('heatmaps should be numpy.ndarray')
  assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

  N, K, _, W = heatmaps.shape
  heatmaps_reshaped = heatmaps.reshape((N, K, -1))
  idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
  maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

  preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
  preds[:, :, 0] = preds[:, :, 0] % W
  preds[:, :, 1] = preds[:, :, 1] // W

  preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
  return preds, maxvals

def _msra_generate_target(cfg, joints_3d, joints_3d_visible, sigma):
  """Generate the target heatmap via "MSRA" approach.

  Args:
    cfg (dict): data config
    joints_3d: np.ndarray ([num_joints, 3])
    joints_3d_visible: np.ndarray ([num_joints, 3])
    sigma: Sigma of heatmap gaussian
  Returns:
    tuple: A tuple containing targets.

    - target: Target heatmaps.
    - target_weight: (1: visible, 0: invisible)
  """
  num_joints = cfg['num_joints']
  image_size = cfg['image_size']
  W, H = cfg['image_size']
  joint_weights = cfg['joint_weights']

  target_weight = np.zeros((num_joints, 1), dtype=np.float32)
  target = np.zeros((num_joints, H, W), dtype=np.float32)

  # 3-sigma rule
  tmp_size = sigma * 3
  for joint_id in range(num_joints):
    target_weight[joint_id] = joints_3d_visible[joint_id, 0]

    feat_stride = image_size / [W, H]
    mu_x = int(joints_3d[joint_id][0] / feat_stride[0] + 0.5)
    mu_y = int(joints_3d[joint_id][1] / feat_stride[1] + 0.5)
    # Check that any part of the gaussian is in-bounds
    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
    if ul[0] >= W or ul[1] >= H or br[0] < 0 or br[1] < 0:
      target_weight[joint_id] = 0

    if target_weight[joint_id] > 0.5:
      size = 2 * tmp_size + 1
      x = np.arange(0, size, 1, np.float32)
      y = x[:, None]
      x0 = y0 = size // 2
      # The gaussian is not normalized,
      # we want the center value to equal 1
      g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

      # Usable gaussian range
      g_x = max(0, -ul[0]), min(br[0], W) - ul[0]
      g_y = max(0, -ul[1]), min(br[1], H) - ul[1]
      # Image range
      img_x = max(0, ul[0]), min(br[0], W)
      img_y = max(0, ul[1]), min(br[1], H)

      target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
  return target, target_weight

class Curve:
  '''Adaptation of straug.warp.Curve that also warps keypoints in the image'''
  def __init__(self, square_side=224, rng=None):
    self.tps = cv2.createThinPlateSplineShapeTransformer()
    self.side = square_side
    self.rng = np.random.default_rng() if rng is None else rng

  def __call__(self, img, cfg, joints_3d, joints_3d_visible, sigma=2, mag=-1, prob=1.):
    if self.rng.uniform(0, 1) > prob:
      return img

    orig_w, orig_h = img.size

    if orig_h != self.side or orig_w != self.side:
      img = img.resize((self.side, self.side), Image.BICUBIC)

    isflip = self.rng.uniform(0, 1) > 0.5
    if isflip:
      img = ImageOps.flip(img)

    # Sample parameters an get transformation
    img = np.asarray(img)
    w = self.side
    h = self.side
    w_25 = 0.25 * w
    w_50 = 0.50 * w
    w_75 = 0.75 * w

    b = [1.1, .95, .8]
    if mag < 0 or mag >= len(b):
      index = 0
    else:
      index = mag
    rmin = b[index]
    
    r = self.rng.uniform(rmin, rmin + .1) * h
    x1 = (r ** 2 - w_50 ** 2) ** 0.5
    h1 = r - x1
    
    t = self.rng.uniform(0.4, 0.5) * h
    w2 = w_50 * t / r
    hi = x1 * t / r
    h2 = h1 + hi

    sinb_2 = ((1 - x1 / r) / 2) ** 0.5
    cosb_2 = ((1 + x1 / r) / 2) ** 0.5

    w3 = w_50 - r * sinb_2
    h3 = r - r * cosb_2

    w4 = w_50 - (r - t) * sinb_2
    h4 = r - (r - t) * cosb_2

    w5 = 0.5 * w2
    h5 = h1 + 0.5 * hi
    h_50 = 0.50 * h

    srcpt = [(0, 0), (w, 0), (w_50, 0), (0, h), (w, h), (w_25, 0), (w_75, 0), (w_50, h), (w_25, h), (w_75, h),
         (0, h_50), (w, h_50)]
    dstpt = [(0, h1), (w, h1), (w_50, 0), (w2, h2), (w - w2, h2), (w3, h3), (w - w3, h3), (w_50, t), (w4, h4),
         (w - w4, h4), (w5, h5), (w - w5, h5)]

    n = len(dstpt)
    matches = [cv2.DMatch(i, i, 0) for i in range(n)]
    dst_shape = np.asarray(dstpt).reshape((-1, n, 2))
    src_shape = np.asarray(srcpt).reshape((-1, n, 2))
    self.tps.estimateTransformation(dst_shape, src_shape, matches)

    # Warp image
    img = self.tps.warpImage(img)
    img = Image.fromarray(img)

    if isflip:
      img = ImageOps.flip(img)
      rect = (0, self.side // 2, self.side, self.side)
    else:
      rect = (0, 0, self.side, self.side // 2)

    img = img.crop(rect)
    img = img.resize((orig_w, orig_h), Image.BICUBIC)

    # Warp keypoints
    # 1) Make heatmaps of the keypoints
    # 2) Warp each heatmap w/ the same parameters used for the image
    # 3) Find the max location in the warped heatmaps -> New kpt location
    # Possible optimization: Stack heatmaps in groups of 3 and process them as if they were an rgb image, then unstack.
    heatmaps, _ = _msra_generate_target(cfg, joints_3d, joints_3d_visible, sigma)

    new_kps = []
    for hm in heatmaps:
      resized = np.asarray(Image.fromarray(hm).resize((self.side, self.side), Image.BICUBIC))
      warped = self.tps.warpImage(resized)
      # warped image is still a square, need to convert coords back to original refrence frame
      # Dumb implementation: Just do the same as we do for img before getting the location
      # Optimized implementation: Calculate the correct coordnate mapping and apply it.
      # For now we use the dumb one.
      warped = Image.fromarray(warped)
      if isflip:
        warped = ImageOps.flip(warped)
        rect = (0, self.side // 2, self.side, self.side)
      else:
        rect = (0, 0, self.side, self.side // 2)

      warped = warped.crop(rect)
      warped = np.asarray(warped.resize((orig_w, orig_h), Image.BICUBIC))
      loc, _ = _get_max_preds(warped[None,None,...])

      new_kps.append(loc.squeeze().tolist()+[0])

    return img, np.asarray(new_kps)


@PIPELINES.register_module()
class CurveAugmentation:
  def __init__(self,
               curve_prob=0.3):
    self.curve_prob = curve_prob
    self.curve = Curve()
  
  def __call__(self, results):
    if np.random.rand() <= self.curve_prob:
      im, kps = self.curve(Image.fromarray(results['img']), results['ann_info'], results['joints_3d'], results['joints_3d_visible'])
      results['img'] = np.asarray(im)
      results['joints_3d'] = kps.astype(np.float32)
    return results
