# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import xml.etree.ElementTree as ET
import math
import numpy as np
from configparser import ConfigParser

import mmcv

from mmocr.utils import convert_annotations


def collect_files(img_dir, gt_dir):
    """Collect all images and their corresponding groundtruth files.

    Args:
        img_dir (str): The image directory
        gt_dir (str): The groundtruth directory

    Returns:
        files (list): The list of tuples (img_file, groundtruth_file)
    """
    assert isinstance(img_dir, str)
    assert img_dir
    assert isinstance(gt_dir, str)
    assert gt_dir

    ann_list, imgs_list = [], []
    for img_file in os.listdir(img_dir):
        ann_path = osp.join(gt_dir, osp.splitext(img_file)[0] + '.ini')
        if os.path.exists(ann_path):
            ann_list.append(ann_path)
            imgs_list.append(osp.join(img_dir, img_file))

    files = list(zip(imgs_list, ann_list))
    assert len(files), f'No images found in {img_dir}'
    print(f'Loaded {len(files)} images from {img_dir}')

    return files


def collect_annotations(files, nproc=1):
    """Collect the annotation information.

    Args:
        files (list): The list of tuples (image_file, groundtruth_file)
        nproc (int): The number of process to collect annotations

    Returns:
        images (list): The list of image information dicts
    """
    assert isinstance(files, list)
    assert isinstance(nproc, int)

    if nproc > 1:
        images = mmcv.track_parallel_progress(
            load_img_info, files, nproc=nproc)
    else:
        images = mmcv.track_progress(load_img_info, files)

    return images


def load_img_info(files):
    """Load the information of one image.

    Args:
        files (tuple): The tuple of (img_file, groundtruth_file)

    Returns:
        img_info (dict): The dict of the img and annotation information
    """
    assert isinstance(files, tuple)

    img_file, gt_file = files
    assert osp.basename(gt_file).split('.')[0] == osp.basename(img_file).split(
        '.')[0]
    # read imgs while ignoring orientations
    img = mmcv.imread(img_file, 'unchanged')

    try:
        img_info = dict(
            file_name=osp.join(osp.basename(img_file)),
            height=img.shape[0],
            width=img.shape[1],
            segm_file=osp.join(osp.basename(gt_file)))
    except AttributeError:
        print(f'Skip broken img {img_file}')
        return None

    if osp.splitext(gt_file)[1] == '.ini':
        img_info = load_ini_info(gt_file, img_info)
    else:
        raise NotImplementedError

    return img_info

def sort_vertex(vertices):
  assert vertices.ndim == 2
  assert vertices.shape[-1] == 2
  N = vertices.shape[0]
  if N == 0:
    return vertices

  center = np.mean(vertices, axis=0)
  directions = vertices - center
  angles = np.arctan2(directions[:, 1], directions[:, 0])
  sort_idx = np.argsort(angles)
  vertices = vertices[sort_idx]

  left_top = np.min(vertices, axis=0)
  dists = np.linalg.norm(left_top - vertices, axis=-1, ord=2)
  lefttop_idx = np.argmin(dists)
  indexes = (np.arange(N, dtype=int) + lefttop_idx) % N
  return vertices[indexes]

def parse_annotation(fn):
  cp = ConfigParser()
  cp.optionxform = str
  cp.read(fn)
  n_obj = cp.getint('General', 'ObjectsCount')
  if any((
    'Position' not in cp['General'], # we lack Position annotation, the sample is useless.
    n_obj != 1                       # we have no (complete) or multiple objects in the image.
    )):
    return [],[] 
  
  return ([ # polygon and bbox
    np.asarray(list(map(float, cp[f'OBJECT{i}']['Polyline'].split(' ')))).reshape(-1,2)
    for i in range(n_obj)
  ], np.asarray(list(map(int, cp['General']['Position'].split(' '))) ))

def load_ini_info(gt_file, img_info):
    """Collect the annotation information.

    The annotation format is as the following (only relevant keys shown):
    [General]
    Position = 85 72 262 101                                                     # BBox of detected characters (x1,y1,x2,y2)
    ObjectsCount =1                                                              # Number of objects in the annotation (must be 1)        
    [OBJECT0]
    Polyline =60.0319 68.1914 272.745 65.6515 271.729 105.856 60.5399 109.412    # Annotated corner points (x,y * 4)

    Args:
        gt_file (str): The path to ground-truth
        img_info (dict): The dict of the img and annotation information

    Returns:
        img_info (dict): The dict of the img and annotation information
    """
    poly, char_bbox_xyxy = parse_annotation(gt_file)
    if not poly:
        # annotation was empty, mark sample to be skipped
        return None
    poly = sort_vertex(poly[0])
    min_p = poly.min(axis=0)
    max_p = poly.max(axis=0)
    bbox = [*min_p.tolist(), *(max_p-min_p).tolist()]
    w, h = max_p - min_p
    anno_info = [dict(
        iscrowd=0,
        category_id=1,
        bbox=bbox,
        char_bbox=char_bbox_xyxy, # Additional "extra info" field that I can use later in the data pipeline if I want.
        area=w * h,
        segmentation=[poly.flatten().tolist()],
        keypoints=sum([r.tolist()+[2] for r in poly], []),
        num_keypoints=4
    )]
    img_info.update(anno_info=anno_info)
    return img_info

def rotatePoint(xc, yc, xp, yp, theta):        
    xoff = xp-xc
    yoff = yp-yc

    cosTheta = math.cos(theta)
    sinTheta = math.sin(theta)
    pResx = cosTheta * xoff + sinTheta * yoff
    pResy = - sinTheta * xoff + cosTheta * yoff
    return xc+pResx,yc+pResy


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate training and val set of a PascalVOC-compatible dataset')
    parser.add_argument('root_path', help='Root dir path of the dataset')
    parser.add_argument(
        '--nproc', default=1, type=int, help='Number of processes')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    root_path = args.root_path
    with mmcv.Timer(print_tmpl='It takes {}s to convert annotations'):
        for split in ['train', 'test']:
            files = collect_files(
                osp.join(root_path, 'images', split), osp.join(root_path, 'annos', split))
            image_infos = collect_annotations(files, nproc=args.nproc)
            convert_annotations(
                list(filter(None, image_infos)),
                osp.join(root_path, 'instances_' + split + '.json'))


if __name__ == '__main__':
    main()
