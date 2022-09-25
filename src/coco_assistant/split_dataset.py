import os.path
import json
import argparse
import numpy as np
import random
import datetime as dt
import copy
import shutil
from pathlib import Path
from tqdm import tqdm

"""
Create train/val/test sets from an image directory and the associated annotations.json file.
The structure of the new directory is the following:
|──output_dir
      |──annotations/
           |──coco_annotations_train.json
           |──coco_annotations_val.json
           |──coco_annotations_test.json
      |──images/
           |──train/
           |──val/
           |──test/
"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='User args')
    parser.add_argument('--image_dir', required=True, help='Path to image directory')
    parser.add_argument('--output_dir', required=True, help='Path to new directory')
    parser.add_argument('--json_file', required=True, help='Path to annotation file')
    parser.add_argument('--test_percentage', type=int, default=10, required=False, help='Percentage of images used for the testing set')
    parser.add_argument('--val_percentage', type=int, default=10, required=False, help='Percentage of images used for the validation set')

    args = parser.parse_args()

    ann_input_path = Path(args.json_file)
    img_input_path = Path(args.image_dir)
    output_path = Path(args.output_dir)

    # Load annotations
    with open(ann_input_path, 'r') as f:
        dataset = json.loads(f.read())

    anns = dataset['annotations']
    if 'scene_annotations' in dataset:
        scene_anns = dataset['scene_annotations']
    else:
        scene_anns=[]
    imgs = dataset['images']
    nr_images = len(imgs)

    nr_testing_images = int(nr_images*args.test_percentage*0.01+0.5)
    nr_nontraining_images = int(nr_images*(args.test_percentage+args.val_percentage)*0.01+0.5)

    random.shuffle(imgs)

    if scene_anns:
        # Add new datasets
        train_set = {
            'info': None,
            'images': [],
            'annotations': [],
            'scene_annotations': [],
            'licenses': [],
            'categories': [],
            'scene_categories': [],
        }
        train_set['scene_categories'] = dataset['scene_categories']
    else:
        # Add new datasets
        train_set = {
            'info': None,
            'images': [],
            'annotations': [],
            'licenses': [],
            'categories': [],
        }

    train_set['info'] = dataset['info']
    train_set['categories'] = dataset['categories']

    val_set = copy.deepcopy(train_set)
    test_set = copy.deepcopy(train_set)

    test_set['images'] = imgs[0:nr_testing_images]
    val_set['images'] = imgs[nr_testing_images:nr_nontraining_images]
    train_set['images'] = imgs[nr_nontraining_images:nr_images]

    # Aux Image Ids to split annotations
    test_img_ids, val_img_ids, train_img_ids = [],[],[]
    for img in test_set['images']:
        test_img_ids.append(img['id'])

    for img in val_set['images']:
        val_img_ids.append(img['id'])

    for img in train_set['images']:
        train_img_ids.append(img['id'])

    # Split instance annotations
    for ann in anns:
        if int(ann['image_id']) in test_img_ids:
            test_set['annotations'].append(ann)
        elif int(ann['image_id']) in val_img_ids:
            val_set['annotations'].append(ann)
        elif int(ann['image_id']) in train_img_ids:
            train_set['annotations'].append(ann)

    # Split scene tags
    for ann in scene_anns:
        if ann['image_id'] in test_img_ids:
            test_set['scene_annotations'].append(ann)
        elif ann['image_id'] in val_img_ids:
            val_set['scene_annotations'].append(ann)
        elif ann['image_id'] in train_img_ids:
            train_set['scene_annotations'].append(ann)

    if not output_path.exists():
        output_path.mkdir()
    else:
        print("The output directory {} already exists...".format(output_path))

    IMG_DIR = os.path.join(output_path, "images")
    if not os.path.isdir(IMG_DIR):
      os.mkdir(IMG_DIR)
    ANN_DIR = os.path.join(output_path, "annotations")
    if not os.path.isdir(ANN_DIR):
      os.mkdir(ANN_DIR)

    TRAIN_DIR = os.path.join(IMG_DIR, "train")
    if not os.path.isdir(TRAIN_DIR):
      os.mkdir(TRAIN_DIR)
    TEST_DIR = os.path.join(IMG_DIR, "test")
    if not os.path.isdir(TEST_DIR):
      os.mkdir(TEST_DIR)
    VAL_DIR = os.path.join(IMG_DIR, "val")
    if not os.path.isdir(VAL_DIR):
      os.mkdir(VAL_DIR)

    # Write dataset splits
    ann_train_out_path = os.path.join(ANN_DIR, "coco_annotations_train.json")
    ann_val_out_path = os.path.join(ANN_DIR, "coco_annotations_val.json")
    ann_test_out_path = os.path.join(ANN_DIR, "coco_annotations_test.json")

    with open(ann_train_out_path, 'w+') as f:
        f.write(json.dumps(train_set))

    with open(ann_val_out_path, 'w+') as f:
        f.write(json.dumps(val_set))

    with open(ann_test_out_path, 'w+') as f:
        f.write(json.dumps(test_set))

    # copy of train images into /train/images/
    for img in train_set['images']:
        train_img_name=img['file_name']
        img_train_path1 = os.path.join(img_input_path, train_img_name)
        img_train_path2 = os.path.join(TRAIN_DIR, train_img_name)
        shutil.copyfile(img_train_path1, img_train_path2)

    for img in test_set['images']:
        test_img_name=img['file_name']
        img_test_path1 = os.path.join(img_input_path, test_img_name)
        img_test_path2 = os.path.join(TEST_DIR, test_img_name)
        shutil.copyfile(img_test_path1, img_test_path2)

    for img in val_set['images']:
        val_img_name=img['file_name']
        img_val_path1 = os.path.join(img_input_path, val_img_name)
        img_val_path2 = os.path.join(VAL_DIR, val_img_name)
        shutil.copyfile(img_val_path1, img_val_path2)