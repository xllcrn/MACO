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

    ann_input_path = Path(r"F:\Deep_Learning\Mask-RCNN-inspired-by-taco\datasets\taco\annotations.json")
    img_input_path = Path(r"F:\Deep_Learning\Mask-RCNN-inspired-by-taco\datasets\taco")
    output_path = Path(r"F:\Deep_Learning\Mask-RCNN-inspired-by-taco\datasets\taco\taco_dataset")

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


    # Load annotations
    with open(ann_input_path, 'r') as f:
        dataset = json.loads(f.read())

    anns = dataset['annotations']
    if 'scene_annotations' in dataset:
        scene_anns = dataset['scene_annotations']
    else:
        scene_anns=[]

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

    i = 0
    dic_id={}
    for im in tqdm(dataset['images']):
        new_im = copy.deepcopy(im)
        # old
        filename = new_im["file_name"]
        image_id = new_im["id"]
        # new
        new_im["id"] = i
        ext = filename.split(".")[-1]
        new_file = f"{new_im['id']:08d}"+'.'+ ext
        new_im["file_name"] = new_file
        # save id (old : new)
        dic_id[image_id] = new_im["id"]
        # copy files
        img_train_path1 = os.path.join(img_input_path, filename)
        img_train_path2 = os.path.join(IMG_DIR, new_file)
        # print(img_train_path1,img_train_path2)
        shutil.copyfile(img_train_path1, img_train_path2)
        # copy list images
        train_set['images'].append(new_im)
        i += 1

    # for dico in dataset['images']:
    #     print(dico["file_name"])

    # instance annotations
    for ann in dataset['annotations']:
        new_ann = copy.deepcopy(ann)
        image_id = new_ann["image_id"]
        new_ann["image_id"] = dic_id[image_id]
        train_set['annotations'].append(new_ann)

    # scene tags
    for ann in scene_anns:
        new_ann = copy.deepcopy(ann)
        image_id = new_ann["image_id"]
        new_ann["image_id"] = dic_id[image_id]
        train_set['scene_annotations'].append(new_ann)

    # Write new json file
    ann_train_out_path = os.path.join(ANN_DIR, "coco_annotations_taco.json")

    with open(ann_train_out_path, 'w+') as f:
        f.write(json.dumps(train_set))

    # copy of train images into /train/images/
    # for img in dataset['images']:
    #     train_img_name = img['file_name']
    #     # print(train_img_name)
    #     new_img_name = str(Path(train_img_name).name)
    #     img_train_path1 = os.path.join(img_input_path, train_img_name)
    #     img_train_path2 = os.path.join(IMG_DIR, new_img_name)
        # print(img_train_path1,img_train_path2)
        # shutil.copyfile(img_train_path1, img_train_path2)
