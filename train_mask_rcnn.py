import os
import sys
import json
import numpy as np
import time
from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import mrcnn libraries
sys.path.append(ROOT_DIR)

import src.mrcnn.utils as utils
from src.mrcnn import visualize
import src.mrcnn.model as modellib
from src.mrcnn.config_cig_and_mask import CigaretteMaskConfig, CocoLikeDataset

# Set the WEIGHTS_DIR variable to the root directory of the Mask_RCNN git repo
WEIGHTS_DIR = './weights'
assert os.path.exists(WEIGHTS_DIR), 'WEIGHTS_DIR does not exist. Did you forget to read the instructions above? ;)'

# Directory to save logs and trained model
MODEL_DIR = os.path.join(WEIGHTS_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(WEIGHTS_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run Mask R-CNN.')
    parser.add_argument('--lrate', required=False, default=0.001, type=float, help='learning rate')
    parser.add_argument('--use_aug', dest='aug', action='store_true')
    parser.set_defaults(aug=False)
    args = parser.parse_args()

    config = CigaretteMaskConfig()

    # Path to dataset
    train_json_file = './datasets/trash_split/annotations/coco_annotations_train.json'
    train_img_dir = './datasets/trash_split/images/train'
    val_json_file = './datasets/trash_split/annotations/coco_annotations_val.json'
    val_img_dir = './datasets/trash_split/images/val'

    # Prepare training set
    dataset_train = CocoLikeDataset()
    dataset_train.load_data(train_json_file,train_img_dir)
    dataset_train.prepare()
    if (len(dataset_train.image_ids)!=0):
        print("Number of images in the training set :", len(dataset_train.image_ids))
    else:
        raise Exception("W : No images in the training set")
    nr_classes = dataset_train.num_classes
    print("Number of categories in the training set :", nr_classes)

    # Prepare validation set
    dataset_val = CocoLikeDataset()
    dataset_val.load_data(val_json_file,val_img_dir)
    dataset_val.prepare()
    if (len(dataset_val.image_ids)!=0):
        print("Number of images in the val set :", len(dataset_val.image_ids))
    else:
        raise Exception("W : No images in the val set")

    class TrainConfig(config.__class__):
      IMAGES_PER_GPU = 1
      GPU_COUNT = 1
      STEPS_PER_EPOCH = min(1000, int(dataset_train.num_images / (IMAGES_PER_GPU * GPU_COUNT)))
      # USE_MINI_MASK = True
      # MINI_MASK_SHAPE = (512, 512)
      # USE_OBJECT_ZOOM = False
      NUM_CLASSES = nr_classes
      LEARNING_RATE = args.lrate
    config = TrainConfig()
    config.display()

    # Visualization of masks
    dataset = dataset_train
    image_ids = np.random.choice(dataset.image_ids, 4)
    for image_id in image_ids:
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)
        visualize.display_top_masks(image, mask, class_ids, dataset.class_names)

    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,model_dir=MODEL_DIR)

    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)

    # if args.aug:
    #     if not config.USE_OBJECT_ZOOM:
    #         # Image Augmentation Pipeline
    #         augmentation_pipeline = iaa.Sequential([
    #             iaa.AdditiveGaussianNoise(scale=0.01 * 255, name="AWGN"),
    #             iaa.GaussianBlur(sigma=(0.0, 3.0), name="Blur"),
    #             # iaa.Dropout([0.0, 0.05], name='Dropout'), # drop 0-5% of all pixels
    #             iaa.Fliplr(0.5),
    #             iaa.Add((-20, 20),name="Add"),
    #             iaa.Multiply((0.8, 1.2), name="Multiply"),
    #             iaa.Affine(scale=(0.8, 2.0)),
    #             iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
    #             iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees
    #         ], random_order=True)
    #     else:
            # Nevermind the image translation and scaling as this is done already during zoom in
    if args.aug:
        augmentation_pipeline = iaa.Sequential([
            iaa.AdditiveGaussianNoise(scale=0.01 * 255, name="AWGN"),
            iaa.GaussianBlur(sigma=(0.0, 3.0), name="Blur"),
            # iaa.Dropout([0.0, 0.05], name='Dropout'), # drop 0-5% of all pixels
            iaa.Fliplr(0.5),
            iaa.Add((-20, 20), name="Add"),
            iaa.Multiply((0.8, 1.2), name="Multiply"),
            iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees
        ], random_order=True)
    else:
        augmentation_pipeline = None

    # Save training meta to log dir
    training_meta = {
        'number of classes': nr_classes,
        'use_augmentation': args.aug,
        'learning_rate': config.LEARNING_RATE,
        'layers_trained': 'heads',
        'json_train': train_json_file,
        'img_train': train_img_dir,
        'json_val': val_json_file,
        'img_val': val_img_dir
        }
    subdir = os.path.dirname(model.log_dir)
    if not os.path.isdir(subdir):
      os.mkdir(subdir)

    if not os.path.isdir(model.log_dir):
      os.mkdir(model.log_dir)

    train_meta_file = model.log_dir + '_meta.json'
    with open(train_meta_file, 'w+') as f:
      f.write(json.dumps(training_meta))

    print("Save information of training parameters in ",train_meta_file)

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    start_train = time.time()
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=25,
                layers='heads',
                augmentation=augmentation_pipeline
                )

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    # print("Fine tune Resnet stage 4 and up")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=0.0001,
    #             epochs=30,
    #             layers='3+',
    #             augmentation=augmentation_pipeline)
    #
    # # Training - Stage 3
    # # Fine tune all layers
    # print("Fine tune all layers")
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE / 10,
    #             epochs=28,
    #             layers='all',
    #             augmentation=augmentation_pipeline)

    end_train = time.time()
    minutes = round((end_train - start_train) / 60, 2)
    print(f'Training took {minutes} minutes')