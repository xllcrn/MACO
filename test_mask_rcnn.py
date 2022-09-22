import os
import sys
from sklearn.metrics import auc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Import Mask RCNNmodel_path
sys.path.append(ROOT_DIR)  # To find local version of the library
from src.mrcnn import utils
from src.mrcnn import visualize
from src.mrcnn.visualize import display_images
import src.mrcnn.model as modellib
from src.mrcnn.model import log
from src.mrcnn.config_cig_and_mask import CigaretteMaskConfig, CocoLikeDataset

IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.7
GROUND_TRUTH = 0

def get_ax(rows=1, cols=1, size=16):
  """Return a Matplotlib Axes array to be used in
  all visualizations in the notebook. Provide a
  central point to control graph sizes.

  Adjust the size attribute to control how big to render images
  """
  _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
  return ax


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run Mask R-CNN.')
    parser.add_argument('--json_file', dest='json_file', required=True, type=str, help='annotation file')
    parser.add_argument('--img_dir', dest='img_dir', required=True, type=str, help='image directory')
    parser.add_argument('--model_dir', dest='model_dir', required=False, type=str, help='model directory')
    parser.set_defaults(aug=False)
    args = parser.parse_args()

    # Load json file and image dir
    # TRASH_DIR = os.path.join(ROOT_DIR, "./datasets/trash_split/")
    # json_file = os.path.join(TRASH_DIR, "annotations/coco_annotations_test.json")
    # img_dir = Path(os.path.join(TRASH_DIR, "images/test"))
    img_dir = Path(args.img_dir)
    if (not img_dir.exists()):
        raise AssertionError("Images directory is not found at {}".format(img_dir))

    # Load Test Dataset
    dataset_test = CocoLikeDataset()
    dataset_test.load_data(args.json_file, img_dir)
    dataset_test.prepare()
    if (len(dataset_test.image_ids)==0):
        raise Exception("Le nombre d'images du test set est nul")
    nr_classes = dataset_test.num_classes

    print("Images: {}\nClasses: {}".format(len(dataset_test.image_ids), dataset_test.class_names))

    # Prepare config
    config = CigaretteMaskConfig()
    # Override the training configurations with a few changes for inferencing.
    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        USE_MINI_MASK = False

    config = InferenceConfig()
    config.display()

    # Load weights & model
    # MODEL_DIR = "./weights/logs/cig_and_mask20220910T1722"
    if (args.model_dir):
      MODEL_DIR = args.model_dir
    else:
      MODEL_DIR = "./weights/logs"
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    # model.keras_model.summary()
    # model_path = os.path.join(MODEL_DIR, "mask_rcnn_cig_and_mask_0025.h5")
    # model.load_weights(model_path, by_name=True)
    model.load_weights(model.find_last(), by_name=True)

    # Run detection
    # ======================================================
    image_ids = np.random.choice(dataset_test.image_ids, 4)

    for image_id in image_ids:

      image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_test, config, image_id)
      # image and mask have been resized in load_image_gt
      info = dataset_test.image_info[image_id]

      print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
                                             dataset_test.image_reference(image_id)))

      # convert pixel values (e.g. center)
      scaled_image = modellib.mold_image(image, config)

      # Run object detection
      results = model.detect([scaled_image], verbose=0)

      # Display results
      ax = get_ax(1, 2)
      visualize.draw_boxes(
        image,
        boxes=gt_bbox,
        title="Ground true box " + info['name'],
        ax=ax[0])

      r = results[0]
      visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                  dataset_test.class_names, r['scores'], ax=ax[1],
                                  title="Predictions de " + info['name'])

      AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"],
                                                           r["scores"], r['masks'], iou_threshold=0.5)

      log("gt_class_id", gt_class_id)
      log("gt_bbox", gt_bbox)
      log("gt_mask", gt_mask)
      plt.show()

    # filename = Path(model.find_last()).parent.stem + ".png"
    # plot_file = Path(model.find_last()).parent / filename
    # print("Plot saved in the file: " + str(plot_file))
    # plt.savefig(plot_file)
    # plt.show()
