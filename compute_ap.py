import os
import sys
from sklearn.metrics import auc
import glob
import os
import pandas as pd
import shutil
import matplotlib.pyplot as plt
from pathlib import Path
import tensorflow as tf
from tqdm import tqdm

# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Import mrcnn libraries
sys.path.append(ROOT_DIR)

from src.metrics.BoundingBoxes import BoundingBoxes
from src.metrics.BoundingBox import BoundingBox
from src.metrics.Evaluator import Evaluator
from src.metrics.utils import BBType, BBFormat, CoordinatesType, MethodAveragePrecision


# Root directory of the project
ROOT_DIR = os.path.abspath(".")

# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
# DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0
DEVICE = "/gpu:0"
# Import Mask RCNNmodel_path
sys.path.append(ROOT_DIR)  # To find local version of the library
from src.mrcnn import utils
from src.mrcnn import visualize
from src.mrcnn.visualize import display_images
import src.mrcnn.model as modellib
from src.mrcnn.model import log
from src.mrcnn.config_cig_and_mask import CigaretteMaskConfig, CocoLikeDataset

SCORE_THRESHOLD = 0.7
GROUND_TRUTH = 0

# Get current path to set default folders
currentPath = os.path.dirname(os.path.abspath(__file__))


def get_ax(rows=1, cols=1, size=16):
  """Return a Matplotlib Axes array to be used in
  all visualizations in the notebook. Provide a
  central point to control graph sizes.

  Adjust the size attribute to control how big to render images
  """
  _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
  return ax

def ValidatePaths(arg, nameArg, errors):
  if arg is None:
    errors.append('argument %s: invalid directory' % nameArg)
  elif os.path.isdir(arg) is False and os.path.isdir(os.path.join(currentPath, arg)) is False:
    errors.append('argument %s: directory does not exist \'%s\'' % (nameArg, arg))
  # elif os.path.isdir(os.path.join(currentPath, arg)) is True:
  #     arg = os.path.join(currentPath, arg)
  else:
    arg = os.path.join(currentPath, arg)
  return arg



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run Mask R-CNN.')
    # parser.add_argument('--json_file', dest='json_file', required=True, type=str, help='annotation file')
    # parser.add_argument('--img_dir', dest='img_dir', required=True, type=str, help='image directory')
    # parser.add_argument('--model_dir', dest='model_dir', required=False, type=str, help='model directory')
    # parser.add_argument('--iouThreshold', dest='iouThreshold', required=False, type=int, help='iou Threshold')
    # parser.set_defaults(aug=False)
    # Optional
    parser.add_argument('-t',
                        '--threshold',
                        dest='iouThreshold',
                        type=float,
                        default=0.5,
                        metavar='',
                        help='IOU threshold. Default 0.5')
    parser.add_argument('-sp',
                        '--savepath',
                        dest='savePath',
                        metavar='',
                        help='folder where the plots are saved')
    parser.add_argument('-np',
                        '--noplot',
                        dest='showPlot',
                        action='store_false',
                        help='no plot is shown during execution')
    args = parser.parse_args()

    iouThreshold = args.iouThreshold

    errors = []
    if args.savePath is not None:
      savePath = ValidatePaths(args.savePath, '-sp/--savepath', errors)
    else:
      savePath = os.path.join(currentPath, 'results')

    # Validate savePath
    # If error, show error messages
    if len(errors) != 0:
      print("""usage: Object Detection Metrics [-h] [-v] [-gt] [-det] [-t] [-gtformat]
                                    [-detformat] [-save]""")
      print('Object Detection Metrics: error(s): ')
      [print(e) for e in errors]
      sys.exit()

    # Check if path to save results already exists and is not empty
    if os.path.isdir(savePath) and os.listdir(savePath):
      key_pressed = ''
      while key_pressed.upper() not in ['Y', 'N']:
        print(f'Folder {savePath} already exists and may contain important results.\n')
        print(f'Enter \'Y\' to continue. WARNING: THIS WILL REMOVE ALL THE CONTENTS OF THE FOLDER!')
        print(f'Or enter \'N\' to abort and choose another folder to save the results.')
        key_pressed = input('')

      if key_pressed.upper() == 'N':
        print('Process canceled')
        sys.exit()

    # Clear folder and save results
    shutil.rmtree(savePath, ignore_errors=True)
    os.makedirs(savePath)
    # Show plot during execution
    showPlot = args.showPlot

    # img_dir = args.img_dir
    # json_file = args.json_file
    # iouThreshold = args.iouThreshold
    # if (args.model_dir):
    #   MODEL_DIR = args.model_dir
    # else:

    MODEL_DIR = "./weights/logs"

    # Load json file and image dir
    TRASH_DIR = os.path.join(ROOT_DIR, "./datasets/trash_split/")
    json_file = os.path.join(TRASH_DIR, "annotations/coco_annotations_test.json")
    img_dir = Path(os.path.join(TRASH_DIR, "images/test"))
    img_dir = Path(img_dir)
    if (not img_dir.exists()):
        raise AssertionError("Images directory is not found at {}".format(img_dir))

    # Load Test Dataset
    dataset_test = CocoLikeDataset()
    dataset_test.load_data(json_file, img_dir)
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

    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    # model.keras_model.summary()
    # model_path = os.path.join(MODEL_DIR, "mask_rcnn_cig_and_mask_0025.h5")
    # model.load_weights(model_path, by_name=True)
    model.load_weights(model.find_last(), by_name=True)

    # Run detection
    # ======================================================
    # image_ids = np.random.choice(dataset_test.image_ids, 4)
    allBoundingBoxes = BoundingBoxes()
    for i in tqdm(range(len(dataset_test.image_ids))):
        image_id = dataset_test.image_ids[i]
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_test, config, image_id)
        # image and mask have been resized in load_image_gt
        info = dataset_test.image_info[image_id]
        nameOfImage = info["name"]
        width = info["width"]
        height = info["height"]

        # print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
        #                                        dataset_test.image_reference(image_id)))

        # convert pixel values (e.g. center)
        scaled_image = modellib.mold_image(image, config)

        # Run object detection
        results = model.detect([scaled_image], verbose=0)

        # gt bbox
        for j, bbox in enumerate(gt_bbox):
            y1, x1, y2, x2 = bbox
            idClass = gt_class_id[j]
            bb = BoundingBox(nameOfImage,
                             idClass,
                             x1, y1, x2 - x1, y2 - y1,
                             CoordinatesType.Absolute,
                             (width, height),
                             BBType.GroundTruth,
                             format=BBFormat.XYWH)
            allBoundingBoxes.addBoundingBox(bb)

        # predicted bbox
        r = results[0]
        for j, bbox in enumerate(r["rois"]):
            y1, x1, y2, x2 = bbox
            idClass = r["class_ids"][j]
            bb = BoundingBox(nameOfImage,
                             idClass,
                             x1, y1, x2 - x1, y2 - y1,
                             CoordinatesType.Absolute,
                             scaled_image.shape,
                             BBType.Detected,
                             r["scores"][j],
                             format=BBFormat.XYWH)
            allBoundingBoxes.addBoundingBox(bb)

      # A prediction is said to be correct if :
      #     - the class label of the predicted bounding box
      #       and the ground truth bounding box is the same
      #     - the IoU between them is greater than a threshold value

      # TP : The model predicted that a bounding box exists at a certain position (positive) and it was correct (true)
      # FP : The model predicted that a bounding box exists at a particular position (positive) but it was wrong (false)
      # FN : False Negative: The model did not predict a bounding box at a certain position (negative) and it was wrong (false)
      #      i.e. a ground truth bounding box existed at that position
      # TN : The model did not predict a bounding box (negative) and it was correct (true). This corresponds to the background,
      #      the area without bounding boxes, and is not used to calculate the final metrics

  # Create an evaluator object in order to obtain the metrics
    evaluator = Evaluator()

    # results = evaluator.GetPascalVOCMetrics(allBoundingBoxes, iouThreshold, MethodAveragePrecision.EveryPointInterpolation)
    # print(results)
    acc_AP = 0
    validClasses = 0

    # Plot Precision x Recall curve
    detections = evaluator.PlotPrecisionRecallCurve(
      allBoundingBoxes,  # Object containing all bounding boxes (ground truths and detections)
      IOUThreshold=iouThreshold,  # IOU threshold
      method=MethodAveragePrecision.EveryPointInterpolation,
      showAP=True,  # Show Average Precision in the title of the plot
      showInterpolatedPrecision=False,  # Don't plot the interpolated precision curve
      savePath=savePath,
      showGraphic=showPlot)

    f = open(os.path.join(savePath, 'results.txt'), 'w')
    f.write('Object Detection Metrics\n')
    f.write('https://github.com/rafaelpadilla/Object-Detection-Metrics\n\n\n')
    f.write('Average Precision (AP), Precision and Recall per class:')

    # each detection is a class
    for metricsPerClass in detections:

      # Get metric values per each class
      cl = metricsPerClass['class']
      ap = metricsPerClass['AP']
      precision = metricsPerClass['precision']
      recall = metricsPerClass['recall']
      totalPositives = metricsPerClass['total positives']
      total_TP = metricsPerClass['total TP']
      total_FP = metricsPerClass['total FP']

      if totalPositives > 0:
        validClasses = validClasses + 1
        acc_AP = acc_AP + ap
        prec = ['%.2f' % p for p in precision]
        rec = ['%.2f' % r for r in recall]
        ap_str = "{0:.2f}%".format(ap * 100)
        # ap_str = "{0:.4f}%".format(ap * 100)
        print('AP: %s (%s)' % (ap_str, cl))
        f.write('\n\nClass: %s' % cl)
        f.write('\nAP: %s' % ap_str)
        f.write('\nPrecision: %s' % prec)
        f.write('\nRecall: %s' % rec)

    mAP = acc_AP / validClasses
    mAP_str = "{0:.2f}%".format(mAP * 100)
    print('mAP: %s' % mAP_str)
    f.write('\n\n\nmAP: %s' % mAP_str)