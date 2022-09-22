import os
import sys
import json
import numpy as np
import time
from PIL import Image, ImageDraw
from pathlib import Path
import cv2
from .config import Config
from . import utils

class CigaretteMaskConfig(Config):
  """Configuration for training on the cigarette butts dataset.
  Derives from the base Config class and overrides values specific
  to the cigarette butts dataset.
  """
  # Give the configuration a recognizable name
  NAME = "cig_and_mask"

  # Train on 1 GPU and 1 image per GPU. Batch size is 1 (GPUs * images/GPU).
  GPU_COUNT = 1
  IMAGES_PER_GPU = 2

  # Number of classes (including background)
  NUM_CLASSES = 1 + 1 + 1 # background + 1 (cig_butt) + 1 (mask)

  # All of our training images are 512x512
  IMAGE_MIN_DIM = 512
  IMAGE_MAX_DIM = 512

  # You can experiment with this number to see if it improves training
  STEPS_PER_EPOCH = 100

  # This is how often validation is run. If you are using too much hard drive space
  # on saved models (in the MODEL_DIR), try making this value larger.
  VALIDATION_STEPS = 5

  # Matterport originally used resnet101, but I downsized to fit it on my graphics card
  BACKBONE = 'resnet50'

  # To be honest, I haven't taken the time to figure out what these do
  RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
  TRAIN_ROIS_PER_IMAGE = 32
  MAX_GT_INSTANCES = 50
  POST_NMS_ROIS_INFERENCE = 500
  POST_NMS_ROIS_TRAINING = 1000

class CocoLikeDataset(utils.Dataset):
  """ Generates a COCO-like dataset, i.e. an image dataset annotated in the style of the COCO dataset.
      See http://cocodataset.org/#home for more information.
  """

  def load_data(self, annotation_json, images_dir):
    """ Load the coco-like dataset from json
    Args:
        annotation_json: The path to the coco annotations json file
        images_dir: The directory holding the images referred to by the json file
    """
    # Load json from file
    json_file = open(annotation_json)
    coco_json = json.load(json_file)
    json_file.close()

    # Add the class names using the base method from utils.Dataset
    source_name = "coco_like"
    for category in coco_json['categories']:
      class_id = category['id']
      class_name = category['name']
      if class_id < 1:
        print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(class_name))
        return

      self.add_class(source_name, class_id, class_name)

    # Get all annotations
    annotations = {}
    for annotation in coco_json['annotations']:
      image_id = annotation['image_id']
      if image_id not in annotations:
        annotations[image_id] = []
      annotations[image_id].append(annotation)

    # Get all images and add them to the dataset
    seen_images = {}
    for image in coco_json['images']:
      image_id = image['id']
      if image_id in seen_images:
        print("Warning: Skipping duplicate image id: {}".format(image))
      else:
        seen_images[image_id] = image
        # initialization
        image_file_name = ""
        image_path = "null"
        image_width = 0
        image_height = 0
        image_format = "null"
        image_mode = "null"
        image_annotations = 0
        try:
          image_file_name = image['file_name']
          image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
          image = cv2.imread(image_path)
          # image = Image.open(image_path)

          image_width = image.shape[1]   #image['width']
          image_height = image.shape[0]  #image['height']
          image_format = Path(image_file_name).suffix
          # image_mode = image.mode
          # check if the image has one or more annotations
          try:
              image_annotations = annotations[image_id]
              # Add the image using the base method from utils.Dataset
              self.add_image(
                source=source_name,
                image_id=image_id,
                path=image_path,
                name=image_file_name,
                width=image_width,
                height=image_height,
                # mode=image_mode,
                format=image_format,
                annotations=image_annotations
              )
          except KeyError as key:
              print("Warning: Skipping image (id: {}) with missing annotation: {}".format(image_id, key))

        except KeyError as key:
          print("Warning: Skipping image (id: {}) with missing file_name: {}".format(image_id, key))

  def load_mask(self, image_id):
    """ Load instance masks for the given image.
    MaskRCNN expects masks in the form of a bitmap [height, width, instances].
    Args:
        image_id: The id of the image to load masks for
    Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
    """
    image_info = self.image_info[image_id]
    annotations = image_info['annotations']
    instance_masks = []
    class_ids = []

    for annotation in annotations:
      class_id = annotation['category_id']
      mask = Image.new('1', (image_info['width'], image_info['height']))
      #mask = np.zeros((image_info['width'], image_info['height'],len(annotation['segmentation'])), np.uint8)
      mask_draw = ImageDraw.ImageDraw(mask, '1')
      #mask_draw =np.zeros((image_info['width'], image_info['height']), np.uint8)
      for segmentation in annotation['segmentation']:
        mask_draw.polygon(segmentation, fill=1)
        # poly = np.array(segmentation).reshape((int(len(segmentation) / 2), 2))
        # polygons.append(Polygon(poly))
        bool_array = np.array(mask) > 0
        instance_masks.append(bool_array)
        class_ids.append(class_id)

    mask = np.dstack(instance_masks)
    try :
        class_ids = np.array(class_ids, dtype=np.int32)
    except TypeError as e:
        class_ids = 0

    return mask, class_ids