from numba import cuda, njit
from operator import pow
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

print(cuda.gpus)

background_path = "../datasets/trash_composition/backgrounds/Pavement/20220411_090257.jpg"
foreground_path1 = "../datasets/trash_composition/foregrounds/Mask/mask/20220419_195834.png"

@njit
def load_and_resize_image(background_path):
  bg_path = str(background_path)
  background = cv2.imread(bg_path)
  # background = cv2.cvtColor(background, cv2.COLOR_BGR2RGBA)
  new_size = (int(512 * 1.1), int(512 * 1.1))
  return cv2.resize(background, new_size, interpolation=cv2.INTER_CUBIC)

@njit
def crop_image(image):
  img_width, img_height = image.shape[:2]
  max_crop_x_pos = img_width - 512
  max_crop_y_pos = img_height - 512
  crop_x_pos = random.randint(0, max_crop_x_pos)
  crop_y_pos = random.randint(0, max_crop_y_pos)
  return image[crop_x_pos:crop_x_pos + 512, crop_y_pos:crop_y_pos + 512, :]


background = load_and_resize_image(background_path)
composite = crop_image(background)