
<img src="./assets/maco7.png" align="left" style='width:1000px'/>

<br></br>
<br></br>


MACO project has been designed as an extension of the [TACO dataset](https://github.com/pedropro/TACO)
coupled with a trash detection model based on the [Mask R-CNN by Matterport](https://github.com/matterport/Mask_RCNN).
It allows to create your own dataset with [basic image composition by Adam Kelly](https://github.com/akTwelve/cocosynth)
and add any class to the existing TACO dataset.


<img src="./assets/Segmentation_result.png" align="left" style='width:1000px'/>


The repository includes:
* Source code of Mask R-CNN built on FPN and ResNet50.
* Training code for MS COCO
* [Pre-trained weights for MS COCO](https://github.com/matterport/Mask_RCNN/releases)
* Jupyter notebooks to visualize the detection pipeline at every step
* ParallelModel class for multi-GPU training
* Evaluation on MS COCO metrics (AP)
* Example of training on your own dataset

## About Mask R-CNN
This is an implementation of [Mask R-CNN](https://arxiv.org/abs/1703.06870) on Python 3, Keras 2.9,
and TensorFlow 2.4. The model generates bounding boxes and segmentation masks for each instance of
an object in the image. It's based on Feature Pyramid Network (FPN) and a ResNet50 backbone.

The implementation of the [Mask R-CNN by Matterport](https://github.com/matterport/Mask_RCNN) is included 
in /src/mrcnn with a few modifications to help for better performance.


## About image composition
The implementation of the image composition script is based on the [basic image composition by Adam Kelly](https://github.com/akTwelve/cocosynth).
It is included in /src/coco_assistant with the following modifications:
* adapted to the TACO dataset structure adding scene categories for background images
* usage of OpenCV instead of PIL
* random position is made more complex to avoid too many overlaps
* foreground scaling keeps ratio of the original foreground images 

## About AP computation
The implementation of [object detection metrics by Rafael Padilla](https://github.com/rafaelpadilla/Object-Detection-Metrics) 
is included in /src/metrics.


# Trash Detection and Segmentation with Mask R-CNN


# 0. Requirements 

To install the required python packages simply type
```
pip3 install -r requirements.txt
```


# 1. Create your own dataset with Image Composition

## Getting started
[image_composition.ipynb](./tutorials/image_composition.ipynb) shows the different pre-processing steps to
compose an arbitrary image from a background image and some foreground images.

## Run the script
We're providing the script to generate images from scratch based on a set of background and foreground images.

##### * Generate image from scratch 
```
python3 ./src/coco_assistant/image_composition.py --input_dir=/path/to/dataset --output_dir=/path/to/new_dataset --count=number_of_images_to_generate --width=width --height=height
```

##### * Create coco annotations
```
python3 ./src/coco_assistant/coco_json_annotations.py --dataset_info=/path/to/new_dataset/dataset_info.json
                                                      --mask_definition=/path/to/new_dataset/mask_definitions.json 
```

##### * Split dataset into tran/val/test sets
```
python3 ./src/coco_assistant/split_dataset.py 
                            --image_dir=/path/to/new_dataset/images
                            --output_dir=/path/to/new_dataset2/
                            --json_file=/path/to/new_dataset/coco_annotations.json
                            --test_percentage=10 (default)
                            --val_percentage=10 (default)
```

#### * If merge or removal categories is needed for both options indicate directly into source code img_dir and ann_files list
```
python3 ./src/coco_assistant/coco_merge_remove.py merge --output_dir=/path/to/new_dataset2/

python3 ./src/coco_assistant/coco_merge_remove.py remove --output_dir=/path/to/new_dataset2/ --rcat=['cat1']
```

# 2. Training on MS COCO

The implementation of object detection and segmentation model is included in /src/mrcnn.
 You can use the pre-trained weights for MS COCO as a starting point to train your own variation on the network.
You can run the code directly from the command line as such:

#### Train a new model starting from pre-trained COCO weights
```
python3 samples/coco/coco.py train --dataset=/path/to/coco/ --model=coco
```

#### Continue training a model that you had trained earlier
```
python3 samples/coco/coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5
```
#### Continue training the last model you trained. This will find the last trained weights in the model directory.
```
python3 samples/coco/coco.py train --dataset=/path/to/coco/ --model=last
```

# 3. Result analysis

[image_test_results.ipynb](./tutorials/image_test_results.ipynb) proposes to visualize 
some results from the trained model on the test set.

## 3.1 Tensorboard
TensorBoard is another great debugging and visualization tool. The model is configured to log losses and save weights at the end of every epoch.
```
tensorboard --logdir path/to/weights
```

## 3.2 Metrics & AP computation
The implementation of object detection metrics is included in /src/metrics.
```
python3 compute_ap.py
```


# 4. Mix your dataset with existing Taco dataset
Taco stands for Trash Annotations in Context. It is an open image dataset of waste in the wild.
It contains photos of litter taken under diverse environments, from tropical beaches to London streets.
These images are manually labeled and segmented according to a hierarchical taxonomy to train and
evaluate object detection algorithms.
Taco dataset can be downloaded at http://tacodataset.org/

<p align="middle">
  <img src="http://tacodataset.org/img/mosaic/1.png" width="100" />
  <img src="http://tacodataset.org/img/mosaic/2.png" width="100" /> 
  <img src="http://tacodataset.org/img/mosaic/3.png" width="100" />
  <img src="http://tacodataset.org/img/mosaic/4.png" width="100" />
</p>
<p align="middle">
  <img src="http://tacodataset.org/img/mosaic/5.png" width="100" />
  <img src="http://tacodataset.org/img/mosaic/6.png" width="100" /> 
  <img src="http://tacodataset.org/img/mosaic/7.png" width="100" />
  <img src="http://tacodataset.org/img/mosaic/8.png" width="100" />
</p>
<p align="middle">
  <img src="http://tacodataset.org/img/mosaic/9.png" width="100" />
  <img src="http://tacodataset.org/img/mosaic/10.png" width="100" /> 
  <img src="http://tacodataset.org/img/mosaic/11.png" width="100" />
  <img src="http://tacodataset.org/img/mosaic/12.png" width="100" />
</p>

## Getting started
[dataset_structuration.ipynb](./tutorials/dataset_structuration.ipynb) shows the different steps to
mix a part of taco dataset with our synthetic dataset.

[inspect_data_statistics.ipynb](./tutorials/inspect_data_statistics.ipynb) is another notebook to extract quickly
some information from the dataset, like the number of annotations per image, or the different image sizes.