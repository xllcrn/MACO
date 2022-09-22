# -*- coding: utf-8 -*-
import json
import logging
from pathlib import Path
import shutil
import sys
from pycocotools.coco import COCO
from tqdm import tqdm
from utils import remapper, misc
import random
import math
import pandas as pd

logging.basicConfig(level=logging.ERROR)
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("parso.python.diff").disabled = True


class COCO_Assistant:
    """COCO_Assistant object"""

    def __init__(self, img_dir=None, ann_files=None, output_dir=None):
        """

        Args:
            img_dir (str): Path to images folder.
            ann_dir (str): Path to annotations folder.

        """
        self.img_dir = img_dir
        self.ann_files = ann_files
        self.res_dir = Path(output_dir)
        self.ann_anchors = []

        # Create directory to store results (will contain : merged, removal, ...)
        if not self.res_dir.exists():
            self.res_dir.mkdir()
        else:
            print("The output directory {} already exists...".format(self.res_dir))

        self.dh = misc.DirectoryHandler(self.ann_files, self.res_dir)
        self.annfiles = [COCO(i) for i in self.ann_files]
        self.anndict = dict(zip(self.ann_files, self.annfiles))
        self.imgdict = dict(zip(self.ann_files, self.img_dir))

    def merge(self):
        """
        Merge multiple coco datasets
        """

        resimg_dir = self.dh.create("merged/images")
        resann_dir = self.dh.create("merged/annotations")

        # on initialise le fichier merge
        cann = {"images": [], "annotations": [], "info": None, "licenses": None, "categories": None}
        ann_name = 'coco_merged.json'
        dst_ann = resann_dir / ann_name
        print("Merging Annotations in {}...".format(dst_ann))

        # Boucle sur tous les fichiers .json
        for json_path in tqdm(self.ann_files):
            cj = self.anndict[json_path].dataset
            img_dir = Path(self.imgdict[json_path])

            ind = self.ann_files.index(json_path)
            # Check if this is the 1st annotation.
            # If it is, continue else modify current annotation
            if ind == 0:
                cann["images"] = cann["images"] + cj["images"]
                cann["annotations"] = cann["annotations"] + cj["annotations"]
                if "info" in list(cj.keys()):
                    cann["info"] = cj["info"]
                if "licenses" in list(cj.keys()):
                    cann["licenses"] = cj["licenses"]
                cann["categories"] = sorted(cj["categories"], key=lambda i: i["id"])

                last_imid = cann["images"][-1]["id"]
                last_annid = cann["annotations"][-1]["id"]

                # If last imid or last_annid is a str, convert it to int
                if isinstance(last_imid, str) or isinstance(last_annid, str):
                    logging.debug("String Ids detected. Converting to int")
                    id_dict = {}
                    # Change image id in images field
                    for i, im in enumerate(cann["images"]):
                        original_file = im["file_name"]
                        id_dict[im["id"]] = i
                        im["id"] = i
                        ext = original_file.split(".")[-1]
                        original_path= img_dir / original_file
                        new_file = f"{im['id']:08d}"+'.'+ ext
                        im["file_name"] = new_file
                        target_path = resimg_dir / new_file
                        shutil.copyfile(original_path, target_path)

                    # Change annotation id & image id in annotations field
                    for i, im in enumerate(cann["annotations"]):
                        im["id"] = i
                        if isinstance(last_imid, str):
                            im["image_id"] = id_dict[im["image_id"]]
                else:
                    for i, im in enumerate(cann["images"]):
                        original_file = im["file_name"]
                        original_path = img_dir / original_file
                        target_path = resimg_dir / original_file
                        shutil.copyfile(original_path, target_path)

                last_imid = max(im["id"] for im in cann["images"])
                last_annid = max(ann["id"] for ann in cann["annotations"])

            else:

                id_dict = {}
                # Change image id in images field
                for i, im in enumerate(cj["images"]):
                    id_dict[im["id"]] = last_imid + i + 1
                    im["id"] = last_imid + i + 1
                    original_file = im["file_name"]
                    ext = original_file.split(".")[-1]
                    original_path= img_dir / original_file
                    new_file = f"{im['id']:08d}"+'.'+ext
                    im["file_name"] = new_file
                    target_path = resimg_dir / new_file
                    shutil.copyfile(original_path, target_path)

                # Change annotation and image ids in annotations field
                for i, ann in enumerate(cj["annotations"]):
                    ann["id"] = last_annid + i + 1
                    ann["image_id"] = id_dict[ann["image_id"]]

                # Remap categories
                cmapper = remapper.CatRemapper(cann["categories"], cj["categories"])
                cann["categories"], cj["annotations"] = cmapper.remap(cj["annotations"])

                cann["images"] = cann["images"] + cj["images"]
                cann["annotations"] = cann["annotations"] + cj["annotations"]
                if "info" in list(cj.keys()):
                    cann["info"] = cj["info"]
                if "licenses" in list(cj.keys()):
                    cann["licenses"] = cj["licenses"]

                last_imid = cann["images"][-1]["id"]
                last_annid = cann["annotations"][-1]["id"]

        with open(dst_ann, "w") as aw:
            json.dump(cann, aw)

    def remove_cat(self, rcats=None):

        """
        Remove categories.

        In interactive mode, you can input the json and the categories to be
        removed (as a list, see Usage for example)

        In non-interactive mode, you manually pass in json filename and
        categories to be removed. Note that jc and rcats cannot be None if run
        with interactive=False.

        Raises:
            AssertionError: if specified index exceeds number of datasets
            AssertionError: if rcats is not a list of strings
            AssertionError: if jc = rcats = None
        """

        resrm_dir = self.dh.create("removal")

        # if jc is None or rcats is None:
        if rcats is None:
            raise AssertionError(
                "Categories need to be provided in non-interactive mode"
            )

        for jc, ann in self.anndict.items():
            # If passed, json_choice needs to be full path
            json_choice = Path(jc)  # Full path
            json_name = json_choice.name  # abc

            print("Removing specified categories...")

            # Gives you a list of category ids of the categories to be removed
            catids_remove = ann.getCatIds(catNms=rcats)
            if (catids_remove==[]):
                raise AssertionError(
                    "Categories to remove not found in json file {}".format(jc)
                )
            # Gives you a list of ids of annotations that contain those categories
            annids_remove = ann.getAnnIds(catIds=catids_remove)

            # Get keep category ids
            catids_keep = list(set(ann.getCatIds()) - set(catids_remove))
            # Get keep annotation ids
            annids_keep = list(set(ann.getAnnIds()) - set(annids_remove))

            with open(json_choice) as it:
                x = json.load(it)

            del x["annotations"]
            x["annotations"] = ann.loadAnns(annids_keep)
            del x["categories"]
            x["categories"] = ann.loadCats(catids_keep)

            print("Writing in new json file {}...".format(resrm_dir / json_name))
            with open(resrm_dir / json_name, "w") as oa:
                json.dump(x, oa)

    def keep_cat(self, kcats=None):

        """
        Keep a list of categories.
        In non-interactive mode, you manually pass in json filename and
        categories to be removed. Note that jc and kcats cannot be None if run
        with interactive=False.

        Raises:
            AssertionError: if specified index exceeds number of datasets
            AssertionError: if kcats is not a list of strings
            AssertionError: if jc = kcats = None
        """

        resrm_dir = self.dh.create("kept")

        # if jc is None or kcats is None:
        if kcats is None:
            raise AssertionError(
                "Categories need to be provided in non-interactive mode"
            )

        for jc, ann in self.anndict.items():
            # If passed, json_choice needs to be full path
            json_choice = Path(jc)  # Full path
            json_name = json_choice.name  # abc

            print("Keeping specified categories...")

            # Gives you a list of category ids of the categories to be kept
            catids_keep = ann.getCatIds(catNms=kcats)
            if (catids_keep==[]):
                raise AssertionError(
                    "Categories to keep not found in json file {}".format(jc)
                )
            # Gives you a list of ids of annotations that contain those categories
            annids_keep = ann.getAnnIds(catIds=catids_keep)

            # Get keep category ids
            catids_keep = list(set(catids_keep))
            # Get keep annotation ids
            annids_keep = list(set(annids_keep))

            with open(json_choice) as it:
                x = json.load(it)

            del x["annotations"]
            x["annotations"] = ann.loadAnns(annids_keep)
            del x["categories"]
            x["categories"] = ann.loadCats(catids_keep)

            # instance annotations
            dict_ids = {}
            for new_id, ids in enumerate(catids_keep):
                dict_ids[ids] = new_id + 1
            d_ann=[]
            for ann in x['annotations']:
                try:
                    ann["category_id"] = dict_ids[ann["category_id"]]
                    d_ann.append(ann)
                except KeyError:
                    continue
            x['annotations'] = d_ann
            # instance categories
            for cate in x['categories']:
                cate["id"] = dict_ids[cate["id"]]

            print("Writing in new json file {}...".format(resrm_dir / json_name))
            with open(resrm_dir / json_name, "w") as oa:
                json.dump(x, oa)

    def sampling_cat(self, sample_value):

        """
        Keep a list of categories.
        In non-interactive mode, you manually pass in json filename and
        categories to be removed. Note that jc and kcats cannot be None if run
        with interactive=False.

        Raises:
            AssertionError: if specified index exceeds number of datasets
            AssertionError: if kcats is not a list of strings
            AssertionError: if jc = kcats = None
        """

        resrm_dir = self.dh.create("sampling")

        # if jc is None or kcats is None:
        if sample_value ==0:
            raise AssertionError(
                "Sample number cannot be zero."
            )

        for jc, ann in self.anndict.items():
            # If passed, json_choice needs to be full path
            json_choice = Path(jc)  # Full path
            json_name = json_choice.name  # abc

            print("Sampling all categories...")

            with open(json_choice) as it:
                x = json.load(it)

            # get annotation per image
            img_card = []
            nb_cate = len(ann.getCatIds())
            name_cate = ['category_'+str(category["id"]) for category in x["categories"]]
            # print("name_cate",name_cate)

            for image in x["images"]:
                annid_per_img = ann.getAnnIds(imgIds=image["id"])
                dict_cate={name_cat:0 for name_cat in name_cate}
                dict_cate['img_id']=image["id"]
                for annid in annid_per_img :
                    # load annotation
                    annotation = ann.loadAnns(annid)
                    # retrieve category of annotation
                    cate_id = annotation[0]['category_id']
                    dict_cate['category_' + str(cate_id)] += 1

                sum_cate = 0
                for cate in name_cate:
                    sum_cate += dict_cate[cate]
                if (sum_cate!=0):
                    img_card.append(dict_cate)
            # print(img_card)
            df = pd.DataFrame.from_records(img_card)

            var={}
            for cate in name_cate :
                df.sort_values(by=[cate], ascending=False, inplace=True)
                df_cumsum = df.cumsum()
                # print(df_cumsum)
                category_max = df.sum().drop(['img_id']).idxmax()
                sample_df = df[df_cumsum[category_max]<=sample_value]
                # print(sample_df.sum())
                sample_sum=sample_df.sum()
                frame = {'Category Count': sample_df.sum(), 'Variance': (sample_sum-sample_value)**2}
                result = pd.DataFrame(frame)
                # print("result",result)
                # print(result['Variance'].drop(['img_id']).sum())
                var[cate] = result['Variance'].drop(['img_id']).sum()
            min_cate = min(var, key=var.get)
            # print("min_cate",min_cate)

            df.sort_values(by=[min_cate], ascending=False, inplace=True)
            category_max = df.sum().drop(['img_id']).idxmax()
            # get cumsum
            df_cumsum = df.cumsum().iloc[-1]
            # print(df_cumsum)

            sample_df = df[df.cumsum()[category_max] <= sample_value]
            # print("sample_df:")
            # print(sample_df.sum())
            back_df = df.drop(sample_df.index)
            # print(back_df)

            # get categories with value reached
            def category_reached(sample_df, value):
                df_reached = sample_df.sum().drop(['img_id'])[sample_df.sum() >= value]
                cate_reached = df_reached.index.tolist()
                # print("cate_reached",cate_reached)
                return cate_reached
            cate_reached = category_reached(sample_df, sample_value)



            if (len(cate_reached)<len(name_cate) and not back_df.empty):
                back_drop_df = back_df.copy()
                cate_non_reached = name_cate
                # on calcule les cards where le nb ann for cate reached=0
                # on calcule la diff entre les cate non reached et la value
                for cate in cate_reached:
                    back_drop_df = back_drop_df[back_drop_df[cate] == 0]
                    cate_non_reached = cate_non_reached.remove(cate)
                # print(back_drop_df)
                # print(cate_non_reached)
                counter = {cate : sample_value-sample_df.sum()[cate] for cate in name_cate}
                counter['img_id'] = 0
                max_cate = max(counter, key=counter.get)
                # print("le max", max_cate)
                df_counter = pd.DataFrame.from_dict([counter])
                # print(df_counter)

                back_drop_df.sort_values(by=[max_cate], ascending=False, inplace=True)

                # select lines of back_drop_df en reactualisant le counter
                counter_tmp = df_counter.copy()
                for jj in range(back_drop_df.shape[0]):
                    # print(back_drop_df.iloc[jj].drop(['img_id']))
                    counter_tmp_tmp = counter_tmp - back_drop_df.iloc[jj]
                    if (counter_tmp_tmp.iloc[0].drop(['img_id']).min()>=0):
                        counter_tmp = counter_tmp_tmp.copy()
                        sample_df = pd.concat([sample_df,back_drop_df.iloc[[jj]]])
                    # print(back_drop_df.iloc[[jj]])
                # print(sample_df.sum())
                cate_reached = category_reached(sample_df, sample_value)

                newback_df = df.copy()
                for cate in cate_reached:
                    newback_df = newback_df.drop((newback_df[newback_df[cate] != 0]).index)
                # print(newback_df.tail())

                # select lines of newback_df en reactualisant le counter
                counter = {cate : sample_value-sample_df.sum()[cate] for cate in name_cate}
                counter['img_id'] = 0
                max_cate = max(counter, key=counter.get)
                df_counter = pd.DataFrame.from_dict([counter])
                # print(df_counter)
                # print(sample_df.sum())
                counter_tmp = df_counter.copy()
                while(len(cate_reached)!=len(name_cate)):
                    for jj in range(newback_df.shape[0]):
                        counter_tmp_tmp = counter_tmp - newback_df.iloc[jj]
                        if (counter_tmp_tmp.iloc[0].drop(['img_id']).min() >= 0):
                            counter_tmp = counter_tmp_tmp.copy()
                            sample_df = pd.concat([sample_df, newback_df.iloc[[jj]]])
                    # print(sample_df.sum())
                    cate_reached = category_reached(sample_df, sample_value)

            # sample_df contains all images to keep : create json
            # retrieve image ids to keep annotations
            img_ids = sample_df["img_id"].to_list()
            annids_keep = ann.getAnnIds(imgIds=img_ids)
            catids_keep = ann.getCatIds()

            # write annotations
            del x["annotations"]
            x["annotations"] = ann.loadAnns(annids_keep)
            del x["categories"]
            x["categories"] = ann.loadCats(catids_keep)

            print("Writing in new json file {}...".format(resrm_dir / json_name))
            with open(resrm_dir / json_name, "w") as oa:
                json.dump(x, oa)

if __name__ == "__main__":
    import argparse
    #
    parser = argparse.ArgumentParser(description="Coco merge")
    parser.add_argument("method", type=str, help="merge, remove, keep, sampling")
    parser.add_argument("--output_dir", type=str, dest="output_dir", required=True, help="Output directory")
    # parser.add_argument("--ann_dir", type=str, default=None, dest="ann_dir", required=False,help="Annotation directory (can be used for removal)")
    # parser.add_argument("--json_file", type=str,default=None, dest="json_file", required=False, help="Specific json file name to remove category. If not, all ann files in input_dir will be concerned")
    # parser.add_argument('--rcat', '--list', type=str, required='remove' in sys.argv, help='Categories to remove')  # only required if remove option is given
    parser.add_argument('--sampling_value', required=True, type=int, help='sampling value')
    args = parser.parse_args()
    print(sys.argv)

    if (args.method=="merge"):
        # img_dir=[r'F:\Deep_Learning\Mask-RCNN-inspired-by-taco\datasets\trash_generated\images',
        #          r'F:\Deep_Learning\Mask-RCNN-inspired-by-taco\datasets\cig_butts_split\images\train']
        # ann_files=[r'F:\Deep_Learning\Mask-RCNN-inspired-by-taco\datasets\trash_generated\coco_annotations.json',
        #            r'F:\Deep_Learning\Mask-RCNN-inspired-by-taco\datasets\cig_butts_split\annotations\coco_annotations_train.json']

        # img_dir=[r'F:\Deep_Learning\Mask-RCNN-inspired-by-taco\datasets\taco_split\images\train',
        #          r'F:\Deep_Learning\Mask-RCNN-inspired-by-taco\datasets\trash_split\images\train']
        # ann_files=[r'F:\Deep_Learning\Mask-RCNN-inspired-by-taco\datasets\taco_split\annotations\coco_annotations_train.json',
        #            r'F:\Deep_Learning\Mask-RCNN-inspired-by-taco\datasets\trash_split\annotations\coco_annotations_train.json']

        img_dir=[r'F:\Deep_Learning\Mask-RCNN-inspired-by-taco\datasets\taco_light\images',
                 r'F:\Deep_Learning\Mask-RCNN-inspired-by-taco\datasets\trash_generated4\images']
        ann_files=[r'F:\Deep_Learning\Mask-RCNN-inspired-by-taco\datasets\taco_light\kept\coco_annotations_taco.json',
                   r'F:\Deep_Learning\Mask-RCNN-inspired-by-taco\datasets\trash_generated4\coco_annotations.json']

        cas = COCO_Assistant(img_dir=img_dir, ann_files=ann_files, output_dir=args.output_dir)
        cas.merge()
    elif (args.method == "remove"):
        img_dir=[r'F:\Deep_Learning\Mask-RCNN-inspired-by-taco\datasets\taco_light\images']
        ann_files=[r'F:\Deep_Learning\Mask-RCNN-inspired-by-taco\datasets\taco_light\annotations\coco_annotations_taco.json']
        cas = COCO_Assistant(img_dir=img_dir, ann_files=ann_files, output_dir=args.output_dir)
        rcat=["Battery","Aluminium blister pack"]
        cas.remove_cat(rcat)
    elif (args.method == "keep"):
        img_dir=[r'F:\Deep_Learning\Mask-RCNN-inspired-by-taco\datasets\taco_light\images']
        ann_files=[r'F:\Deep_Learning\Mask-RCNN-inspired-by-taco\datasets\taco_light\annotations\coco_annotations_taco.json']
        cas = COCO_Assistant(img_dir=img_dir, ann_files=ann_files, output_dir=args.output_dir)
        kcat=["Plastic film","Clear plastic bottle","Drink can","Cigarette","Plastic bottle cap","Broken glass"]
        cas.keep_cat(kcat)
    elif (args.method == "sampling"):
        img_dir=[r'F:\Deep_Learning\Mask-RCNN-inspired-by-taco\datasets\taco_final\merged\images']
        ann_files=[r'F:\Deep_Learning\Mask-RCNN-inspired-by-taco\datasets\taco_final\merged\annotations\coco_merged.json']
        cas = COCO_Assistant(img_dir=img_dir, ann_files=ann_files, output_dir=args.output_dir)
        cas.sampling_cat(args.sampling_value)