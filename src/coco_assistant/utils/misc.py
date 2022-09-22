import shutil
from pathlib import Path
from typing import Union
import os

def make_clean(dpath):
    if dpath.exists():
        shutil.rmtree(dpath)
    dpath.mkdir(parents=True)

class DirectoryHandler:
    def __init__(
        self, ann_files: Union[str, Path], res_dir: Union[str, Path]) -> None:
        self.ann_files=ann_files
        # Create directory to store results
        self.res_dir = Path(res_dir)
        if not res_dir.exists():
            self.res_dir.mkdir()
        # else: warn user

    def create(self, folder: str) -> Path:
        dpath = self.res_dir / folder
        if dpath.exists():
            shutil.rmtree(dpath)
        dpath.mkdir(parents=True)
        return dpath

    # def check(self):
    #     dh_names = {}
    #     for key, list_value in self.dict_train.items():
    #         path_json = []
    #         for value in list_value:    # boucle sur train/val/test
    #             for i_path in self.ann_files:
    #                 if (value in i_path.stem):
    #                     path_json.append(i_path)
    #         dh_names[key]=path_json
    #     print("dh_names",dh_names)
    #     return dh_names