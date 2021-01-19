import argparse
import configparser
from typing import Any, Callable, List
import pandas as pd
import os
import glob
from tqdm import tqdm
import numpy as np
from PIL import Image


class MOTCropper(object):
    data_split = {'MOT16': {
        "train": ("train", (
            "MOT16-02", "MOT16-04", "MOT16-05", "MOT16-11", "MOT16-13",
        )),
        "val": ("train", (
            "MOT16-09", "MOT16-10",
        )),
        "trainval": ("train", (
            "MOT16-02", "MOT16-04", "MOT16-05", "MOT16-09", "MOT16-10",
            "MOT16-11", "MOT16-13",
        )),
        "test": ("test", (
            "MOT16-01", "MOT16-03", "MOT16-06", "MOT16-07", "MOT16-08",
            "MOT16-12", "MOT16-14",
        )),
    },
        'MOT17': {
            "train": ("train", (
                "MOT17-02-DPM", "MOT17-04-DPM", "MOT17-05-DPM", "MOT17-11-DPM", "MOT17-13-DPM",
            )),
            "val": ("train", (
                "MOT17-09-DPM", "MOT17-10-DPM",
            )),
            "trainval": ("train", (
                "MOT17-02-DPM", "MOT17-04-DPM", "MOT17-05-DPM", "MOT17-09-DPM", "MOT17-10-DPM",
                "MOT17-11-DPM", "MOT17-13-DPM",
            )),
            "test": ("test", (
                "MOT17-01-DPM", "MOT17-03-DPM", "MOT17-06-DPM", "MOT17-07-DPM", "MOT17-08-DPM",
                "MOT17-12-DPM", "MOT17-14-DPM",
            ))
        }
    }

    def __init__(self, options):
        super(MOTCropper, self).__init__()
        assert options.dataset in self.data_split.keys(), "%s dataset not supported yet!" % options.dataset
        self.data_dir: str = options.input_dir
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError('data dir {} not found'.format(self.data_dir))
        self.output_dir: str = options.output_dir
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
        self.crop_target: str = options.crop_target
        if self.crop_target not in self.data_split[options.dataset].keys():
            raise ValueError("crop_target not identified")
        target_prefix, target_dir = self.data_split[options.dataset][self.crop_target]
        join_func: Callable[[str], str] = lambda _dir: os.path.join(self.data_dir, target_prefix, _dir)
        target_paths = tuple(map(join_func, target_dir))  # like ('MOT16/train/MOT16-02', ...)

        self.seqinfo = pd.concat([self.get_seqinfo(path) for path in target_paths])
        self.gtinfo = {
            os.path.basename(path): self.get_gtinfo(path) for path in target_paths
            # 'MOT16-02': gtinfo
        }
        self.class_mapping = {
            os.path.basename(path): {} for path in target_paths
        }
        if options.det:
            self.detinfo = {
                os.path.basename(path): self.get_detinfo(path) for path in target_paths
            }
        self.imgs = sorted([name for path in target_paths for name in glob.glob(os.path.join(path, 'img1', '*'))])
        # load all img files

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item) -> (Image.Image, list, pd.DataFrame, str):
        img_file = self.imgs[item]
        _id = int(os.path.splitext(img_file.split(os.sep)[-1])[0])  # 000001
        _dir = img_file.split(os.sep)[-3]  # MOT16-02

        gtinfo = self.gtinfo[_dir].query(
            f"frame == {_id} and score != 0 and visibility >= 0.5")  # those records with _id frame
        class_label = (np.asarray(gtinfo["identity"])).astype(np.int32)
        class_label_2 = (np.asarray(gtinfo["class"])).astype(np.int32)
        # label_min = np.min(class_label)
        # norm_class_label = class_label - label_min
        left = np.asarray(gtinfo["left"])
        top = np.asarray(gtinfo["top"])
        width = np.asarray(gtinfo["width"])
        height = np.asarray(gtinfo["height"])

        left[left < 0] = 0
        top[top < 0] = 0
        width[width < 0] = 0
        height[height < 0] = 0

        # regularize annotations

        bbox = np.stack((left, top, width, height), axis=-1).astype(np.float32)
        img: Image.Image = Image.open(img_file).convert('RGB')
        roi_list = []
        for l, t, w, h in bbox:
            roi = img.crop((l, t, l + w, t + h))
            roi_list.append(roi)

        return img, roi_list, class_label, _dir, class_label_2

    @staticmethod
    def get_seqinfo(path) -> pd.DataFrame:
        config = configparser.ConfigParser()
        config.optionxform = str
        assert os.path.exists(os.path.join(path, "seqinfo.ini"))
        config.read(os.path.join(path, "seqinfo.ini"))
        return pd.DataFrame(dict(config["Sequence"]), index=[0])

    @staticmethod
    def get_gtinfo(path) -> pd.DataFrame:
        gt = pd.read_csv(os.path.join(path, 'gt', 'gt.txt'),
                         names=("frame", "identity",
                                "left", "top", "width", "height",
                                "score", "class", "visibility"))
        return gt

    @staticmethod
    def get_detinfo(path) -> pd.DataFrame:
        det = pd.read_csv(os.path.join(path, 'det', 'det.txt'), usecols=range(7),  # only fetch 0~6 cols
                          names=("frame", "identity",
                                 "left", "top", "width", "height",
                                 "score"))
        return det


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MOT16', help='MOT dataset type')
    parser.add_argument('--input_dir', type=str, default='MOT16-new/', help='MOT dataset path')
    parser.add_argument('--output_dir', type=str, default='MOT16Cropped/', help='MOT cropped output path')
    parser.add_argument('--crop_target', type=str, default='train',
                        help='Crop target, should be in ("train", "val", "trainval", "test")')
    parser.add_argument('--frm_thld', type=int, default=10,
                        help='Filter those classes with less than `frm_thld` pictures')
    parser.add_argument('--det', type=bool, default=False,
                        help='whether need det info')  # det info only needed when performing tracking
    options = parser.parse_args()
    cropper = MOTCropper(options)

    class_mapping = {}
    bbox_cropped_list: List[Image.Image]
    for (img, bbox_cropped_list, labels, dir_name, labels_2) in tqdm(cropper):
        for idx in range(len(bbox_cropped_list)):  # for each frame
            subdir: str = dir_name + '_' + str(labels[idx]) + '_' + str(
                labels_2[idx])  # 'MOT16-02_44' -> class_id 44 in MOT16-02 sequence
            if subdir not in class_mapping:
                class_mapping[subdir] = 0
            save_path = os.path.join(options.output_dir, subdir)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            bbox_cropped_list[idx].save(os.path.join(save_path, str(class_mapping[subdir]) + '.jpg'))
            class_mapping[subdir] += 1

    # filter those classes with less than `frm_thld` frames
    for root, name, files in os.walk(options.output_dir):
        if not name:
            if len(files) < options.frm_thld:
                for file in files:
                    del_path = os.path.join(root, file)
                    os.remove(del_path)
                os.rmdir(root)
