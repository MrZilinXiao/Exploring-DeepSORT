from itertools import groupby
import torch
import copy
import numpy as np
import random

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision import transforms, datasets
from PIL import Image
from abc import abstractmethod, ABC
from typing import Tuple, List
from collections import defaultdict
from tricks.random_erasing import RandomErasing


def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids


def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids


def build_transforms(exp_dict, train=True):
    sz: Tuple[int, int] = exp_dict['input_size']
    if not train:
        return transforms.Compose([
            transforms.Resize(sz),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    rhf_p = exp_dict['transforms']['random_horizontal_flip_p']
    pad = exp_dict['transforms']['padding']
    random_erasing = exp_dict['transforms']['random_erasing']
    random_erasing_p = exp_dict['transforms']['random_erasing_p']
    train_transform_list = [
        transforms.Resize(sz),
        transforms.RandomHorizontalFlip(p=rhf_p),
        transforms.Pad(pad),
        transforms.RandomCrop(sz),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # normalize
    ]
    if random_erasing:
        train_transform_list.append(RandomErasing(probability=random_erasing_p, mean=[0.485, 0.456, 0.406]))
    return transforms.Compose(train_transform_list)


class RandomIdentitySampler(Sampler):
    """
    Random sampling N identities out of training set  -> batch_size
    for each identity, sample K instances -> num_instances
    only used for training
    Example batch pids (batch_size = 64, num_instances = 4) would be [3,3,3,3,122,122,122,122,...]

    This sampler makes sure that identities in each batch remain balanced
    """

    def __init__(self, data_list: List[Tuple[str, int, int]], batch_size: int, num_instances: int):
        assert batch_size % num_instances == 0, "Batch_Size {} has to be a multiplier of num_instances {}!".format(batch_size, num_instances)
        super(RandomIdentitySampler).__init__()
        self.data_list = data_list
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances

        self.index_dict = defaultdict(list)
        for idx, (_, pid, _) in enumerate(self.data_list):
            self.index_dict[pid].append(idx)  # build mapping (pid->idx in datalist)
        self.pid_list = list(self.index_dict.keys())

        self.len = 0
        for pid in self.pid_list:
            id_list = self.index_dict[pid]
            num_pid = len(id_list)  # instances num of this identity
            if num_pid < self.num_instances:  # this person with less instances than num_instances
                num_pid = self.num_instances
            self.len += num_pid - num_pid % self.num_instances
            # drop some to make sure self.len is a multiplier of self.num_instances

    def __iter__(self):
        batch_idx_dict = defaultdict(list)  # {pid: [[1,2,3,4], [5,6,7,8]}

        for pid in self.pid_list:  # for each identity
            idxs = copy.deepcopy(self.index_dict[pid])
            if len(idxs) < self.num_instances:  # make sure mini-batch is full
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            np.random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idx_dict[pid].append(batch_idxs)  # num_instances full
                    batch_idxs = []
        avail_pids = copy.deepcopy(self.pid_list)
        final_idxs = []

        while len(avail_pids) >= self.num_pids_per_batch:  # reduce avail_pids
            selected_pids = random.sample(avail_pids, k=self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idx_dict[pid].pop(0)  #
                final_idxs.extend(batch_idxs)
                if len(batch_idx_dict[pid]) == 0:  # no more batch for this pid
                    avail_pids.remove(pid)

        self.len = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.len


class MOTSplitDataset:
    _dir = {
        'MOT16': './MOT16Cropped',
        'MOT17': './MOT17Cropped'
    }

    def __init__(self, exp_dict, data_type, split_ratio=0.2, random_seed=233):
        data = datasets.ImageFolder(self._dir[data_type])
        X, Y = zip(*data.imgs)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_ratio, random_state=random_seed,
                                                            stratify=Y)
        self.class_to_idx = data.class_to_idx
        self.train_set = MOTDataset(exp_dict, X_train, Y_train, train=True)
        self.val_set = MOTDataset(exp_dict, X_test, Y_test, train=False)
        self.reid_set = MOTReIDDataset(exp_dict, X_test, Y_test)


class MOTDataset(Dataset):
    def __init__(self, exp_dict, path_list, label_list, train: bool = True):
        self.exp_dict = exp_dict
        self.path_list = path_list
        self.label_list = label_list
        assert len(path_list) == len(label_list)
        self.transforms = build_transforms(exp_dict, train)

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, item):
        img = Image.open(self.path_list[item]).convert('RGB')
        img = self.transforms(img)
        # return img, self.label_list[item], 1, self.path_list[item]
        return img, self.label_list[item]
        # img, pid, camid, img_path, for MOT dataset camid set to 1


class MOTReIDDataset(Dataset):
    """
    ReID Sampling Strategy
    """

    def __init__(self, exp_dict, path_list: list, label_list: list, query_cnt=1):
        super(MOTReIDDataset, self).__init__()
        self.query_cnt = query_cnt
        assert len(path_list) == len(label_list), "Length of both lists should equal!"
        zipped = list(zip(path_list, label_list))
        self.zipped = {k: list(g) for k, g in groupby(sorted(zipped), lambda x: x[1])}
        # self.zipped: {0: [(), (), ...]}  ()->('./MOT16Cropped/MOT16-02_10_1/0.jpg', 0)
        self.query_list, self.gallery_list = [], []
        for v in self.zipped.values():
            self.query_list.extend(v[:query_cnt])
            self.gallery_list.extend(v[query_cnt + 1:])

        # self.query_list: [('./MOT16Cropped/MOT16-02_10_1/0.jpg', 0), ...]

        self.transforms = build_transforms(exp_dict, train=False)

    def __getitem__(self, item) -> (torch.Tensor, int, bool):
        """
        (data, identity, if is query)
        :param item:
        :return:
        """
        _tuple = self.query_list[item] if item < len(self.query_list) else self.gallery_list[
            item - len(self.query_list)]
        img = Image.open(_tuple[0]).convert('RGB')
        img = self.transforms(img)
        return img, _tuple[1], item < len(self.query_list)

    def __len__(self):
        return len(self.query_list + self.gallery_list)


class SplitDataset:
    _dir = {
        'MOT16': './MOT16Cropped',
        'MOT17': './MOT17Cropped'
    }

    def __init__(self, resize: Tuple[int, int] = (224, 224), data_type='MOT16', split_ratio=0.2, random_seed=233):
        data = datasets.ImageFolder(self._dir[data_type])
        X, Y = zip(*data.imgs)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_ratio, random_state=random_seed,
                                                            stratify=Y)
        self.class_to_idx = data.class_to_idx
        self.train_dataset = ExpDataset(resize, X_train, Y_train, train=True)
        self.eval_dataset = ExpDataset(resize, X_test, Y_test, train=False)
        self.reid_dataset = ReIDDataset(resize, X_test, Y_test)


class ReIDDataset(Dataset):
    def __init__(self, resize: Tuple[int, int], path_list: list, label_list: list, query_cnt=1):
        super(ReIDDataset, self).__init__()
        assert len(path_list) == len(label_list), "Length of both lists should equal!"
        zipped = list(zip(path_list, label_list))
        self.zipped = {k: list(g) for k, g in groupby(sorted(zipped), lambda x: x[1])}
        # self.zipped: {0: [(), (), ...]}  ()->('./MOT16Cropped/MOT16-02_10_1/0.jpg', 0)
        self.query_list, self.gallery_list = [], []
        for v in self.zipped.values():
            self.query_list.extend(v[:query_cnt])
            self.gallery_list.extend(v[query_cnt + 1:])

        # self.query_list: [('./MOT16Cropped/MOT16-02_10_1/0.jpg', 0), ...]

        self.transforms = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, item) -> (torch.Tensor, int, bool):
        """
        (data, identity, if is query)
        :param item:
        :return:
        """
        _tuple = self.query_list[item] if item < len(self.query_list) else self.gallery_list[
            item - len(self.query_list)]
        img = Image.open(_tuple[0]).convert('RGB')
        img = self.transforms(img)
        return img, _tuple[1], item < len(self.query_list)

    def __len__(self):
        return len(self.query_list + self.gallery_list)

    def reset_size(self, sz: Tuple[int, int]):
        """
        Reset size of transform
        :param sz:
        :return:
        """
        self.transforms = transforms.Compose([
            transforms.Resize(sz),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


class ExpDataset(Dataset):
    def __init__(self, resize: Tuple[int, int], path_list: list, label_list: list, train=True):
        super(ExpDataset, self).__init__()
        self.train = train
        self.reset_size(resize)
        self.path_list, self.label_list = path_list, label_list
        assert len(self.path_list) == len(self.label_list), 'Size dismatch between images and labels...'

    def __len__(self):
        return len(self.path_list)

    def reset_size(self, sz: Tuple[int, int]):
        if self.train:
            self.transforms = transforms.Compose([
                transforms.Resize(sz),
                transforms.RandomCrop(sz, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize(sz),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, item):
        img = Image.open(self.path_list[item]).convert('RGB')
        img = self.transforms(img)
        return img, self.label_list[item]


if __name__ == '__main__':
    split_dataset = SplitDataset()


class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()
        self.transforms = None

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass


class CropTrainDataset(BaseDataset, ABC):
    def __init__(self, path_list, label_list):
        super(CropTrainDataset, self).__init__()
        self.transforms = transforms.Compose([
            transforms.Resize((128, 64)),
            transforms.RandomCrop((128, 64), padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.path_list = path_list
        self.label_list = label_list
        assert len(path_list) == len(label_list), "Length of both lists should equal!"

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, item):
        img = Image.open(self.path_list[item]).convert("RGB")
        # img.show()
        img = self.transforms(img)
        return img, self.label_list[item]


class CropTestDataset(CropTrainDataset):
    def __init__(self, path_list, label_list):
        super(CropTestDataset, self).__init__(path_list, label_list)
        self.transforms = transforms.Compose([
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


if __name__ == '__main__':
    # test build train transform
    import yaml
    exp_dict = yaml.load(open('experiments/resnet50_softmax_last_stride.yml'), Loader=yaml.Loader)
    trans = build_transforms(exp_dict, True)
    img = Image.open('/home/zilin/bettersort/MOT16Cropped/MOT16-13_9_1/0.jpg').convert('RGB')
    t = trans(img)
    pass