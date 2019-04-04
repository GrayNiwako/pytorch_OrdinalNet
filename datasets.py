# -*- coding: utf-8 -*-
import os
import json
from PIL import Image
import torch as t
from torch.utils import data
from torchvision.transforms import functional as Tfn
from torch.utils.data import DataLoader
from core.config import Config
from createdatacatalog import create_catalog

class CloudDataset(data.Dataset):
    label2score = [0.0, 0.1, 0.25, 0.75, 1.0, 0.0]

    def __init__(self, catalog_json, img_resize=None):
        # type:(str,list,list)->CloudDataset
        assert os.path.isfile(catalog_json), "{} is not exist or not a file".format(catalog_json)
        self.catalog_json = catalog_json
        with open(catalog_json, 'r') as fp:
            self.scene_list = json.load(fp)

        h, w = img_resize[0:2]
        self.parent_init_size = [w * 2, h * 4]
        self.boxes = [[0, 0], [w, 0],
                      [0, h], [w, h],
                      [0, 2 * h], [w, 2 * h],
                      [0, 3 * h], [w, 3 * h]]
        self.out_size = [h, w]

    def __getitem__(self, scene_index):
        try:
            scene = self.scene_list[scene_index]
            # scene_id = scene["id"]
            sublist = scene["sublist"]
            sub_dict = {}
            for sub in sublist:
                sub_idx = sub["index"]
                sub_label = sub["label"]
                sub_path = sub["path"]
                sub_img = Image.open(sub_path)
                sub_img = Tfn.resize(sub_img, self.out_size)
                sub_dict[sub_idx] = [sub_img, sub_label]
        except KeyError as e:
            print("Invalid key '{}' for {}".format(e, self.catalog_json))
        sub_tensors = []
        label_s = []
        parent_img = Image.new('RGB', self.parent_init_size)
        label_p = 0
        label_s_count_F = 0
        # combine sub-images into parent-image
        for i, (img, label) in sub_dict.items():
            parent_img.paste(img, self.boxes[i])
            sub_tensors.append(Tfn.to_tensor(img))
            if label == 5:
                label_s_count_F += 1
            label_p += self.label2score[label] / 8
            label_s.append(label)
        tensor_s = t.cat(sub_tensors)
        label_s = t.tensor(label_s)
        tensor_p = Tfn.to_tensor(Tfn.resize(parent_img, self.out_size))
        label_p = t.tensor(count_label(label_p, label_s_count_F))
        return tensor_s, label_s, tensor_p, label_p

    def __len__(self):
        return len(self.scene_list)


def count_label(number, count_F):
    if count_F == 8:
        return 5
    elif number == 0.0:
        return 0
    elif number <= 0.1:
        return 1
    elif number <= 0.25:
        return 2
    elif number <= 0.75:
        return 3
    else:
        return 4

def CloudDataLoader(data_type, config):
    # type:(str,Config)->DataLoader
    datadir = os.path.join(config.data_save_path, data_type)
    if data_type == 'train':
        data_path = './logs/train_catalog.json'
        if os.path.exists(data_path):
            print("Find exist catalog '{}' for '{}'".format(data_path, datadir))
        else:
            create_catalog(datadir, config, data_path)
        shuffle = config.shuffle_train
        drop_last = config.drop_last_train
    elif data_type == 'validation':
        data_path = './logs/validation_catalog.json'
        if os.path.exists(data_path):
            print("Find exist catalog '{}' for '{}'".format(data_path, datadir))
        else:
            create_catalog(datadir, config, data_path)
        shuffle = config.shuffle_val
        drop_last = config.drop_last_val
    dataset = CloudDataset(data_path, config.image_resize)
    assert len(dataset) > config.batch_size
    return DataLoader(dataset, config.batch_size, shuffle, num_workers=config.num_data_workers,
                      pin_memory=config.pin_memory, drop_last=drop_last, timeout=config.time_out)
