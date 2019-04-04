# -*- coding: utf-8 -*-
import os
import json
from core import Config
from collections import defaultdict

def create_catalog(datadir, config: Config, save_path):
    img_paths, parent_ids, labels = step1_getpath(datadir, config.classes_list)
    img_dict, intact_stat = step2_findparent(img_paths, parent_ids, labels)
    print("integrality statistics (total %d):" % len(img_dict.keys()))
    for k, v in intact_stat:
        print("%d sub-images:    %d scenes" % (k, v))

    scene_list = step3_dict2list(img_dict)
    with open(save_path, 'w+') as f:
        json.dump(scene_list, f)
    print("Catalog for {} saved into {}".format(datadir, save_path))


def step1_getpath(datadir, classes_list):
    img_paths, parent_ids, labels = [], [], []
    for i, annotation in enumerate(classes_list):
        dir = os.path.join(datadir, annotation)
        fs = os.listdir(dir)
        parent_id = [int(os.path.basename(f).split('_')[0]) for f in fs]
        fs = [os.path.abspath(os.path.join(dir, item)) for item in fs]
        img_paths.extend(fs)
        parent_ids.extend(parent_id)
        labels.extend([i] * len(fs))

    return img_paths, parent_ids, labels


def step2_findparent(img_paths, parent_ids, labels):
    img_dict = defaultdict(dict)
    for path, parent, label in zip(img_paths, parent_ids, labels):
        img_dict[parent][path] = label
    # statistic
    intact_stat = defaultdict(int)
    for i, (k, v) in enumerate(img_dict.items()):
        intact_stat[len(v)] += 1
    return img_dict, sorted(intact_stat.items(), key=lambda item: item[0], reverse=True)


def step3_dict2list(img_dict):
    scenelist = []
    scenelist.clear()
    for id, subimg in img_dict.items():
        sublist = []
        for subpath, sublabel in subimg.items():
            basename = os.path.basename(subpath)
            idx = int(basename[basename.index('_') + 1])
            sublist.append({"index": idx, "label": sublabel, "path": subpath})
        sublist.sort(key=lambda x: x["index"])
        scenelist.append({"id": id, "sublist": sublist})
    scenelist = sorted(scenelist, key=lambda x: int(x["id"]), reverse=False)
    return scenelist
