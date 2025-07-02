"""
dataset
"""
from pathlib import Path
import json
import numpy as np
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __getitem__(self, idx):
        sample_data = self.data[idx]
        sample_target = self.targets[idx]
        return sample_data, sample_target
    
    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"DNADataset(len={len(self)}, feature_shape={np.shape(self.data[0])})"



def read_data(data_path):
    with np.load(data_path, allow_pickle=True) as data:
        data_array = data['nested_structure']
    array_list = [seq for seq in data_array[0]]
    return array_list


def read_target(label_file):
    try:
        with open(label_file, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {label_file}: {e}")


def normalize_anno(anno_list):
    for idx, dict in enumerate(anno_list):
        dict['boxes'] = [[element / dict['seq_len'] for element in sub_list] for sub_list in dict['boxes']]
        anno_list[idx] = dict
    return anno_list


def build(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f"Provided data path {root} does not exist"

    PATHS = {
        "train": (root / "train_features.npz", root / "train_label.json"),
        "val": (root / "test_features.npz", root / "test_label.json"),
    }

    data_path, target_path = PATHS[image_set]
    features = read_data(data_path)
    labels = normalize_anno(read_target(target_path))
    return Dataset(features, labels)
