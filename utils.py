import torch
import torch.nn as nn
import sys, os

REA_PATH = '../results'

COLOR_DICT_3D = {
    0: [0, 0, 0],
    1: [0.2, 0.9, 0.2],
    2: [0.9, 0.1, 0.9],
    3: [0.3, 0.3, .9],
    4: [0.9, 0.2, 0.2],
    5: [.5, .55, .5],
    6: [.5, 0.1, .5],
    7: [1, .64, 0.3],
    8: [0.2, 0.9, 0.9],
    9: [1, 0.3, 1],
}

def get_root_dir():
    dirname = os.getcwd()
    dirname_split = dirname.split("/")
    index = dirname_split.index("zeroc")
    dirname = "/".join(dirname_split[:index + 1])
    return dirname


def load_dataset(filename, directory=None, isTorch=True):
    if directory is None:
        directory = 'datasets/ARC/data/training'
    with open(os.path.join(get_root_dir(), directory, filename)) as json_file:
        dataset = json.load(json_file)
    if isTorch:
        dataset = to_Variable_recur(dataset, type="long")
    return dataset


class ClevrImagePreprocessor(nn.Module):
    def __init__(self, resolution, crop = tuple(), rgb_mean = 0.5, rgb_std = 0.5):
        super().__init__()
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.resolution = resolution
        self.crop = crop
        
    def forward(self, img, normalize = True, interpolate_mode = 'bilinear', **kwargs):
        assert img.is_floating_point()
        img = (img - self.rgb_mean) / self.rgb_std if normalize else img

        img = img[..., self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]] if self.crop else img

        img = F.interpolate(img, self.resolution, mode = interpolate_mode)
        img = img.clamp(-1 if normalize else 0, 1)
        return img