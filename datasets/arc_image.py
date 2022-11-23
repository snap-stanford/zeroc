from collections import Iterable
from copy import deepcopy
import itertools
import json
import matplotlib.pylab as plt
import numpy as np
import pdb
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
from torchsummary import summary

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..', '..'))
from zeroc.concept_library.util import visualize_matrices, to_one_hot, to_np_array, record_data, to_Variable, to_Variable_recur
from zeroc.utils import get_root_dir


# In[ ]:


def load_task_as_images(filename, directory=None, canvas_size=None, isTorch=True):
    """ Load a task and return a list of images of the inputs and targets. 
    This is a helper function for training encoder."""
    task = load_dataset(filename, directory, isTorch)
    task_images = []
    for key in task.keys():
        for pair in task[key]:
            if canvas_size is None:
                task_images.append(pair['input'])
                task_images.append(pair['output'])
            else:
                if pair['input'].shape[0] <= canvas_size and pair['input'].shape[1] <= canvas_size:
                    task_images.append(pair['input'])
                if pair['output'].shape[0] <= canvas_size and pair['output'].shape[1] <= canvas_size:
                    task_images.append(pair['output'])
    return task_images



def load_dataset_as_images(directory=None, canvas_size=None, isTorch=True):
    """ Load all inputs and targets as tensors into a single list. It takes in the directory of json files and return a list of tensors.
    This is a helper function for training encoder"""
    if directory is None:
        directory = 'datasets/ARC/data/training'
    dataset_images = []
    for file_name in [file for file in os.listdir(os.path.join(get_root_dir(), directory)) if file.endswith('.json')]:
        dataset_images = dataset_images + load_task_as_images(file_name, directory, canvas_size=canvas_size, isTorch=isTorch)
    return dataset_images

def get_max_shape(dataset):
    """ Get the max height and width of all images in the dataset (a list of tensors).
    The return of ARC Training is 30x30"""
    max_x = -1
    max_y = -1
    for image in dataset:
        if image.size()[0] > max_x:
            max_x = image.size()[0]
        if image.size()[1] > max_y:
            max_y = image.size()[1]
    return (max_x, max_y)

def centralize_image(image, shape):
    """ Centralize an image with the given shape. Fill the blank with 0."""
    new_image = np.zeros(shape) # Fill all pixels with 0
    # centralize
    x, y = image.size()
    x_dis = (int) ((shape[0] - x) / 2)
    y_dis  = (int) ((shape[1] - y) / 2)   
    new_image[x_dis: x_dis + x, y_dis: y_dis + y] = image
    return new_image
        
def preprocess_data(dataset, canvas_size=32):
    """ Centralize all images in a dataset. Return a list of preprocessed images."""
    processed_data = []
    # We assume that we always want height = width
    shape = (canvas_size, canvas_size)
    for image in dataset:
        new_image = centralize_image(image, shape=shape)
        processed_data.append(to_one_hot(new_image, n_channels=10))
    return np.stack(processed_data)


# In[ ]:


class ARCDataset(Dataset):
    """ARC Dataset"""
    def __init__(
        self,
        canvas_size=32,
        n_examples=None,
        directory_name=None,
        output_mode="None",
        idx_list=None,
        data=None,
        transform=None,
    ):
        """
        Args:
            directory_name (string): Path to the directory of dataset.
        """
        self.canvas_size = canvas_size
        self.n_examples = n_examples
        self.directory_name = directory_name
        self.output_mode = output_mode
        if idx_list is None:
            assert data is None
            self.data = torch.FloatTensor(preprocess_data(load_dataset_as_images(directory_name, canvas_size=canvas_size), canvas_size=canvas_size))
            if self.n_examples is not None:
                # Randomly sample data augmentations (rotate{A,B,C}, {h,v}Flip, DiagFlip{A,B}) to fill to n_examples:
                if len(self.data) < self.n_examples:
                    data_aug = torch.cat([
                        torch.rot90(self.data, k=1, dims=(-2,-1)),  # RotateA
                        torch.rot90(self.data, k=2, dims=(-2,-1)),  # RotateB
                        torch.rot90(self.data, k=3, dims=(-2,-1)),  # RotateC
                        self.data.flip(-1),  # hFlip
                        self.data.flip(-2),  # vFlip
                        torch.rot90(self.data, k=1, dims=(-2,-1)).flip(-1),  # DiagFlipA=hFlip(RotateA(obj))
                        torch.rot90(self.data, k=1, dims=(-2,-1)).flip(-2),  # DiagFlipB=vFlip(RotateA(obj))  
                    ])
                    if data_aug.shape[0] < self.n_examples - len(self.data):
                        data = torch.cat([self.data, data_aug])
                    else:
                        chosen_idx = np.random.choice(data_aug.shape[0], size=self.n_examples - len(self.data), replace=False)
                        data = torch.cat([self.data, data_aug[chosen_idx]])
                    self.data = data[torch.randperm(data.shape[0])]
                else:
                    self.data = self.data[torch.randperm(self.n_examples)]
            self.idx_list = list(range(len(self.data))) if idx_list is None else idx_list
        else:
            self.idx_list = idx_list
            self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        elif isinstance(idx, slice):
            return self.__class__(
                canvas_size=self.canvas_size,
                n_examples=self.n_examples,
                directory_name=self.directory_name,
                output_mode=self.output_mode,
                idx_list=self.idx_list[idx],
                data=self.data,
                transform=self.transform,               
            ) 

        sample = self.data[self.idx_list[idx]]
        if self.transform:
            sample = self.transform(sample)

        if self.output_mode == "energy":
            return (sample, 0)
        elif self.output_mode == "counts":
            return (sample, -1)
        elif self.output_mode == "None":
            return sample
        else:
            raise

    def draw(self, idx):
        if not isinstance(idx, Iterable):
            idx = [idx]
        for index in idx:
            sample = self[index]
            if len(sample.shape) == 3:
                sample = sample.argmax(0)
            visualize_matrices([sample])

    def __repr__(self):
        return "ARCAmazingDataset({})".format(len(self))


# In[ ]:


if __name__ == "__main__":
    dataset = ARCDataset(
        canvas_size=8,
        n_examples=10000,
    )


# In[ ]:


if __name__ == "__main__":
    # load dataset as a list of images
    dataset = load_dataset_as_images()
    load_task_as_images(filename="007bbfb7.json")
    len(dataset) # length is 3434
    dataset[0].size()[0]
    get_max_shape(dataset) # 30 x 30

    # visualize one image
#    numpy.set_printoptions(threshold=sys.maxsize)
    image1 = load_task_as_images(filename="007bbfb7.json")[0]
    centralize_image(image1, (11, 11))

    # preprocess data
    new_data = preprocess_data(dataset)
    len(new_data)
    new_data[0]

    from PIL import Image
    plt.imshow(new_data[14])

    for index in range(len(dataset)):
        if dataset[index].size()[0] == 30:
            plt.imshow(new_data[index])
            print(index)
    plt.imshow(new_data[13])

    new_data[0].shape

    # test split channel
    a = load_task_as_images(filename="007bbfb7.json")
    b = centralize_image(a[0], (11, 11))
    split_channel(b, (11, 11))

