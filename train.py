#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Script for training EBMs for discovering concepts, relations and operators.
"""
try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except:
    pass

import argparse
from collections import OrderedDict, Iterable
from copy import deepcopy
from datetime import datetime
import itertools
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
from numbers import Number
import os
import pdb
import pickle
import pprint as pp
import random
from typing import Union
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions.normal import Normal
from torch.distributions.one_hot_categorical import OneHotCategorical as cat
from torchvision import datasets, transforms, utils
from torchvision.transforms import functional as F_tr
from torchvision.transforms.functional import InterpolationMode
from einops import rearrange, repeat
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from zeroc.datasets.arc_image import ARCDataset
from zeroc.datasets.BabyARC.code.dataset.dataset import *
from zeroc.utils import ClevrImagePreprocessor
# from zeroc.clevr_dataset_gen.dataset import ClevrRelationDataset, create_easy_dataset
# from zeroc.clevr_dataset_gen.generate_concept_dataset import get_clevr_concept_data
from zeroc.argparser import get_args_EBM
# from zeroc.slot_attention.clevr import CLEVR
# from zeroc.slot_attention.multi_dsprites import MultiDsprites
# from zeroc.slot_attention.tetrominoes import Tetrominoes
from zeroc.concept_library.models import get_model_energy, load_model_energy, neg_mask_sgd, neg_mask_sgd_with_kl, id_to_tensor, requires_grad
from zeroc.concept_library.settings import REPR_DIM, DEFAULT_OBJ_TYPE
from zeroc.utils import REA_PATH, EXP_PATH, get_root_dir
from zeroc.concept_transfer import convert_babyarc
from zeroc.concept_library.util import to_cpu_recur, try_call, Printer, transform_dict, MineDataset, is_diagnose, reduce_tensor, get_hashing, pdump, pload, remove_elements, loss_op_core, filter_kwargs, to_Variable, gather_broadcast, get_pdict, COLOR_LIST, set_seed, Zip, Early_Stopping, init_args, make_dir, str2bool, get_filename, get_filename_short, get_machine_name, get_device, record_data, plot_matrices, filter_filename, get_next_available_key, to_np_array, to_Variable, get_filename_short, write_to_config, Dictionary, Batch, to_cpu
from zeroc.concept_library.util import model_parallel, color_dict, clip_grad, identity_fun, seperate_concept, to_one_hot, onehot_to_RGB, get_module_parameters, assign_embedding_value, get_hashing, to_device_recur, visualize_matrices, repeat_n, mask_iou_score, shrink, get_obj_from_mask
p = Printer()


# ## 1. Dataset:

# ### 1.1 ConceptDataset:

# In[ ]:


class ConceptDataset(Dataset):
    """Concept Dataset for learning basic concepts for ARC.

    mode:
        Concepts:  E(x; a; c)
            "Pixel": one or many pixels
            "Line": one or many lines
            "Rect": hollow rectangles
            "{}+{}+...": each "{}" can be a concept.

        Symmetries: E(x; a; c)
            "hFlip", "vFlip": one image where some object has property of symmetry w.r.t. hflip
            "Rotate": one image where some object has property of symmetry w.r.t. rotation.

        Relations: E(x; a1, a2; c)
            "Vertical": lines where some of them are vertical
            "Parallel": lines where some of them are parallel
            "Vertical+Parallel": lines where some of them are vertical or parallel
            "IsInside": obj_1 is inside obj_2
            "SameRow": obj_1 and obj_2 are at the same row
            "SameCol": obj_1 and obj_2 are at the same column

        Operations: E(x1,x2; a1,a2; c1,c2)
            "RotateA+vFlip(Line+Rect)": two images where some object1 in image1 is rotated or vertically-flipped w.r.t. some object2 in image2, and the objects are chosen from Line or Rect.
            "hFlip(Lshape)", "vFlip(Lshape+Line)": two images where some object1 in image1 is flipped w.r.t. some object2 in image2.

        ARC+:
            "arc^{}": ARC images with property "{}" masked as above.
        ""
    """
    def __init__(
        self,
        mode,
        canvas_size=8,
        n_examples=10000,
        rainbow_prob=0.,
        data=None,
        idx_list=None,
        concept_collection=None,
        allowed_shape_concept=None,
        w_type="image+mask",
        color_avail="-1",
        min_n_distractors=0,
        max_n_distractors=-1,
        n_operators=1,
        allow_connect=True,
        parsing_check=False,
        focus_type=None,
        transform=None,
        save_interval=-1,
        save_filename=None,
    ):
        if allowed_shape_concept is None:
            allowed_shape_concept=[
                "Line", "Rect", "RectSolid", "Lshape", "Randshape", "ARCshape",
                "Tshape", "Eshape",
                "Hshape", "Cshape", "Ashape", "Fshape",
                "RectE1a", "RectE1b", "RectE1c", 
                "RectE2a", "RectE2b", "RectE2c",
                "RectE3a", "RectE3b", "RectE3c", 
                "RectF1a", "RectF1b", "RectF1c", 
                "RectF2a", "RectF2b", "RectF2c",
                "RectF3a", "RectF3b", "RectF3c",
            ]
        self.mode = mode
        self.canvas_size = canvas_size
        self.rainbow_prob = rainbow_prob
        self.n_examples = n_examples
        self.allowed_shape_concept = allowed_shape_concept
        self.min_n_distractors = min_n_distractors
        self.max_n_distractors = max_n_distractors
        self.n_operators = n_operators
        self.w_type = w_type
        self.allow_connect = allow_connect
        self.parsing_check = parsing_check
        self.focus_type = focus_type
        if isinstance(color_avail, str):
            if color_avail == "-1":
                self.color_avail = None
            else:
                self.color_avail = [int(c) for c in color_avail.split(",")]
                for c in self.color_avail:
                    assert c >= 1 and c <= 9
        else:
            self.color_avail = color_avail
        if idx_list is None:
            assert data is None
            if mode.startswith("arc^"):
                if "(" in mode:
                    self.data = []
                    # Operator:
                    concept_raw = mode.split("(")[0].split("+")
                    concept_collection = []
                    for c in concept_raw:
                        if "^" in c:
                            concept_collection.append(c.split("^")[1])
                        else:
                            concept_collection.append(c)
                    self.concept_collection = concept_collection
                    arcDataset = ARCDataset(
                        n_examples=n_examples*2,
                        canvas_size=canvas_size,
                    )
                    babyArcDataset = BabyARCDataset(
                        pretrained_obj_cache=os.path.join(get_root_dir(), 'datasets/arc_objs.pt'),
                        save_directory=get_root_dir() + "/datasets/BabyARCDataset/",
                        object_limit=None,
                        noise_level=0,
                        canvas_size=canvas_size,
                    )
                    if set(self.concept_collection).issubset({"RotateA", "RotateB", "RotateC", 
                                                              "hFlip", "vFlip", "DiagFlipA", "DiagFlipB", 
                                                              "Identity", 
                                                              "Move"}):
                        for arc_example_one_hot in arcDataset:
                            arc_image = torch.zeros_like(arc_example_one_hot[0])
                            for i in range(0, 10):
                                arc_image += arc_example_one_hot[i]*i
                            arc_image = arc_image.type(torch.int32)

                            repre_dict = babyArcDataset.sample_task_canvas_from_arc(
                                arc_image,
                                color=np.random.choice([True, False], p=[0.6, 0.4]),
                                is_plot=False,
                            )
                            if repre_dict == -1:
                                continue
                            in_canvas = Canvas(repre_dict=repre_dict)

                            # Operate on the input:
                            if len(list(repre_dict["node_id_map"].keys())) == 0:
                                continue # empty arc canvas
                            chosen_obj_key = np.random.choice(list(repre_dict["node_id_map"].keys()))
                            chosen_obj_id = repre_dict["node_id_map"][chosen_obj_key]
                            chosen_op = np.random.choice(self.concept_collection)
                            if chosen_op in ["Identity"]:
                                inplace = True if random.random() < 0.5 else False
                                out_canvas_list, concept = OperatorEngine().operator_identity(
                                    [in_canvas],
                                    [[chosen_obj_key]],
                                    inplace=inplace,
                                )
                                if out_canvas_list == -1:
                                    continue
                            elif chosen_op in ["Move"]:
                                # create operator spec as move is a complex operator
                                move_spec = OperatorMoveSpec(
                                                autonomous=False,
                                                direction=random.randint(0,3), 
                                                distance=-1, 
                                                hit_type=None, # either wall, agent or None
                                                linkage_move=False, 
                                                linkage_move_distance_ratio=None,
                                            )
                                out_canvas_list, concept = OperatorEngine().operator_move(
                                    [in_canvas],
                                    [[chosen_obj_key]],
                                    [[move_spec]], 
                                    allow_overlap=False, 
                                    allow_shape_break=False,
                                    allow_connect=self.allow_connect,
                                    allow_stay=False,
                                )
                                if out_canvas_list == -1:
                                    continue
                            elif chosen_op in ["RotateA", "RotateB", "RotateC", "hFlip", "vFlip", "DiagFlipA", "DiagFlipB"]:
                                out_canvas_list, concept = OperatorEngine().operate_rotate(
                                    [in_canvas],
                                    [[chosen_obj_key]],
                                    operator_tag=f"#{chosen_op}",
                                    allow_connect=self.allow_connect,
                                    allow_shape_break=False,
                                )
                                if out_canvas_list == -1:
                                    continue
                            else:
                                raise Exception(f"operator={chosen_op} is not supported!")

                            # Add to self.data:
                            in_canvas_dict = in_canvas.repr_as_dict()
                            out_canvas_dict = out_canvas_list[0].repr_as_dict()
                            in_mask = in_canvas_dict["id_object_mask"][chosen_obj_id][None]
                            out_mask = out_canvas_dict["id_object_mask"][chosen_obj_id][None]
                            self.data.append(
                                ((to_one_hot(in_canvas_dict["image_t"]), to_one_hot(out_canvas_dict["image_t"])),
                                 (in_mask, out_mask),
                                 chosen_op,
                                 Dictionary({}),
                                )
                            )
                            if len(self.data) >= n_examples:
                                break
                            if i > n_examples * 2 and len(self.data) < n_examples * 0.05:
                                raise Exception("Sampled {} times and only {} of them satisfies the specified condition. Try relaxing the condition!".format(i, len(self.data)))
                else:
                    mode_core = mode.split("^")[1]
                    self.concept_collection = mode_core.split("+")
                    dataset = ARCDataset(
                        n_examples=n_examples,
                        canvas_size=canvas_size,
                    )
                    examples_all = []
                    masks_all = []
                    concepts_all = []
                    examples = dataset.data
                    examples_argmax = examples.argmax(1)
                    self.data = []
                    for i in range(len(examples)):
                        concept_dict = seperate_concept(examples_argmax[i])
                        masks, concepts = get_masks(concept_dict, allowed_concepts=self.concept_collection, canvas_size=canvas_size)
                        if masks is not None:
                            for mask, concept in zip(masks, concepts):
                                self.data.append((
                                        examples[i],
                                        (mask,),
                                        concept,
                                        Dictionary({}),
                                    )
                                )
            else:
                if "(" in mode:
                    # Operator:
                    self.concept_collection = mode.split("(")[0].split("+")
                    input_concepts = mode.split("(")[1][:-1].split("+")
                else:
                    self.concept_collection = mode.split("-")[-1].split("+")
                    input_concepts = [""]
                dataset = BabyARCDataset(
                    pretrained_obj_cache=os.path.join(get_root_dir(), 'datasets/arc_objs.pt'),
                    save_directory=get_root_dir() + "/datasets/BabyARCDataset/",
                    object_limit=None,
                    noise_level=0,
                    canvas_size=canvas_size,
                )
                concept_str_mapping = {
                    "line": "Line", 
                    "rectangle": "Rect", 
                    "rectangleSolid": "RectSolid",
                    "Lshape": "Lshape", 
                    "Tshape": "Tshape", 
                    "Eshape": "Eshape", 
                    "Hshape": "Hshape", 
                    "Cshape": "Cshape", 
                    "Ashape": "Ashape", 
                    "Fshape": "Fshape",
                    "randomShape": "Randshape",
                    "arcShape": "ARCshape"}  # Mapping between two conventions
                concept_str_reverse_mapping = {
                    "Line": "line", 
                    "Rect": "rectangle", 
                    "RectSolid": "rectangleSolid", 
                    "Lshape": "Lshape", 
                    "Tshape": "Tshape", 
                    "Eshape": "Eshape", 
                    "Hshape": "Hshape", 
                    "Cshape": "Cshape", 
                    "Ashape": "Ashape", 
                    "Fshape": "Fshape",
                    "Randshape": "randomShape",
                    "ARCshape": "arcShape"}  # Mapping between two conventions
                composite_concepts = [                
                    "RectE1a", "RectE1b", "RectE1c", 
                    "RectE2a", "RectE2b", "RectE2c",
                    "RectE3a", "RectE3b", "RectE3c", 
                    "RectF1a", "RectF1b", "RectF1c", 
                    "RectF2a", "RectF2b", "RectF2c",
                    "RectF3a", "RectF3b", "RectF3c",
                ]
                for c in composite_concepts:
                    concept_str_mapping[c] = c
                    concept_str_reverse_mapping[c] = c

                if set(get_c_core(self.concept_collection)).issubset({
                    "Image"
                }):
                    # Image is a collection of all shapes.
                    if max_n_distractors == -1:
                        max_n_objs = 3
                    else:
                        max_n_objs = 1 + max_n_distractors # 1 is for the core concept itself.
                    self.data = generate_samples(
                        dataset=dataset,
                        obj_spec_fun=obj_spec_fun,
                        n_examples=n_examples,
                        mode="concept-image",
                        concept_collection=["Line", "Rect", "Lshape", 
                                            "RectSolid", "Randshape", "ARCshape", 
                                            "Tshape", "Eshape", 
                                            "Hshape", "Cshape", "Ashape", "Fshape"],
                        min_n_objs=1+self.min_n_distractors,
                        max_n_objs=max_n_objs,
                        canvas_size=canvas_size,
                        rainbow_prob=rainbow_prob,
                        concept_str_mapping=concept_str_mapping,
                        concept_str_reverse_mapping=concept_str_reverse_mapping,
                        allowed_shape_concept=["Line", "Rect", "Lshape", 
                                               "RectSolid", "Randshape", "ARCshape", 
                                               "Tshape", "Eshape", 
                                               "Hshape", "Cshape", "Ashape", "Fshape"],
                        color_avail=self.color_avail,
                        allow_connect=self.allow_connect,
                        parsing_check=self.parsing_check,
                        save_interval=10,
                        save_filename=save_filename,
                    )
                elif set(get_c_core(self.concept_collection)).issubset({
                    "Line", "Rect", "Lshape",
                    "RectSolid", "Randshape", "ARCshape",
                    "Tshape", "Eshape",
                    "Hshape", "Cshape", "Ashape", "Fshape",
                    "RectE1a", "RectE1b", "RectE1c", 
                    "RectE2a", "RectE2b", "RectE2c",
                    "RectE3a", "RectE3b", "RectE3c", 
                    "RectF1a", "RectF1b", "RectF1c", 
                    "RectF2a", "RectF2b", "RectF2c",
                    "RectF3a", "RectF3b", "RectF3c",
                }):
                    if max_n_distractors == -1:
                        max_n_objs = 3
                    else:
                        max_n_objs = 1 + max_n_distractors # 1 is for the core concept itself.
                    self.data = generate_samples(
                        dataset=dataset,
                        obj_spec_fun=obj_spec_fun,
                        n_examples=n_examples,
                        mode="concept",
                        concept_collection=self.concept_collection,
                        min_n_objs=1+self.min_n_distractors,
                        max_n_objs=max_n_objs,
                        canvas_size=canvas_size,
                        rainbow_prob=rainbow_prob,
                        concept_str_mapping=concept_str_mapping,
                        concept_str_reverse_mapping=concept_str_reverse_mapping,
                        allowed_shape_concept=self.allowed_shape_concept,
                        color_avail=self.color_avail,
                        allow_connect=self.allow_connect,
                        parsing_check=self.parsing_check,
                        focus_type=self.focus_type,
                        save_interval=10,
                        save_filename=save_filename,
                    )
                elif set(self.concept_collection).issubset({
                    "RectE1a", "RectE1b", "RectE1c", 
                    "RectE2a", "RectE2b", "RectE2c",
                    "RectE3a", "RectE3b", "RectE3c", 
                    "RectF1a", "RectF1b", "RectF1c", 
                    "RectF2a", "RectF2b", "RectF2c",
                    "RectF3a", "RectF3b", "RectF3c",
                }):
                    max_n_objs = 1 # we currently don't allow distractors to be sampled.
                    self.data = generate_samples(
                        dataset=dataset,
                        obj_spec_fun=obj_spec_fun,
                        n_examples=n_examples,
                        mode="compositional-concept",
                        concept_collection=self.concept_collection,
                        min_n_objs=1+self.min_n_distractors,
                        max_n_objs=max_n_objs,
                        canvas_size=canvas_size,
                        rainbow_prob=rainbow_prob,
                        concept_str_mapping=concept_str_mapping,
                        concept_str_reverse_mapping=concept_str_reverse_mapping,
                        allowed_shape_concept=self.allowed_shape_concept,
                        color_avail=self.color_avail,
                        allow_connect=self.allow_connect,
                        parsing_check=self.parsing_check,
                        focus_type=self.focus_type,
                        save_interval=10,
                        save_filename=save_filename,
                    )
                elif set(self.concept_collection).issubset({"Vertical", "Parallel"}):
                    if max_n_distractors == -1:
                        max_n_objs = 3
                    else:
                        max_n_objs = 2 + max_n_distractors # 2 is for the core concept itself.
                    def obj_spec_fun_re(concept_collection, min_n_objs, max_n_objs, canvas_size, allowed_shape_concept=None, color_avail=None, focus_type=None):
                        n_objs = np.random.randint(min_n_objs, max_n_objs+1)
                        obj_spec = [(('obj_{}'.format(k), 'line_[-1,1,-1]'), 'Attr') for k in range(n_objs)]
                        return obj_spec
                    self.data = generate_samples(
                        dataset=dataset,
                        obj_spec_fun=obj_spec_fun_re,
                        n_examples=n_examples,
                        mode="relation",
                        concept_collection=self.concept_collection,
                        min_n_objs=2+self.min_n_distractors,
                        max_n_objs=max_n_objs,
                        canvas_size=canvas_size,
                        rainbow_prob=rainbow_prob,
                        concept_str_mapping=concept_str_mapping,
                        concept_str_reverse_mapping=concept_str_reverse_mapping,
                        allowed_shape_concept=self.allowed_shape_concept,
                        color_avail=self.color_avail,
                        allow_connect=self.allow_connect,
                        parsing_check=self.parsing_check,
                        save_interval=10,
                        save_filename=save_filename,
                    )
                elif set(self.concept_collection).issubset({"VerticalMid", "VerticalEdge", "VerticalSepa", "Parallel"}):
                    if max_n_distractors == -1:
                        max_n_objs = 3
                    else:
                        max_n_objs = 2 + max_n_distractors # 2 is for the core concept itself.
                    self.data = generate_lines_full_vertical_parallel(
                        n_examples=n_examples,
                        concept_collection=self.concept_collection,
                        min_n_objs=2+self.min_n_distractors,
                        max_n_objs=max_n_objs,
                        canvas_size=canvas_size,
                        min_size=3,
                        max_size=canvas_size-2,
                        color_avail=self.color_avail,
                        isplot=False,
                    )
                elif set(self.concept_collection).issubset({
                    "SameAll", "SameShape", "SameColor", 
                    "SameRow", "SameCol", "IsInside", 
                    "IsTouch", "IsNonOverlapXY",
                    "IsEnclosed",
                }):
                    if max_n_distractors == -1:
                        max_n_objs = 3
                    else:
                        max_n_objs = 2 + max_n_distractors # 2 is for the core relation itself.
                    def obj_spec_fun_re(
                        concept_collection, min_n_objs, max_n_objs, 
                        canvas_size, allowed_shape_concept=None, 
                        color_avail=None,
                        focus_type=None,
                    ):
                        assert allowed_shape_concept != None
                        n_objs = np.random.randint(min_n_objs, max_n_objs+1)
                        # two slots are for the relation
                        sampled_relation = np.random.choice(concept_collection)
                        obj_spec = [(('obj_0', 'obj_1'), sampled_relation)]
                        max_rect_size = canvas_size//2
                        # choose distractors
                        for k in range(2, n_objs):
                            # choose a distractor shape
                            distractor_shape = np.random.choice(allowed_shape_concept)
                            if distractor_shape == "Line":
                                obj_spec += [(('obj_{}'.format(k), 'line_[-1,-1,-1]'), 'Attr')]
                            elif distractor_shape == "Rect":
                                obj_spec += [(('obj_{}'.format(k), 'rectangle_[-1,-1]'), 'Attr')]
                            elif distractor_shape == "RectSolid":
                                obj_spec += [(('obj_{}'.format(k), 'rectangleSolid_[-1,-1]'), 'Attr')]
                            elif distractor_shape == "Lshape":
                                obj_spec += [(('obj_{}'.format(k), 'Lshape_[-1,-1,-1]'), 'Attr')]
                            elif distractor_shape == "Tshape":
                                w = np.random.randint(3, max_rect_size+2)
                                h = np.random.randint(3, max_rect_size+2)
                                obj_spec += [(('obj_{}'.format(k), f'Tshape_[{w},{h}]'), 'Attr')]
                            elif distractor_shape == "Eshape":
                                w = np.random.randint(3, max_rect_size+1)
                                h = np.random.randint(5, max_rect_size+3)
                                obj_spec += [(('obj_{}'.format(k), f'Eshape_[{w},{h}]'), 'Attr')]
                            elif distractor_shape == "Hshape":
                                w = np.random.randint(3, max_rect_size+2)
                                h = np.random.randint(3, max_rect_size+2)
                                obj_spec += [(('obj_{}'.format(k), f'Hshape_[{w},{h}]'), 'Attr')]
                            elif distractor_shape == "Cshape":
                                w = np.random.randint(3, max_rect_size+1)
                                h = np.random.randint(3, max_rect_size+2)
                                obj_spec += [(('obj_{}'.format(k), f'Cshape_[{w},{h}]'), 'Attr')]
                            elif distractor_shape == "Ashape":
                                w = np.random.randint(3, max_rect_size+2)
                                h = np.random.randint(4, max_rect_size+3)
                                obj_spec += [(('obj_{}'.format(k), f'Ashape_[{w},{h}]'), 'Attr')]
                            elif distractor_shape == "Fshape":
                                w = np.random.randint(3, max_rect_size+1)
                                h = np.random.randint(4, max_rect_size+3)
                                obj_spec += [(('obj_{}'.format(k), f'Fshape_[{w},{h}]'), 'Attr')]   
                            elif distractor_shape == "Randshape":
                                max_rect_size = canvas_size // 2
                                w, h = np.random.randint(2, max_rect_size+1, size=2) # hard-code for the size
                                obj_spec += [(('obj_{}'.format(k), f'randomShape_[{w},{h}]'), 'Attr')]
                            elif distractor_shape == "ARCshape":
                                max_rect_size = canvas_size // 2
                                w, h = np.random.randint(2, max_rect_size+1, size=2) # hard-code for the size
                                obj_spec += [(('obj_{}'.format(k), f'arcShape_[{w},{h}]'), 'Attr')]
                        return obj_spec
                    if len(input_concepts) == 1 and input_concepts[0] == "":
                        _shape_concept=[c for c in self.allowed_shape_concept]
                    else:
                        _shape_concept=[c for c in input_concepts]

                    self.data = generate_samples(
                        dataset=dataset,
                        obj_spec_fun=obj_spec_fun_re,
                        n_examples=n_examples,
                        mode="relation",
                        concept_collection=self.concept_collection,
                        min_n_objs=2+self.min_n_distractors,
                        max_n_objs=max_n_objs,
                        canvas_size=canvas_size,
                        rainbow_prob=rainbow_prob,
                        concept_str_mapping=concept_str_mapping,
                        concept_str_reverse_mapping=concept_str_reverse_mapping,
                        allowed_shape_concept=_shape_concept,
                        color_avail=self.color_avail,
                        allow_connect=self.allow_connect,
                        parsing_check=self.parsing_check,
                    )
                elif set(self.concept_collection).issubset({
                    "RotateA", "RotateB", "RotateC", 
                    "hFlip", "vFlip", "DiagFlipA", 
                    "DiagFlipB", "Identity", "Move"
                }):
                    if max_n_distractors == -1:
                        max_n_objs = 3
                    else:
                        max_n_objs = 1 + max_n_distractors # 1 is for the core operator itself.
                    self.data = []
                    for i in range(self.n_examples * 5):
                        # Initialize input concept instance:
                        obj_spec = obj_spec_fun(
                            concept_collection=input_concepts,
                            min_n_objs=1+self.min_n_distractors,
                            max_n_objs=max_n_objs,
                            canvas_size=canvas_size,
                        )
                        # get the number of the objects
                        operatable_obj_set = set([])
                        for spec in obj_spec:
                            if spec[1] == "Attr":
                                operatable_obj_set.add(spec[0][0])
                            else:
                                operatable_obj_set.add(spec[0][0])
                                operatable_obj_set.add(spec[0][1])
                        operatable_obj_set = list(operatable_obj_set)
                        # let us enable distractors
                        if set(input_concepts).issubset({"SameColor", "IsTouch"}):
                            n_distractors = np.random.randint(0, max_n_distractors+1)
                            max_rect_size = canvas_size//2
                            for i in range(n_distractors):
                                k = i+len(operatable_obj_set)
                                distractor_shape = np.random.choice(self.allowed_shape_concept)
                                if distractor_shape == "Line":
                                    obj_spec += [(('obj_{}'.format(k), 'line_[-1,-1,-1]'), 'Attr')]
                                elif distractor_shape == "Rect":
                                    obj_spec += [(('obj_{}'.format(k), 'rectangle_[-1,-1]'), 'Attr')]
                                elif distractor_shape == "RectSolid":
                                    obj_spec += [(('obj_{}'.format(k), 'rectangleSolid_[-1,-1]'), 'Attr')]
                                elif distractor_shape == "Lshape":
                                    obj_spec += [(('obj_{}'.format(k), 'Lshape_[-1,-1,-1]'), 'Attr')]
                                elif distractor_shape == "Tshape":
                                    w = np.random.randint(3, max_rect_size+2)
                                    h = np.random.randint(3, max_rect_size+2)
                                    obj_spec += [(('obj_{}'.format(k), f'Tshape_[{w},{h}]'), 'Attr')]
                                elif distractor_shape == "Eshape":
                                    w = np.random.randint(3, max_rect_size+1)
                                    h = np.random.randint(5, max_rect_size+3)
                                    obj_spec += [(('obj_{}'.format(k), f'Eshape_[{w},{h}]'), 'Attr')]
                                elif distractor_shape == "Hshape":
                                    w = np.random.randint(3, max_rect_size+2)
                                    h = np.random.randint(3, max_rect_size+2)
                                    obj_spec += [(('obj_{}'.format(k), f'Hshape_[{w},{h}]'), 'Attr')]
                                elif distractor_shape == "Cshape":
                                    w = np.random.randint(3, max_rect_size+1)
                                    h = np.random.randint(3, max_rect_size+2)
                                    obj_spec += [(('obj_{}'.format(k), f'Cshape_[{w},{h}]'), 'Attr')]
                                elif distractor_shape == "Ashape":
                                    w = np.random.randint(3, max_rect_size+2)
                                    h = np.random.randint(4, max_rect_size+3)
                                    obj_spec += [(('obj_{}'.format(k), f'Ashape_[{w},{h}]'), 'Attr')]
                                elif distractor_shape == "Fshape":
                                    w = np.random.randint(3, max_rect_size+1)
                                    h = np.random.randint(4, max_rect_size+3)
                                    obj_spec += [(('obj_{}'.format(k), f'Fshape_[{w},{h}]'), 'Attr')]     
                                elif distractor_shape == "Randshape":
                                    max_rect_size = canvas_size // 2
                                    w, h = np.random.randint(2, max_rect_size+1, size=2) # hard-code for the size
                                    obj_spec += [(('obj_{}'.format(k), f'randomShape_[{w},{h}]'), 'Attr')]
                                elif distractor_shape == "ARCshape":
                                    max_rect_size = canvas_size // 2
                                    w, h = np.random.randint(2, max_rect_size+1, size=2) # hard-code for the size
                                    obj_spec += [(('obj_{}'.format(k), f'arcShape_[{w},{h}]'), 'Attr')]
                        # get all objects include distractors
                        all_obj_set = set([])
                        for spec in obj_spec:
                            if spec[1] == "Attr":
                                all_obj_set.add(spec[0][0])
                            else:
                                all_obj_set.add(spec[0][0])
                                all_obj_set.add(spec[0][1])

                        repre_dict = dataset.sample_single_canvas_by_core_edges(
                            OrderedDict(obj_spec),
                            allow_connect=self.allow_connect,
                            rainbow_prob=rainbow_prob,
                            is_plot=False,
                            color_avail=self.color_avail,
                        )
                        if repre_dict == -1:
                            continue
                        in_canvas = Canvas(repre_dict=repre_dict)

                        # Operate on the input:
                        chosen_obj_id = np.random.choice(len(operatable_obj_set))
                        chosen_obj_name = operatable_obj_set[chosen_obj_id]
                        chosen_op = np.random.choice(self.concept_collection)
                        if chosen_op in ["Identity"]:
                            inplace = True if random.random() < 0.5 else False
                            out_canvas_list, concept = OperatorEngine().operator_identity(
                                [in_canvas],
                                [[chosen_obj_name]],
                                inplace=inplace,
                            )
                            if out_canvas_list == -1:
                                continue
                        elif chosen_op in [
                            "RotateA", "RotateB", "RotateC", 
                            "hFlip", "vFlip", "DiagFlipA", "DiagFlipB"
                        ]:
                            out_canvas_list, concept = OperatorEngine().operate_rotate(
                                [in_canvas],
                                [[chosen_obj_name]],
                                operator_tag=f"#{chosen_op}",
                                allow_connect=self.allow_connect,
                                allow_shape_break=False,
                            )
                            if out_canvas_list == -1:
                                continue
                        elif chosen_op in ["Move"]:
                            # create operator spec as move is a complex operator
                            move_spec = OperatorMoveSpec(
                                            autonomous=False,
                                            direction=random.randint(0,3), 
                                            distance=-1, 
                                            hit_type=None, # either wall, agent or None
                                            linkage_move=False, 
                                            linkage_move_distance_ratio=None,
                                        )
                            out_canvas_list, concept = OperatorEngine().operator_move(
                                [in_canvas],
                                [[chosen_obj_name]],
                                [[move_spec]], 
                                allow_overlap=False, 
                                allow_shape_break=False,
                                allow_connect=self.allow_connect,
                                allow_stay=False,
                            )
                            if out_canvas_list == -1:
                                continue
                        else:
                            raise Exception(f"operator={chosen_op} is not supported!")
                        
                        if n_operators > 1:
                            # operator distractor can act on all objects
                            addition_operators = min(len(all_obj_set)-1,n_operators-1) # we need to have minimum number of objs
                            operated_obj_name = set([])
                            operated_obj_name.add(chosen_obj_name)
                            
                            exclude_ops = set([chosen_op])
                            
                            # we need to operate on other objects.
                            for _ in range(n_operators-1):
                                addition_obj_set = set(all_obj_set) - operated_obj_name
                                addition_obj_name = np.random.choice(list(addition_obj_set))
                                
                                addition_ops = set(self.concept_collection) - exclude_ops
                                addition_op = np.random.choice(list(addition_ops))
                                exclude_ops.add(addition_op)
                                
                                # operate the the previous ouput canvas
                                if addition_op in ["Identity"]:
                                    inplace = True if random.random() < 0.5 else False
                                    out_canvas_list, concept = OperatorEngine().operator_identity(
                                        [out_canvas_list[0]],
                                        [[addition_obj_name]],
                                        inplace=inplace,
                                    )
                                    if out_canvas_list == -1:
                                        break
                                elif addition_op in ["RotateA", "RotateB", "RotateC", "hFlip", "vFlip", "DiagFlipA", "DiagFlipB"]:
                                    out_canvas_list, concept = OperatorEngine().operate_rotate(
                                        [out_canvas_list[0]],
                                        [[addition_obj_name]],
                                        operator_tag=f"#{addition_op}",
                                        allow_connect=self.allow_connect,
                                        allow_shape_break=False,
                                    )
                                    if out_canvas_list == -1:
                                        break
                                elif addition_op in ["Move"]:
                                    # create operator spec as move is a complex operator
                                    move_spec = OperatorMoveSpec(
                                                    autonomous=False,
                                                    direction=random.randint(0,3), 
                                                    distance=-1, 
                                                    hit_type=None, # either wall, agent or None
                                                    linkage_move=False, 
                                                    linkage_move_distance_ratio=None,
                                                )
                                    out_canvas_list, concept = OperatorEngine().operator_move(
                                        [out_canvas_list[0]],
                                        [[addition_obj_name]],
                                        [[move_spec]], 
                                        allow_overlap=False, 
                                        allow_shape_break=False,
                                        allow_connect=self.allow_connect,
                                        allow_stay=False,
                                    )
                                    if out_canvas_list == -1:
                                        break
                                else:
                                    raise Exception(f"operator={addition_op} is not supported!")
                                operated_obj_name.add(addition_obj_name)
                        if out_canvas_list == -1:
                            continue
                        # Add to self.data:
                        in_canvas_dict = in_canvas.repr_as_dict()
                        out_canvas_dict = out_canvas_list[0].repr_as_dict()
                        
                        in_mask = in_canvas_dict["id_object_mask"][in_canvas_dict["node_id_map"][chosen_obj_name]][None]
                        out_mask = out_canvas_dict["id_object_mask"][in_canvas_dict["node_id_map"][chosen_obj_name]][None]
                        # TODO: remove deprecated codes.
                        # in_mask = in_canvas_dict["id_object_mask"][chosen_obj_id][None]
                        # out_mask = out_canvas_dict["id_object_mask"][chosen_obj_id][None]
                        info = {"obj_spec": obj_spec}
                        self.data.append(
                            ((to_one_hot(in_canvas_dict["image_t"]), to_one_hot(out_canvas_dict["image_t"])),
                             (in_mask, out_mask),
                             chosen_op,
                             Dictionary(info),
                            )
                        )
                        if len(self.data) >= n_examples:
                            break
                        if i > n_examples * 2 and len(self.data) < n_examples * 0.05:
                            raise Exception("Sampled {} times and only {} of them satisfies the specified condition. Try relaxing the condition!".format(i, len(self.data)))
                else:
                    raise Exception("concept_collection {} is out of scope!".format(self.concept_collection))
            if "obj" in self.w_type and "mask" not in self.w_type:
                self.data = mask_to_obj(self.data)
            self.idx_list = list(range(len(self.data)))
            if len(self.idx_list) < n_examples:
                p.print("Dataset created with {} examples, less than {} specified.".format(len(self.idx_list), n_examples))
            else:
                p.print("Dataset for {} created.".format(mode))
        else:
            self.data = data
            self.idx_list = idx_list
            self.concept_collection = concept_collection
        self.transform = transform

    def __len__(self):
        return len(self.idx_list)

    def __repr__(self):
        return "ConceptDataset({})".format(len(self))

    def __getitem__(self, idx):
        """Get data instance, where idx can be a number or a slice."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        elif isinstance(idx, slice):
            return self.__class__(
                mode=self.mode,
                canvas_size=self.canvas_size,
                n_examples=self.n_examples,
                rainbow_prob=self.rainbow_prob,
                data=self.data,
                idx_list=self.idx_list[idx],
                concept_collection=self.concept_collection,
                w_type=self.w_type,
                color_avail=self.color_avail,
                max_n_distractors=self.max_n_distractors,
                n_operators=self.n_operators,
                transform=self.transform,
            )
        sample = self.data[self.idx_list[idx]]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def draw(self, idx):
        """Draw one of multiple data instances."""
        if not isinstance(idx, Iterable):
            idx = [idx]
        for index in idx:
            sample = self[index]
            if len(sample) == 4:
                p.print("example {}, {}:".format(index, sample[2]))
                if isinstance(sample[0], tuple):
                    visualize_matrices([sample[0][0].argmax(0), sample[0][1].argmax(0)])
                else:
                    visualize_matrices([sample[0].argmax(0)])
                plot_matrices([sample[1][i].squeeze() for i in range(len(sample[1]))], images_per_row=6)


class ConceptDataset3D(MineDataset):
    def draw(self, idx):
        """Draw one of multiple data instances."""
        if not isinstance(idx, Iterable):
            idx = [idx]
        for index in idx:
            sample = self[index]
            if len(sample) == 4:
                p.print("example {}, {}:".format(index, sample[2]))
                if isinstance(sample[0], tuple):
                    visualize_matrices([sample[0][0].argmax(0), sample[0][1].argmax(0)], use_color_dict=False)
                else:
                    visualize_matrices([sample[0]], use_color_dict=False)
                visualize_matrices([sample[1][i].squeeze() for i in range(len(sample[1]))])


# ### 1.2 ConceptFewshotDataset:

# In[ ]:


def generate_fewshot_dataset(args, concept_mode="standard", n_shot=1, n_queries_per_class=15):
    """For parsing + classify.
    The args.dataset has the format:
        f"pc-{concept_1}+{concept_2}+...+{concept_n}"
    """
    assert args.dataset.startswith("pc") or args.dataset.startswith("pg") or args.dataset.startswith("yc") 
    str_split = args.dataset.split("-")[1].split("^")
    concept_collection = str_split[-1].split("+") 
    concept_dict = {}
    assert n_shot == 1
    # Generate samples for each concept:
    for concept in concept_collection:
        concept_args = init_args({
            "dataset": "c-{}".format(concept),
            "seed": args.seed,
            "n_examples": args.n_examples * n_shot,
            "canvas_size": args.canvas_size,
            "rainbow_prob": 0.,
            "w_type": "image+mask",
            "color_avail": args.color_avail,
            "max_n_distractors": 0,
            "min_n_distractors": 0,
            "allow_connect": True, # No effect
            "parsing_check": False,
        })
        concept_dict[concept] = get_dataset(concept_args, verbose=False)[0]
    if args.dataset.startswith("pc"):
        example_dict = {}
        for concept in concept_collection:
            example_args = init_args({
                "dataset": "c-{}".format(concept),
                "seed": args.seed + 1,
                "n_examples": args.n_examples * n_queries_per_class,
                "canvas_size": args.canvas_size,
                "rainbow_prob": 0.,
                "w_type": "image+mask",
                "color_avail": args.color_avail,
                "max_n_distractors": 0,
                "min_n_distractors": 0,
                "allow_connect": True,
                "parsing_check": False,
            })
            example_dict[concept] = get_dataset(example_args, verbose=False)[0]
    elif args.dataset.startswith("yc"):
        example_dict = {}
        for concept in concept_collection:
            example_args = init_args({
                "dataset": "y-{}".format(concept),
                "seed_3d": args.seed_3d + args.num_processes_3d,
                "num_processes_3d": args.num_processes_3d,
                "color_map_3d": args.color_map_3d,
                "add_thick_surf": args.add_thick_surf,
                "add_thick_depth": args.add_thick_depth,
                "image_size_3d": args.image_size_3d,
                "n_examples": args.n_examples * n_queries_per_class,
                # 2D examples
                "seed": args.seed + 1,
                "use_seed_2d": args.use_seed_2d,
                "canvas_size": args.canvas_size,
                "rainbow_prob": 0.,
                "w_type": "image+mask",
                "color_avail": args.color_avail,
                "max_n_distractors": 0,
                "min_n_distractors": 0,
                "allow_connect": True,
                "parsing_check": False,
            })
            example_dict[concept] = get_dataset(example_args, verbose=False, is_load=True)[0]
    elif args.dataset.startswith("pg"):
        example_dict = {}
        examples_collection = str_split[0].split("+")
        # The tasks are split evenly among the possible concepts for demonstration
        n_tasks = [args.n_examples // len(concept_collection)] * len(concept_collection)
        n_tasks[-1] += args.n_examples % len(concept_collection)
        for idx, concept in enumerate(concept_collection):
            example_args = init_args({
                "dataset": "c-{}+{}^{}".format(concept, str_split[0], concept),
                "seed": args.seed + 1,
                "n_examples": n_tasks[idx] * n_queries_per_class,
                "canvas_size": args.canvas_size,
                "rainbow_prob": 0.,
                "w_type": "image+mask",
                "color_avail": args.color_avail,
                "max_n_distractors": 2, # Important: There can be distractors in examples
                "parsing_check": False,
            })
            example_dict[concept] = get_dataset(example_args, verbose=False)[0]
    else:
        raise
    if args.dataset[1] == "c":
        if concept_mode == "standard":
            dataset = generate_fewshot_dataset_standard(
                concept_dict=concept_dict,
                example_dict=example_dict,
                concept_collection=concept_collection,
                n_examples=args.n_examples,
                n_shot=n_shot,
                n_queries_per_class=n_queries_per_class,
            )
        elif concept_mode == "random":
            dataset = generate_fewshot_dataset_random(
                concept_dict=concept_dict,
                example_dict=example_dict,
                concept_collection=concept_collection,
                n_examples=args.n_examples,
                n_shot=n_shot,
            )
        else:
            raise Exception("concept_mode '{}' is not valid!".format(concept_mode))
    elif args.dataset[1] == "g":
        if concept_mode == "standard":
            dataset = generate_fewshot_grounding_dataset(
                concept_dict=concept_dict,
                example_dict=example_dict,
                concept_collection=concept_collection,
                n_tasks=n_tasks,
                n_shot=n_shot,
                n_queries_per_class=n_queries_per_class,
            )
        else:
            raise Exception("concept_mode '{}' is not valid!".format(concept_mode))
    else:
        raise
    dataset.concept_collection = concept_collection
    return dataset


def generate_fewshot_dataset_standard(
    concept_dict,
    example_dict,
    concept_collection,
    n_examples,
    n_shot=1,
    n_queries_per_class=15,
):
    """
    Format:
        Each data in data_list has the format:
            (data_concept, data_concept_mask, data_concept_id), (data_examples, data_examples_mask, data_examples_id), info
            where data_examples, data_examples_mask, data_examples_id are all lists of all concept examples 
            according to concept_collection.
    """
    # Generate dataset:
    data_list = []
    concept_id_dict = {}
    example_id_dict = {}
    data_list = []
    info = {}
    for concept in concept_collection:
        concept_id_dict[concept] = np.random.choice(n_examples * n_shot, size=n_examples * n_shot, replace=False)
        example_id_dict[concept] = np.random.choice(n_examples * n_queries_per_class, size=n_examples * n_queries_per_class, replace=False)

    for i in range(n_examples):
        # Obtaining the concept examples:
        concept_collection_permute = np.random.choice(concept_collection, size=len(concept_collection), replace=False)
        data_concepts = []
        data_concepts_mask = []
        data_concepts_id = []
        info["concept_info"] = []
        for j, concept in enumerate(concept_collection_permute):
            concept_id = concept_id_dict[concept][i]
            data_ele = concept_dict[concept][concept_id]
            data_concepts.append(data_ele[0])
            data_concepts_mask.append(data_ele[1][0])
            data_concepts_id.append(data_ele[2])
            info["concept_info"].append(data_ele[3])

        # Obtaining the query set:
        data_examples = []
        data_examples_mask = []
        data_examples_id = []
        info["example_info"] = []
        for example_id in concept_collection:
            for kk in range(n_queries_per_class):
                data_ele = example_dict[example_id][i * n_queries_per_class + kk]
                assert data_ele[2] in example_id
                data_examples.append(data_ele[0])
                data_examples_mask.append(data_ele[1][0])
                data_examples_id.append(data_ele[2])
                info["example_info"].append(data_ele[3])
        example_num = n_queries_per_class * len(concept_collection)
        assert len(data_examples) == example_num
        permute_ids_example = np.random.choice(example_num, size=example_num, replace=False)
        data_examples = [data_examples[id] for id in permute_ids_example]
        data_examples_mask = [data_examples_mask[id] for id in permute_ids_example]
        data_examples_id = [data_examples_id[id] for id in permute_ids_example]
        data = ((tuple(data_concepts), tuple(data_concepts_mask), tuple(data_concepts_id)),
                (tuple(data_examples), tuple(data_examples_mask), tuple(data_examples_id)), info)
        data_list.append(data)
    permuted_ids = np.random.choice(n_examples, size=n_examples, replace=False)
    data_list_final = [data_list[i] for i in permuted_ids]
    data_list_final = ConceptFewshotDataset(data=data_list_final, concept_mode="standard")
    return data_list_final


def generate_fewshot_grounding_dataset(
    concept_dict,
    example_dict,
    concept_collection,
    n_tasks,
    n_shot=1,
    n_queries_per_class=15,
):
    task_id = []
    for idx, num_tasks in enumerate(n_tasks):
        # Keep track of the demonstrated concept and the task index for that concept
        task_id.extend([(i, concept_collection[idx]) for i in range(num_tasks)])
    n_examples = len(task_id)
    assert n_examples == sum(n_tasks)
    
    data_list = []
    concept_id_dict = {}
    example_id_dict = {}
    info = {}
    # Randomize the order of concept demonstrations and examples
    for idx, concept in enumerate(concept_collection):
        concept_id_dict[concept] = np.random.choice(n_examples * n_shot, size=n_examples * n_shot, replace=False)
        example_id_dict[concept] = np.random.choice(n_tasks[idx] * n_queries_per_class, size=n_tasks[idx] * n_queries_per_class, replace=False)
    
    # Each task has a single concept that is demonstrated
    for i, concept_id in task_id:
        # Get concept demonstrations
        concept_data_ele = [concept_dict[concept_id][concept_id_dict[concept_id][i * n_shot + j]] for j in range(n_shot)]
        data_concepts = []
        data_concepts_mask = []
        data_concepts_id = []
        info["concept_info"] = []
        for data_ele in concept_data_ele:
            data_concepts.append(data_ele[0])
            data_concepts_mask.append(data_ele[1][0])
            data_concepts_id.append(data_ele[2])
            info["concept_info"].append(data_ele[3])
        
        # Get query set
        data_examples = []
        data_examples_mask = []
        data_examples_id = []
        info["example_info"] = []
        for kk in range(n_queries_per_class):
            data_ele = example_dict[concept_id][example_id_dict[concept_id][i * n_queries_per_class + kk]]
            assert data_ele[2] == concept_id
            data_examples.append(data_ele[0])
            data_examples_mask.append(data_ele[1][0])
            data_examples_id.append(data_ele[2])
            info["example_info"].append(data_ele[3])
        example_num = n_queries_per_class
        assert len(data_examples) == example_num
        data = ((tuple(data_concepts), tuple(data_concepts_mask), tuple(data_concepts_id)),
                (tuple(data_examples), tuple(data_examples_mask), tuple(data_examples_id)), info)
        data_list.append(data)
    permuted_ids = np.random.choice(n_examples, size=n_examples, replace=False)
    data_list_final = [data_list[i] for i in permuted_ids]
    data_list_final = ConceptFewshotDataset(data=data_list_final, concept_mode="standard")
    return data_list_final


def generate_fewshot_dataset_random(concept_dict, example_dict, concept_collection, n_examples, n_shot=1):
    """
    Format:
        Each data in data_list has the format:
            (data_concept,), (data_examples, data_examples_mask, data_examples_id), info
            where data_examples, data_examples_mask, data_examples_id are all lists of all concept examples 
            according to concept_collection.
    """
    # Generate dataset:
    data_list = []
    query_id_dict = {}
    example_id_dict = {}
    data_list = []
    num = int(np.ceil(n_examples / len(concept_collection)))
    for concept in concept_collection:
        query_id_dict[concept] = np.random.choice(len(concept_dict[concept]), size=num*n_shot, replace=False)
        example_id_dict[concept] = np.random.choice(n_examples * 2, size=num, replace=False)
    for concept in concept_collection:
        for i, id in enumerate(query_id_dict[concept]):
            info = {}
            data_concept = (concept_dict[concept][id][0],)
            info["concept_mask"] = (concept_dict[concept][id][1][0],)
            info["concept_id"] = concept_dict[concept][id][2]
            assert concept == concept_dict[concept][id][2]
            info["concept_info"] = (concept_dict[concept][id][3],)
            concept_collection_permute = np.random.choice(concept_collection, size=len(concept_collection), replace=False)
            data_examples = []
            data_examples_mask = []
            data_examples_id = []
            info["example_info"] = []
            for j, concept_example in enumerate(concept_collection_permute):
                example_id = example_id_dict[concept_example][j]
                data_examples.append(example_dict[concept_example][example_id][0])
                data_examples_mask.append(example_dict[concept_example][example_id][1][0])
                data_examples_id.append(example_dict[concept_example][example_id][2])
                info["example_info"].append(example_dict[concept_example][example_id][3])
            data = (data_concept, (tuple(data_examples), tuple(data_examples_mask), tuple(data_examples_id)), info)
            data_list.append(data)
    permuted_ids = np.random.choice(num * len(concept_collection), n_examples, replace=False)
    data_list_final = [data_list[i] for i in permuted_ids]
    data_list_final = ConceptFewshotDataset(data=data_list_final, concept_mode="random")
    return data_list_final


class ConceptFewshotDataset(MineDataset):
    """
    Format:
        Each data in data_list has the format:
            (data_concept,), (data_examples, data_examples_mask, data_examples_id), info
            where data_examples, data_examples_mask, data_examples_id are all lists of all concept examples 
            according to concept_collection.
    """
    def __init__(self, *args, **kwargs):
        filtered_kwargs = deepcopy(kwargs)
        filtered_kwargs.pop("concept_mode")
        super().__init__(*args, **filtered_kwargs)
        self.concept_mode = kwargs["concept_mode"]

    def draw(self, idx):
        """Draw one of multiple data instances."""
        if not isinstance(idx, Iterable):
            idx = [idx]
        print("Drawing dataset with concept_collection of {}:".format(self.concept_collection))
        for index in idx:
            sample = self[index]
            num_concepts = len(sample[0][0])
            num_queries = len(sample[1][0])
            num_ex_channels = sample[1][0][0].shape[0]
            is_ex_3d = (num_ex_channels == 3)
            if self.concept_mode == "standard":
                print(f"Concept demonstration for task {index}:")
                visualize_matrices([concept_image.argmax(0) for concept_image in sample[0][0]], subtitles=sample[0][2], images_per_row=num_concepts)
                print("Instances and masks:")
                if is_ex_3d:
                    visualize_matrices([F.interpolate(concept_instance[None], (32,32), mode="nearest")[0] for concept_instance in sample[1][0]], 
                               use_color_dict=False, subtitles=sample[1][2], images_per_row=min(6, num_concepts))
                    plot_matrices([concept_instance_mask.squeeze(0) for concept_instance_mask in sample[1][1]], images_per_row=min(6, num_concepts))
                else:
                    visualize_matrices([concept_instance.argmax(0) for concept_instance in sample[1][0]], 
                                       subtitles=sample[1][2], images_per_row=min(6, num_concepts))
                    plot_matrices([concept_instance_mask.squeeze(0) for concept_instance_mask in sample[1][1]], images_per_row=min(6, num_concepts))
            elif self.concept_mode == "random":
                label = sample[2]["concept_id"]
                print("example {}, concept '{}':".format(index, label))
                print(f"Concept demonstration for task {index}:")
                visualize_matrices([concept_image.argmax(0) for concept_image in sample[0]], subtitles=[label], images_per_row=6)
                print("Instances and masks:")
                if is_ex_3d:
                    visualize_matrices([concept_instance for concept_instance in sample[1][0]], 
                               use_color_dict=False, subtitles=["[{}]".format(ele) if ele == label else ele for ele in sample[1][2]])
                    visualize_matrices([concept_instance_mask.squeeze(0) for concept_instance_mask in sample[1][1]])
                else:
                    visualize_matrices([concept_instance.argmax(0) for concept_instance in sample[1][0]], 
                                       subtitles=["[{}]".format(ele) if ele == label else ele for ele in sample[1][2]])
                    visualize_matrices([concept_instance_mask.squeeze(0) for concept_instance_mask in sample[1][1]])


# ### 1.3 ConceptCompositionDataset:

# In[ ]:


class ConceptCompositionDataset(Dataset):
    """Concept Composition dataset for learning to compose concepts from elementary concepts.

    mode:
        Concepts:  E(x; a; c)
            "Pixel": one or many pixels
            "Line": one or many lines
            "Rect": hollow rectangles
            "{}+{}+...": each "{}" can be a concept.

        Relations: E(x; a1, a2; c)
            "Vertical": lines where some of them are vertical
            "Parallel": lines where some of them are parallel
            "Vertical+Parallel": lines where some of them are vertical or parallel
            "IsInside": obj_1 is inside obj_2
            "SameRow": obj_1 and obj_2 are at the same row
            "SameCol": obj_1 and obj_2 are at the same column

        Operations: E(x1,x2; a1,a2; c1,c2)
            "RotateA+vFlip(Line+Rect)": two images where some object1 in image1 is rotated or vertically-flipped w.r.t. some object2 in image2, and the objects are chosen from Line or Rect.
            "hFlip(Lshape)", "vFlip(Lshape+Line)": two images where some object1 in image1 is flipped w.r.t. some object2 in image2.

        ARC+:
            "arc^{}": ARC images with property "{}" masked as above.
        ""
    """
    def __init__(
        self,
        canvas_size=8,
        n_examples=10000,
        concept_avail=None,
        relation_avail=None,
        additional_concepts=None,
        n_concepts_range=(2,3),
        relation_structure="None",
        rainbow_prob=0.,
        data=None,
        idx_list=None,
        color_avail="-1",
        min_n_distractors=0,
        max_n_distractors=0,
        n_examples_per_task=5,
        transform=None,
    ):
        self.canvas_size = canvas_size
        self.n_examples = n_examples
        self.concept_avail = concept_avail
        self.relation_avail = relation_avail
        self.additional_concepts = additional_concepts
        self.n_concepts_range = n_concepts_range
        self.relation_structure = relation_structure
        self.rainbow_prob = rainbow_prob
        self.min_n_distractors = min_n_distractors
        self.max_n_distractors = max_n_distractors
        self.n_examples_per_task = n_examples_per_task

        if isinstance(color_avail, str):
            if color_avail == "-1":
                self.color_avail = None
            else:
                self.color_avail = [int(c) for c in color_avail.split(",")]
                for c in self.color_avail:
                    assert c >= 1 and c <= 9
        else:
            self.color_avail = color_avail

        if idx_list is None:
            assert data is None
            dataset_engine = BabyARCDataset(
                pretrained_obj_cache=os.path.join(get_root_dir(), 'datasets/arc_objs.pt'),
                save_directory=get_root_dir() + "/datasets/BabyARCDataset/",
                object_limit=None,
                noise_level=0,
                canvas_size=canvas_size,
            )
            self.data = []
            for i in range(self.n_examples * 3):
                task = sample_selector(
                    dataset_engine=dataset_engine,
                    concept_avail=concept_avail,
                    relation_avail=relation_avail,
                    additional_concepts=self.additional_concepts,
                    n_concepts_range=n_concepts_range,
                    relation_structure=relation_structure,
                    min_n_distractors=min_n_distractors,
                    max_n_distractors=max_n_distractors,
                    canvas_size=canvas_size,
                    color_avail=self.color_avail,
                    n_examples_per_task=n_examples_per_task,
                    max_n_trials=5,
                    isplot=False,
                )
                if len(task) == n_examples_per_task:
                    self.data.append(task)
                if len(self.data) % 100 == 0:
                    p.print("Number of tasks generated: {}".format(len(self.data)))
                if len(self.data) >= self.n_examples:
                    break

            self.idx_list = list(range(len(self.data)))
            if len(self.idx_list) < n_examples:
                p.print("Dataset created with {} examples, less than {} specified.".format(len(self.idx_list), n_examples))
            else:
                p.print("Dataset created with {} examples.".format(len(self.idx_list)))
        else:
            self.data = data
            self.idx_list = idx_list
        self.transform = transform

    def __len__(self):
        return len(self.idx_list)

    def __repr__(self):
        return "ConceptDataset({})".format(len(self))

    def __getitem__(self, idx):
        """Get data instance, where idx can be a number or a slice."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        elif isinstance(idx, slice):
            return self.__class__(
                canvas_size=self.canvas_size,
                n_examples=self.n_examples,
                concept_avail=self.concept_avail,
                relation_avail=self.relation_avail,
                additional_concepts=self.additional_concepts,
                n_concepts_range=self.n_concepts_range,
                relation_structure=self.relation_structure,
                rainbow_prob=self.rainbow_prob,
                data=self.data,
                idx_list=self.idx_list[idx],
                color_avail=self.color_avail,
                min_n_distractors=self.min_n_distractors,
                max_n_distractors=self.max_n_distractors,
                n_examples_per_task=self.n_examples_per_task,
                transform=self.transform,
            )

        sample = self.data[self.idx_list[idx]]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def to_dict(self):
        Dict = {}
        for id in self.idx_list:
            Dict[str(id)] = self.data[id]
        return Dict

    def draw(self, idx):
        """Draw one of multiple data instances."""
        if not isinstance(idx, Iterable):
            idx = [idx]
        for index in idx:
            task = self[index]
            if len(task) == self.n_examples_per_task:
                info = task[0][3]
                p.print("structure: {}".format(info["structure"]))
                p.print("obj_spec_core:")
                pp.pprint(info["obj_spec_core"])
                for k, example in enumerate(task):
                    p.print("task {}, example {}:".format(index, k))
                    if isinstance(example[0], tuple):
                        visualize_matrices([example[0][0].argmax(0), example[0][1].argmax(0) if example[0][1].shape[0] == 10 else example[0][1].squeeze(0)])
                    else:
                        visualize_matrices([example[0].argmax(0)])


# ### 1.4 ConceptClevrDataset:

# In[ ]:


class ConceptClevrDataset(MineDataset):
    def draw(self, idx, filename=None):
        """Draw one of multiple data instances."""
        if not isinstance(idx, Iterable):
            idx = [idx]
        for index in idx:
            sample = self[index]
            if len(sample) == 4:
                p.print("example {}, {}:".format(index, sample[2]))
                if isinstance(sample[0], tuple):
                    visualize_matrices([sample[0][0].argmax(0), sample[0][1].argmax(0)], use_color_dict=False, filename=filename)
                else:
                    visualize_matrices([sample[0]], use_color_dict=False, filename=filename)
                plot_matrices([sample[1][i].squeeze() for i in range(len(sample[1]))], images_per_row=6, filename=filename)


# ### 1.4 Dataset helper functions:

# In[ ]:


def sample_selector(
    dataset_engine,
    concept_avail=None,
    relation_avail=None,
    additional_concepts=None,
    n_concepts_range=None,
    relation_structure=None,
    max_n_distractors=0,
    min_n_distractors=0,
    canvas_size=8,
    color_avail=None,
    n_examples_per_task=5,
    max_n_trials=5,
    isplot=False,
):
    def is_exist_all(obj_full_relations, key):
        is_all = True
        for tuple_ele, lst in obj_full_relations.items():
            assert isinstance(tuple_ele, tuple)
            if key not in lst and (tuple_ele[0].startswith("obj_") and tuple_ele[1].startswith("obj_")):
                is_all = False
                break
        return is_all
    assert max_n_distractors >= min_n_distractors
    if concept_avail is None:
        concept_avail = [
            "Line", "Rect", "RectSolid", "Randshape", "ARCshape",
            "Lshape", "Tshape", "Eshape",
            "Hshape", "Cshape", "Ashape", "Fshape",
        ]
    if relation_avail is None:
        relation_avail = [
            "SameAll", "SameShape", "SameColor",
            "SameRow", "SameCol", 
            "SubsetOf", "IsInside", "IsTouch",
        ]
    allowed_shape_concept = concept_avail
    if additional_concepts is not None:
        allowed_shape_concept = allowed_shape_concept + additional_concepts
    concept_str_reverse_mapping = {
        "Line": "line", 
        "Rect": "rectangle", 
        "RectSolid": "rectangleSolid", 
        "Lshape": "Lshape", 
        "Tshape": "Tshape", 
        "Eshape": "Eshape", 
        "Hshape": "Hshape", 
        "Cshape": "Cshape", 
        "Ashape": "Ashape", 
        "Fshape": "Fshape",
        "Randshape": "randomShape",
        "ARCshape": "arcShape",
    }

    if relation_structure == "None":
        # Only concept dataset:
        n_concepts_range = (n_concepts_range, n_concepts_range) if (not isinstance(n_concepts_range, tuple)) and n_concepts_range > 0 else n_concepts_range
        assert isinstance(n_concepts_range, tuple)
        # Sample concepts:
        obj_spec_core = obj_spec_fun(
            concept_collection=concept_avail,
            min_n_objs=n_concepts_range[0],
            max_n_objs=n_concepts_range[1],
            canvas_size=canvas_size,
            color_avail=color_avail,
        )
        obj_id = len(obj_spec_core)
        refer_node_id = None
        structure = None
    else:
        structure_dict = {
            "2a": ["pivot", (0,1), "(refer)"],
            "2ai":["pivot:Rect", (1,0,"IsInside"), "(refer)"],
            "3a": ["pivot", (0,1), "(concept)", (1,2), "(refer)"],
            "3ai":["pivot:Rect", (1,0,"IsInside"), "(concept)", (1,2), "(refer)"],
            "3b": ["pivot", "pivot", (0,2), (1,2), "(refer)"],
            "4a": ["pivot", (0,1), "concept", (1,2), "(concept)", (2,3), "(refer)"],
            "4ai":["pivot:Rect", (1,0,"IsInside"), "(concept)", (1,2), "(concept)", (2,3), "(refer)"],
            "4b": ["pivot", "pivot", (0,2), (1,2), "(concept)", (2,3), "(refer)"],
        }

        # Sample pivot concept:
        structure_key = np.random.choice(relation_structure.split("+"))
        structure = structure_dict[structure_key]
        is_valid = False
        for i in range(3):
            obj_id = 0
            obj_spec_core = []
            refer_node_id = None
            relations_all = []
            for j, element in enumerate(structure):
                if isinstance(element, tuple):
                    if len(element) == 2:
                        if structure_key in ["2a", "3a"] and j == 1:
                            relation = np.random.choice(remove_elements(relation_avail, ["SameShape", "SameAll"]))
                        else:
                            relation = np.random.choice(remove_elements(relation_avail, ["SameAll"]))
                    elif len(element) == 3:
                        relation = np.random.choice(element[2].split("+"))
                    obj_spec_core.append(
                        [("obj_{}".format(element[0]),
                          "obj_{}".format(element[1])),
                          relation,
                    ])
                    relations_all.append(relation)
                elif element.startswith("pivot"):
                    if ":" in element:
                        concept_avail_core = element.split(":")[1].split("+")
                    else:
                        concept_avail_core = concept_avail
                    obj_spec = obj_spec_fun(
                        concept_collection=concept_avail_core,
                        min_n_objs=1,
                        max_n_objs=1,
                        canvas_size=canvas_size,
                        color_avail=color_avail,
                        idx_start=obj_id,
                    )[0]
                    obj_spec_core.append(obj_spec)
                    obj_id += 1
                elif element in ["concept", "refer", "(concept)", "(refer)"]:
                    if not element.startswith("("):
                        obj_spec = obj_spec_fun(
                            concept_collection=allowed_shape_concept,
                            min_n_objs=1,
                            max_n_objs=1,
                            canvas_size=canvas_size,
                            color_avail=color_avail,
                            idx_start=obj_id,
                        )[0]
                        obj_spec_core.append(obj_spec)
                    if element in ["refer", "(refer)"]:
                        assert refer_node_id is None
                        refer_node_id = obj_id
                    obj_id += 1
                else:
                    raise
            relations_unique = np.unique(relations_all)
            if len(relations_unique) == 1:
                if structure_key.startswith("3") and relations_unique[0] in ["SameColor"]:
                    pass
                elif relations_unique[0] in ["SameShape"] or structure_key.startswith("3"):
                    continue
            is_valid = True
            break
        if not is_valid:
            return []

    task = []
    for k in range(n_examples_per_task * 4):
        selector_dict = OrderedDict()
        if max_n_distractors > 0:
            n_distractors = np.random.choice(range(min_n_distractors, max_n_distractors + 1))
            obj_spec_distractor = obj_spec_fun(
                concept_collection=additional_concepts,
                min_n_objs=n_distractors,
                max_n_objs=n_distractors,
                canvas_size=canvas_size,
                color_avail=color_avail,
                idx_start=obj_id,
            )
        else:
            obj_spec_distractor = []

        obj_spec_all = obj_spec_core + obj_spec_distractor
        selector_dict = OrderedDict(obj_spec_all)

        is_valid = False
        for j in range(max_n_trials):
            canvas_dict = dataset_engine.sample_single_canvas_by_core_edges(
                selector_dict,
                allow_connect=True, is_plot=False, rainbow_prob=0.0,
                concept_collection=[concept_str_reverse_mapping[s] for s in concept_avail],
                parsing_check=True,
                color_avail=color_avail,
            )
            if canvas_dict == -1:
                continue
            else:
                is_valid = True
                if isplot:
                    canvas = Canvas(repre_dict=canvas_dict)
                    canvas.render()
                    plt.show()
                break

        if is_valid:
            image = to_one_hot(canvas_dict["image_t"])
            info = Dictionary({"obj_masks" : {}})
            for k, v in canvas_dict["node_id_map"].items():
                info["obj_masks"][k] = canvas_dict["id_object_mask"][v]
            info["obj_full_relations"] = canvas_dict["partial_relation_edges"]
            # Make sure that if the structure has "i" (IsInside), only the obj_1 is inside obj_0 and no other objects:
            if relation_structure != "None":
                n_objs = len(info["obj_masks"])
                for i in range(2, n_objs):
                    if (f"obj_{i}", "obj_0") in info["obj_full_relations"] and "IsInside" in info["obj_full_relations"][(f"obj_{i}", "obj_0")]:
                        p.print(f"obj_{i} is also inside the Rect!")
                        is_valid = False
                        break
                if len(info['obj_masks']) == 3 and is_exist_all(info['obj_full_relations'], key="SameColor"):
                    # is_valid = False
                    pass
                if len(info['obj_masks']) == 2 and is_exist_all(info['obj_full_relations'], key="SameShape"):
                    is_valid = False
            if not is_valid:
                continue

            info["obj_spec_core"] = obj_spec_core
            info["obj_spec_all"] = obj_spec_all
            info["obj_spec_distractor"] = obj_spec_distractor
            info["refer_node_id"] = "obj_{}".format(refer_node_id)
            info["structure"] = structure
            masks = None
            chosen_concepts = None
            if relation_structure != "None":
                target = info["obj_masks"][info["refer_node_id"]][None]
                example = ((image, target), masks, chosen_concepts, info)
            else:
                example = (image, masks, chosen_concepts, info)
            task.append(example)
        else:
            continue

        if len(task) >= n_examples_per_task:
            break
    return task


def get_c_core(c):
    if isinstance(c, list):
        return [get_c_core(ele) for ele in c]
    else:
        assert isinstance(c, str)
        return c.split("[")[0]


def get_c_size(c):
    assert isinstance(c, str)
    if "[" in c:
        string = "[{}]".format(c.split("[")[1][:-1])
        min_size, max_size = eval(string)
    else:
        min_size, max_size = None, None
    return min_size, max_size


def get_masks(concept_dict, allowed_concepts, canvas_size):
    canvas_all = []
    concepts_all = []
    for key, item in concept_dict.items():
        if key in allowed_concepts:
            canvas = torch.zeros(len(item), 1, canvas_size, canvas_size)
            for j, pos in enumerate(item):
                canvas[j, :, pos[0]: pos[0]+pos[2], pos[1]: pos[1]+pos[3]] = 1
            canvas_all.append(canvas)
            concepts_all += [key] * len(item)
    if len(canvas_all) > 0:
        canvas_all = torch.cat(canvas_all)
        return canvas_all, concepts_all
    else:
        return None, None


def obj_spec_fun(
    concept_collection, 
    min_n_objs, max_n_objs, 
    canvas_size, 
    allowed_shape_concept=None,
    is_conjuncture=True,
    color_avail=None,
    idx_start=0,
    focus_type=None,
):
    """Generate specs for several objects for BabyARC.

    Args:
        idx_start: obj id to start with.
    """
    n_objs = np.random.randint(min_n_objs, max_n_objs+1)
    obj_spec = []
    if focus_type is not None:
        assert focus_type in concept_collection
    if set(get_c_core(concept_collection)).issubset({
        "Line", "Rect", "RectSolid", 
        "Lshape", "Randshape", "ARCshape", 
        "Tshape", "Eshape", 
        "Hshape", "Cshape", "Ashape", "Fshape",
        "RectE1a", "RectE1b", "RectE1c", 
        "RectE2a", "RectE2b", "RectE2c",
        "RectE3a", "RectE3b", "RectE3c", 
        "RectF1a", "RectF1b", "RectF1c", 
        "RectF2a", "RectF2b", "RectF2c",
        "RectF3a", "RectF3b", "RectF3c",
    }):
        if focus_type is None:
            partition = np.sort(np.random.choice(n_objs+1, len(concept_collection)-1, replace=True))
            max_rect_size = canvas_size // 2
            for k in range(idx_start, n_objs + idx_start):
                if len(concept_collection) == 1:
                    chosen_concept = concept_collection[0]
                else:
                    gt = k-idx_start >= partition  # gt: greater_than_vector
                    if gt.any():
                        id = np.where(gt)[0][-1] + 1
                    else:
                        id = 0
                    chosen_concept = concept_collection[id]
                chosen_concept_core = get_c_core(chosen_concept)
                min_size, max_size = get_c_size(chosen_concept)
                if chosen_concept_core == "Line":
                    if min_size is None:
                        obj_spec.append((('obj_{}'.format(k), 'line_[-1,1,-1]'), 'Attr'))
                    else:
                        h = np.random.randint(min_size, max_size+1)
                        obj_spec.append((('obj_{}'.format(k), f'line_[{h},1,-1]'), 'Attr'))
                elif chosen_concept_core == "Rect":
                    if min_size is None:
                        w, h = np.random.randint(3, max_rect_size+1, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    obj_spec.append((('obj_{}'.format(k), 'rectangle_[{},{}]'.format(w,h)), 'Attr'))
                elif chosen_concept_core == "RectSolid":
                    if min_size is None:
                        w, h = np.random.randint(2, max_rect_size+1, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    obj_spec.append((('obj_{}'.format(k), 'rectangleSolid_[{},{}]'.format(w,h)), 'Attr'))
                elif chosen_concept_core == "Lshape":
                    if min_size is None:
                        w, h = np.random.randint(3, max_rect_size+1, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    direction = np.random.randint(4)
                    obj_spec.append((('obj_{}'.format(k), 'Lshape_[{},{},{}]'.format(w,h,direction)), 'Attr'))
                elif chosen_concept_core == "Tshape":
                    if min_size is None:
                        w, h = np.random.randint(3, max_rect_size+2, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    obj_spec += [(('obj_{}'.format(k), f'Tshape_[{w},{h}]'), 'Attr')]
                elif chosen_concept_core == "Eshape":
                    if min_size is None:
                        w = np.random.randint(3, max_rect_size+1)
                        h = np.random.randint(5, max_rect_size+3)
                    else:
                        w = np.random.randint(min_size, max_size-1)
                        h = np.random.randint(min_size, max_size+1)
                    obj_spec += [(('obj_{}'.format(k), f'Eshape_[{w},{h}]'), 'Attr')]
                elif chosen_concept_core == "Hshape":
                    if min_size is None:
                        w, h = np.random.randint(3, max_rect_size+2, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    obj_spec += [(('obj_{}'.format(k), f'Hshape_[{w},{h}]'), 'Attr')]
                elif chosen_concept_core == "Cshape":
                    if min_size is None:
                        w = np.random.randint(3, max_rect_size+1)
                        h = np.random.randint(3, max_rect_size+2)
                    else:
                        w = np.random.randint(min_size, max_size)
                        h = np.random.randint(min_size, max_size+1)
                    obj_spec += [(('obj_{}'.format(k), f'Cshape_[{w},{h}]'), 'Attr')]
                elif chosen_concept_core == "Ashape":
                    if min_size is None:
                        w = np.random.randint(3, max_rect_size+2)
                        h = np.random.randint(4, max_rect_size+3)
                    else:
                        w = np.random.randint(min_size, max_size)
                        h = np.random.randint(min_size, max_size+1)
                    obj_spec += [(('obj_{}'.format(k), f'Ashape_[{w},{h}]'), 'Attr')]
                elif chosen_concept_core == "Fshape":
                    if min_size is None:
                        w = np.random.randint(3, max_rect_size+1)
                        h = np.random.randint(4, max_rect_size+3)
                    else:
                        w = np.random.randint(min_size, max_size-1)
                        h = np.random.randint(min_size, max_size+1)
                    obj_spec += [(('obj_{}'.format(k), f'Fshape_[{w},{h}]'), 'Attr')]   
                elif chosen_concept_core == "Randshape":
                    if min_size is None:
                        w, h = np.random.randint(2, max_rect_size+1, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    obj_spec.append((('obj_{}'.format(k), 'randomShape_[{},{}]'.format(w,h)), 'Attr'))
                elif chosen_concept_core == "ARCshape":
                    if min_size is None:
                        w, h = np.random.randint(2, max_rect_size+1, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    obj_spec.append((('obj_{}'.format(k), 'arcShape_[{},{}]'.format(w,h)), 'Attr'))
                elif chosen_concept_core in [
                    "RectE1a", "RectE1b", "RectE1c", 
                    "RectE2a", "RectE2b", "RectE2c",
                    "RectE3a", "RectE3b", "RectE3c", 
                    "RectF1a", "RectF1b", "RectF1c", 
                    "RectF2a", "RectF2b", "RectF2c",
                    "RectF3a", "RectF3b", "RectF3c",
                ]:
                    w, h = -1, -1 # let the canvas size drives here!
                    obj_spec.append((('obj_{}'.format(k), '{}_[{},{}]'.format(chosen_concept_core, w,h)), 'Attr'))
                else:
                    raise
            obj_spec = np.random.permutation(obj_spec).tolist()
        else:
            concept_collection = deepcopy(concept_collection)
            concept_collection.remove(focus_type)
            partition = np.sort(np.random.choice(n_objs, len(concept_collection)-1, replace=True))
            max_rect_size = canvas_size // 2
            for k in range(idx_start, n_objs + idx_start):
                if k == n_objs + idx_start - 1:
                    chosen_concept = focus_type
                else:
                    if len(concept_collection) == 1:
                        chosen_concept = concept_collection[0]
                    else:
                        gt = k-idx_start >= partition  # gt: greater_than_vector
                        if gt.any():
                            id = np.where(gt)[0][-1] + 1
                        else:
                            id = 0
                        chosen_concept = concept_collection[id]
                chosen_concept_core = get_c_core(chosen_concept)
                min_size, max_size = get_c_size(chosen_concept)
                if chosen_concept_core == "Line":
                    if min_size is None:
                        obj_spec.append((('obj_{}'.format(k), 'line_[-1,1,-1]'), 'Attr'))
                    else:
                        h = np.random.randint(min_size, max_size+1)
                        obj_spec.append((('obj_{}'.format(k), f'line_[{h},1,-1]'), 'Attr'))
                elif chosen_concept_core == "Rect":
                    if min_size is None:
                        w, h = np.random.randint(3, max_rect_size+1, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    obj_spec.append((('obj_{}'.format(k), 'rectangle_[{},{}]'.format(w,h)), 'Attr'))
                elif chosen_concept_core == "RectSolid":
                    if min_size is None:
                        w, h = np.random.randint(2, max_rect_size+1, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    obj_spec.append((('obj_{}'.format(k), 'rectangleSolid_[{},{}]'.format(w,h)), 'Attr'))
                elif chosen_concept_core == "Lshape":
                    if min_size is None:
                        w, h = np.random.randint(3, max_rect_size+1, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    direction = np.random.randint(4)
                    obj_spec.append((('obj_{}'.format(k), 'Lshape_[{},{},{}]'.format(w,h,direction)), 'Attr'))
                elif chosen_concept_core == "Tshape":
                    if min_size is None:
                        w, h = np.random.randint(3, max_rect_size+2, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    obj_spec += [(('obj_{}'.format(k), f'Tshape_[{w},{h}]'), 'Attr')]
                elif chosen_concept_core == "Eshape":
                    if min_size is None:
                        w = np.random.randint(3, max_rect_size+1)
                        h = np.random.randint(5, max_rect_size+3)
                    else:
                        w = np.random.randint(min_size, max_size-1)
                        h = np.random.randint(min_size, max_size+1)
                    obj_spec += [(('obj_{}'.format(k), f'Eshape_[{w},{h}]'), 'Attr')]
                elif chosen_concept_core == "Hshape":
                    if min_size is None:
                        w, h = np.random.randint(3, max_rect_size+2, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    obj_spec += [(('obj_{}'.format(k), f'Hshape_[{w},{h}]'), 'Attr')]
                elif chosen_concept_core == "Cshape":
                    if min_size is None:
                        w = np.random.randint(3, max_rect_size+1)
                        h = np.random.randint(3, max_rect_size+2)
                    else:
                        w = np.random.randint(min_size, max_size)
                        h = np.random.randint(min_size, max_size+1)
                    obj_spec += [(('obj_{}'.format(k), f'Cshape_[{w},{h}]'), 'Attr')]
                elif chosen_concept_core == "Ashape":
                    if min_size is None:
                        w = np.random.randint(3, max_rect_size+2)
                        h = np.random.randint(4, max_rect_size+3)
                    else:
                        w = np.random.randint(min_size, max_size)
                        h = np.random.randint(min_size, max_size+1)
                    obj_spec += [(('obj_{}'.format(k), f'Ashape_[{w},{h}]'), 'Attr')]
                elif chosen_concept_core == "Fshape":
                    if min_size is None:
                        w = np.random.randint(3, max_rect_size+1)
                        h = np.random.randint(4, max_rect_size+3)
                    else:
                        w = np.random.randint(min_size, max_size-1)
                        h = np.random.randint(min_size, max_size+1)
                    obj_spec += [(('obj_{}'.format(k), f'Fshape_[{w},{h}]'), 'Attr')]   
                elif chosen_concept_core == "Randshape":
                    if min_size is None:
                        w, h = np.random.randint(2, max_rect_size+1, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    obj_spec.append((('obj_{}'.format(k), 'randomShape_[{},{}]'.format(w,h)), 'Attr'))
                elif chosen_concept_core == "ARCshape":
                    if min_size is None:
                        w, h = np.random.randint(2, max_rect_size+1, size=2)
                    else:
                        w, h = np.random.randint(min_size, max_size+1, size=2)
                    obj_spec.append((('obj_{}'.format(k), 'arcShape_[{},{}]'.format(w,h)), 'Attr'))
                elif chosen_concept_core in [
                    "RectE1a", "RectE1b", "RectE1c", 
                    "RectE2a", "RectE2b", "RectE2c",
                    "RectE3a", "RectE3b", "RectE3c", 
                    "RectF1a", "RectF1b", "RectF1c", 
                    "RectF2a", "RectF2b", "RectF2c",
                    "RectF3a", "RectF3b", "RectF3c",
                ]:
                    w, h = -1, -1 # let the canvas size drives here!
                    obj_spec.append((('obj_{}'.format(k), '{}_[{},{}]'.format(chosen_concept_core, w,h)), 'Attr'))
                else:
                    raise
            obj_spec = obj_spec[-1:] + np.random.permutation(obj_spec[:-1]).tolist()
    elif set(concept_collection).issubset({"SameColor", "IsTouch"}):
        if len(concept_collection) > 1:
            if is_conjuncture:
                # Hard code probability
                if color_avail == None:
                    random_color = np.random.randint(1, 10)
                else:
                    random_color = random.choice(color_avail)
                obj_spec.append((('obj_{}'.format(idx_start), 'obj_{}'.format(idx_start+1)), 'IsTouch'))
                obj_spec.append((('obj_{}'.format(idx_start), f'color_[{random_color}]'), 'Attr'))
                obj_spec.append((('obj_{}'.format(idx_start+1), f'color_[{random_color}]'), 'Attr'))
            else:
                pass # TODO: not implemented
        else:
            if len(concept_collection) == 1:
                chosen_concept = concept_collection[0]
                if chosen_concept == "SameColor":
                    obj_spec.append((('obj_{}'.format(idx_start), 'obj_{}'.format(idx_start+1)), 'SameColor'))
                else:
                    obj_spec.append((('obj_{}'.format(idx_start), 'obj_{}'.format(idx_start+1)), 'IsTouch'))
    # complex shape.
    elif set(concept_collection).issubset({"RectE1a", "RectE1b", "RectE1c", 
                                           "RectE2a", "RectE2b", "RectE2c",
                                           "RectE3a", "RectE3b", "RectE3c", 
                                           "RectF1a", "RectF1b", "RectF1c", 
                                           "RectF2a", "RectF2b", "RectF2c",
                                           "RectF3a", "RectF3b", "RectF3c",}):
        chosen_concept = random.choice(concept_collection)
        if chosen_concept == "RectE1a" or chosen_concept == "RectF1a" or chosen_concept == "RectE1b" or chosen_concept == "RectF1b":
            char_shape = "E" if "E" in chosen_concept else "F"
            out_w = np.random.randint(5, 17)
            out_h = np.random.randint(5, 17)
            in_w = np.random.randint(4, out_w)
            in_h = np.random.randint(4, out_h)
            char_w = np.random.randint(3, 9)
            char_h = np.random.randint(5, 9)
            obj_spec = [(('obj_0', f'rectangle_[{out_w},{out_h}]'), 'Attr'), 
                         (('obj_1', f'rectangle_[{in_w},{in_h}]'), 'Attr'), 
                         (('obj_0', 'obj_1'), 'IsOutside'),
                         (('obj_2', f'{char_shape}shape_[{char_w},{char_h}]'), 'Attr')]
        elif chosen_concept == "RectE1c" or chosen_concept == "RectF1c":
            char_shape = "E" if "E" in chosen_concept else "F"
            out_w = np.random.randint(5, 17)
            out_h = np.random.randint(5, 17)
            in_w = np.random.randint(4, out_w)
            in_h = np.random.randint(4, out_h)
            char_w = np.random.randint(3, 9)
            char_h = np.random.randint(5, 9)
            obj_spec = [(('obj_0', f'rectangle_[{out_w},{out_h}]'), 'Attr'), 
                         (('obj_1', f'rectangle_[{in_w},{in_h}]'), 'Attr'), 
                         (('obj_0', 'obj_1'), 'IsOutside'),
                         (('obj_2', f'{char_shape}shape_[{char_w},{char_h}]'), 'Attr'), 
                         (('obj_1', 'obj_2'), 'SameColor')]
        elif chosen_concept == "RectE2a" or chosen_concept == "RectF2a" or chosen_concept == "RectE2b" or chosen_concept == "RectF2b":
            char_shape = "E" if "E" in chosen_concept else "F"
            out_w = np.random.randint(5, 17)
            out_h = np.random.randint(5, 17)
            in_w = np.random.randint(4, 8)
            in_h = np.random.randint(4, 8)
            char_w = np.random.randint(3, 8)
            char_h = np.random.randint(5, 8)
            obj_spec = [(('obj_0', f'rectangle_[{out_w},{out_h}]'), 'Attr'), 
                         (('obj_1', f'rectangle_[{in_w},{in_h}]'), 'Attr'), 
                         (('obj_0', 'obj_1'), 'IsOutside'),
                         (('obj_2', f'{char_shape}shape_[{char_w},{char_h}]'), 'Attr'), 
                         (('obj_0', 'obj_2'), 'IsOutside')]
        elif chosen_concept == "RectE2c" or chosen_concept == "RectF2c":
            char_shape = "E" if "E" in chosen_concept else "F"
            out_w = np.random.randint(5, 17)
            out_h = np.random.randint(5, 17)
            in_w = np.random.randint(4, 8)
            in_h = np.random.randint(4, 8)
            char_w = np.random.randint(3, 8)
            char_h = np.random.randint(5, 8)
            obj_spec = [(('obj_0', f'rectangle_[{out_w},{out_h}]'), 'Attr'), 
                         (('obj_1', f'rectangle_[{in_w},{in_h}]'), 'Attr'), 
                         (('obj_0', 'obj_1'), 'IsOutside'),
                         (('obj_2', f'{char_shape}shape_[{char_w},{char_h}]'), 'Attr'), 
                         (('obj_0', 'obj_2'), 'IsOutside'), 
                         (('obj_1', 'obj_2'), 'SameColor')]
        elif chosen_concept == "RectE3a" or chosen_concept == "RectF3a" or chosen_concept == "RectE3b" or chosen_concept == "RectF3b":
            char_shape = "E" if "E" in chosen_concept else "F"
            out_w = np.random.randint(9, 17)
            out_h = np.random.randint(9, 17)
            in_w = np.random.randint(7, out_w-1)
            in_h = np.random.randint(7, out_h-1)
            char_w = np.random.randint(3, in_w-1)
            char_h = np.random.randint(5, in_h-1)
            obj_spec = [(('obj_0', f'rectangle_[{out_w},{out_h}]'), 'Attr'), 
                         (('obj_1', f'rectangle_[{in_w},{in_h}]'), 'Attr'), 
                         (('obj_0', 'obj_1'), 'IsOutside'),
                         (('obj_2', f'Eshape_[{char_w},{char_h}]'), 'Attr'), 
                         (('obj_1', 'obj_2'), 'IsOutside')]
        elif chosen_concept == "RectE3c" or chosen_concept == "RectF3c":
            char_shape = "E" if "E" in chosen_concept else "F"
            out_w = np.random.randint(9, 17)
            out_h = np.random.randint(9, 17)
            in_w = np.random.randint(7, out_w-1)
            in_h = np.random.randint(7, out_h-1)
            char_w = np.random.randint(3, in_w-1)
            char_h = np.random.randint(5, in_h-1)
            obj_spec = [(('obj_0', f'rectangle_[{out_w},{out_h}]'), 'Attr'), 
                         (('obj_1', f'rectangle_[{in_w},{in_h}]'), 'Attr'), 
                         (('obj_0', 'obj_1'), 'IsOutside'),
                         (('obj_2', f'Eshape_[{char_w},{char_h}]'), 'Attr'), 
                         (('obj_1', 'obj_2'), 'IsOutside'), 
                         (('obj_1', 'obj_2'), 'SameColor')]
    else:
        raise Exception("concept_collection {} must be a subset of 'Line', 'Rect', 'Lshape', 'Randshape'!".format(concept_collection))
    return obj_spec


def generate_samples(
    dataset,
    obj_spec_fun,
    n_examples,
    mode,
    concept_collection,
    min_n_objs,
    max_n_objs,
    canvas_size,
    rainbow_prob=0.,
    allow_connect=True,
    parsing_check=False,
    focus_type=None,
    inspect_interval="auto",
    save_interval=-1,
    save_filename=None,
    **kwargs
):
    data = []
    if inspect_interval == "auto":
        inspect_interval = max(1, n_examples // 100)
    for i in range(int(n_examples * 150)):
        info = {}
        obj_spec = obj_spec_fun(
            concept_collection=concept_collection,
            min_n_objs=min_n_objs,
            max_n_objs=max_n_objs,
            canvas_size=canvas_size,
            allowed_shape_concept=kwargs["allowed_shape_concept"],
            color_avail=kwargs["color_avail"],
            focus_type=focus_type,
        )
        info["obj_spec"] = obj_spec

        if mode == "relation":
            concept_limits = {kwargs["concept_str_reverse_mapping"][get_c_core(c)]: get_c_size(c) for c in kwargs["allowed_shape_concept"] if get_c_size(c)[0] is not None}
            canvas_dict = dataset.sample_single_canvas_by_core_edges(
                OrderedDict(obj_spec),
                allow_connect=allow_connect,
                rainbow_prob=rainbow_prob,
                is_plot=False,
                concept_collection=[kwargs["concept_str_reverse_mapping"][s] for s in get_c_core(kwargs["allowed_shape_concept"])],
                parsing_check=parsing_check,
                color_avail=kwargs["color_avail"],
                concept_limits=concept_limits,
            )
        else:
            canvas_dict = dataset.sample_single_canvas_by_core_edges(
                OrderedDict(obj_spec),
                allow_connect=allow_connect,
                rainbow_prob=rainbow_prob,
                is_plot=False,
                parsing_check=parsing_check,
                color_avail=kwargs["color_avail"],
            )
        if canvas_dict != -1:
            info["node_id_map"] = canvas_dict["node_id_map"]
            info["id_object_mask"] = canvas_dict["id_object_mask"]
            n_sampled_objs = len(canvas_dict['id_object_mask'])
            if mode == "concept":
                if focus_type is None:
                    for k in range(n_sampled_objs):
                        # The order of id is the same as its first appearance in the obj_spec:
                        data.append((
                            to_one_hot(canvas_dict["image_t"]),
                            (canvas_dict['id_object_mask'][k][None],),
                            kwargs["concept_str_mapping"][obj_spec[k][0][1].split("_")[0]],
                            Dictionary(info),
                        ))
                        if len(data) >= n_examples:
                            break
                else:
                    data.append((
                        to_one_hot(canvas_dict["image_t"]),
                        (canvas_dict['id_object_mask'][0][None],),
                        kwargs["concept_str_mapping"][obj_spec[0][0][1].split("_")[0]],
                        Dictionary(info),
                    ))
            elif mode == "concept-image":
                for k in range(n_sampled_objs):
                    # The order of id is the same as its first appearance in the obj_spec:
                    data.append((
                        to_one_hot(canvas_dict["image_t"]),
                        (canvas_dict['id_object_mask'][k][None],),
                        "Image",
                        Dictionary(info),
                    ))
                    if len(data) >= n_examples:
                        break
            elif mode == "relation":
                if set(concept_collection).issubset({"Parallel", "Vertical"}):
                    def get_chosen_rel(canvas_dict, obj_ids):
                        chosen_direction = []
                        for id in obj_ids:
                            shape = canvas_dict['id_object_map'][id].shape
                            if shape[0] > shape[1]:
                                chosen_direction.append("0")
                            elif shape[0] < shape[1]:
                                chosen_direction.append("1")
                            else:
                                raise Exception("Line must have unequal height and width!")
                        if len(set(chosen_direction)) == 1:
                            chosen_concept = "Parallel"
                        else:
                            assert len(set(chosen_direction)) == 2
                            chosen_concept = "Vertical"
                        return chosen_concept

                    chosen_obj_ids = np.random.choice(n_sampled_objs, size=2, replace=False)
                    masks = list(canvas_dict['id_object_mask'].values())
                    chosen_obj_types = [obj_spec[id][0][1].split("_")[0] for id in chosen_obj_ids]
                    assert set(np.unique(chosen_obj_types)) == {"line"}
                    chosen_masks = [masks[id][None] for id in chosen_obj_ids]  # after: each mask has shape [1, H, W]
                    chosen_concept = get_chosen_rel(canvas_dict, chosen_obj_ids)
                    if chosen_concept not in concept_collection:
                        continue
                    # Consider all relations 
                    relations = []
                    for id1 in range(n_sampled_objs):
                        for id2 in range(id1 + 1, n_sampled_objs):
                            relation = get_chosen_rel(canvas_dict, [id1, id2])
                            relations.append((id1, id2, relation))
                    info["relations"] = relations
                    data.append((
                        to_one_hot(canvas_dict["image_t"]),
                        tuple(chosen_masks),
                        chosen_concept,
                        Dictionary(info),
                    ))
                if set(concept_collection).issubset({
                    "SameAll", "SameShape", "SameColor", 
                    "SameRow", "SameCol", "IsInside", "IsTouch",
                    "IsNonOverlapXY", "IsEnclosed",
                }):
                    masks = list(canvas_dict['id_object_mask'].values())
                    chosen_concept = obj_spec[0][1]
                    if chosen_concept not in concept_collection:
                        continue
                    chosen_obj_ids = [0,1] # we assum it is the first two objs have the relation type always!
                    if chosen_concept == "IsInside":
                        chosen_obj_ids = [1,0] # reverse it.
                        chosen_masks = [masks[id][None] for id in chosen_obj_ids]
                    else:
                        chosen_masks = [masks[id][None] for id in chosen_obj_ids]
                    data.append((
                        to_one_hot(canvas_dict["image_t"]),
                        tuple(chosen_masks),
                        chosen_concept,
                        Dictionary(info),
                    ))
            elif mode == "compositional-concept":
                # we need to filter.
                assert len(concept_collection) == 1 # we only allow 1.
                chosen_concept = concept_collection[0]
                failed = False
                if "1" in chosen_concept:
                    mid_ax_y_l = canvas_dict["id_position_map"][1][0].tolist() + (canvas_dict["id_object_map"][1].shape[0])//2
                    mid_ax_x_l = canvas_dict["id_position_map"][1][1].tolist() + (canvas_dict["id_object_map"][1].shape[1])//2

                    mid_ax_y_r = canvas_dict["id_position_map"][2][0].tolist() + (canvas_dict["id_object_map"][2].shape[0])//2
                    mid_ax_x_r = canvas_dict["id_position_map"][2][1].tolist() + (canvas_dict["id_object_map"][2].shape[1])//2

                    if abs(mid_ax_y_l-mid_ax_y_r) <= 2 or abs(mid_ax_x_l-mid_ax_x_r) <= 2:
                        if ('obj_2', 'obj_0') in canvas_dict["partial_relation_edges"]:
                            if "IsInside" in canvas_dict["partial_relation_edges"][('obj_2', 'obj_0')]:
                                failed = True
                        if ('obj_2', 'obj_1') in canvas_dict["partial_relation_edges"]:
                            if "IsInside" in canvas_dict["partial_relation_edges"][('obj_2', 'obj_1')]:
                                failed = True
                    else:
                        failed = True
                if "2" in chosen_concept:
                    if ('obj_2', 'obj_1') in canvas_dict["partial_relation_edges"]:
                        if "IsInside" in canvas_dict["partial_relation_edges"][('obj_2', 'obj_1')]:
                            failed = True
                if "3" in chosen_concept:
                    pass
                if "b" in chosen_concept:
                    if ('obj_1', 'obj_2') in canvas_dict["partial_relation_edges"]:
                        if "SameColor" in canvas_dict["partial_relation_edges"][('obj_1', 'obj_2')]:
                            failed = True
                if not failed:
                    data.append((
                        to_one_hot(canvas_dict["image_t"]),
                        (canvas_dict["image_t"][None].bool().float(),),
                        "Compositional-Image",
                        Dictionary(info),
                    ))
                    if inspect_interval != "None" and len(data) % inspect_interval == 0:
                        p.print("Finished generating {} out of {} examples.".format(len(data), n_examples))

        if inspect_interval != "None" and len(data) % inspect_interval == 0:
            p.print("Finished generating {} out of {} examples.".format(len(data), n_examples))
        if save_filename is not None and save_interval != -1 and len(data) % save_interval == 0:
            try_call(pdump, args=[data, save_filename])
            p.print("Save intermediate file at {}, with {} examples.".format(save_filename, len(data)))
        if len(data) >= n_examples:
            break
        if i > n_examples * 2 and len(data) < n_examples * 0.005:
            raise Exception("Sampled {} times and only {} of them satisfies the specified condition. Try relaxing the condition!".format(i, len(data)))
    return data


def get_chosen_line_rel(pos1, pos2):
    chosen_direction = []
    # "0" corresponds to upright, "1" corresponds to horizontal
    chosen_direction.append("0" if pos1[2] > pos1[3] else "1")
    chosen_direction.append("0" if pos2[2] > pos2[3] else "1")
    if len(set(chosen_direction)) == 1:
        chosen_concept = "Parallel"
    else:
        assert len(set(chosen_direction)) == 2
        # Set the horizontal and upright 
        if pos1[2] > pos1[3]:
            hori = pos2
            upr = pos1
        else:
            hori = pos1
            upr = pos2
        # Determine whether the intersection is more like a T shape or a rotated T shape
        # by checking if the upright line falls between horizontal line's left and right edges
        isT = upr[1] >= hori[1] and upr[1] < hori[1] + hori[3]
        isRotT= hori[0] >= upr[0] and hori[0] < upr[0] + upr[2]
        if isT:
            # First check for separate by comparing top edge of vertical with hori
            dist1 = abs(upr[0] - hori[0])
            dist2 = abs(upr[0] + upr[2] - 1 - hori[0])
            if min(dist1, dist2) > 1:
                chosen_concept = "VerticalSepa"
            else:
                # Check where the lines are touching w.r.t. horizontal line
                if upr[1] == hori[1] or upr[1] == hori[1] + hori[3] - 1:
                    chosen_concept = "VerticalEdge"
                else:
                    chosen_concept = "VerticalMid"
        elif isRotT:
            dist1 = abs(hori[1] - upr[1])
            dist2 = abs(hori[1] + hori[3] - 1 - upr[1])
            if min(dist1, dist2) > 1:
                chosen_concept = "VerticalSepa"
            else:
                # Check where the lines are touching w.r.t. vertical line
                if hori[0] == upr[0] or hori[0] == upr[0] + upr[2] - 1:
                    chosen_concept = "VerticalEdge"
                else:
                    chosen_concept = "VerticalMid"
        else:
            chosen_concept = "VerticalSepa"
    return chosen_concept


# Line with "VerticalMid", "VerticalEdge", "VerticalNot", "Parallel":
def generate_lines_full_vertical_parallel(
    n_examples,
    concept_collection=["VerticalMid", "VerticalEdge", "VerticalNot", "Parallel"],
    min_n_objs=2,
    max_n_objs=4,
    canvas_size=16,
    min_size=2,
    max_size=None,
    color_avail=None,
    isplot=False,
):                  
    if color_avail is None:
        color_avail = [1,2,3,4,5,6,7,8,9]
    data = []
    if max_size is None:
        max_size = canvas_size
    for i in range(int(n_examples*1.5)):
        if i % 1000 == 0:
            p.print(i)
        image = torch.zeros(1, canvas_size, canvas_size)
        # Sample relation from concept_collection:
        relation = np.random.choice(concept_collection)
        if relation == "Parallel":
            direction = np.random.choice([0,1])
            image, mask1, pos1, _ = get_line(image, direction=direction, pos=None, min_size=min_size, max_size=max_size, color_avail=color_avail)
            image, mask2, pos2, _ = get_line(image, direction=direction, pos=None, min_size=min_size, max_size=max_size, color_avail=color_avail)
            if (direction == 0 and pos1[0] == pos2[0]) or (direction == 1 and pos1[1] == pos2[1]):
                # The two lines cannot be on the same straight line:
                continue
        elif relation == "VerticalMid":
            direction = np.random.choice([0,1])
            image, mask1, pos1, _ = get_line(image, direction=direction, pos=None, min_size=max(3, min_size), max_size=max_size, color_avail=color_avail)
            pos_new = get_pos_new_mid(pos1, direction, min_size=min_size, canvas_size=canvas_size)
            if pos_new is None:
                continue
            image, mask2, pos2, _ = get_line(image, direction=None, pos=pos_new, min_size=min_size, max_size=max_size, color_avail=color_avail)
        elif relation == "VerticalEdge":
            direction = np.random.choice([0,1])
            image, mask1, pos1, _ = get_line(image, direction=direction, pos=None, min_size=max(3, min_size), max_size=max_size, color_avail=color_avail)
            pos_new = get_pos_new_edge(pos1, direction, min_size=min_size, canvas_size=canvas_size)
            if pos_new is None:
                continue
            image, mask2, pos2, _ = get_line(image, direction=None, pos=pos_new, min_size=min_size, max_size=max_size, color_avail=color_avail)
        elif relation == "VerticalSepa":
            direction = np.random.choice([0,1])
            image, mask1, pos1, _ = get_line(image, direction=direction, pos=None, min_size=min_size, max_size=max_size, color_avail=color_avail)
            pos_new = get_pos_new_not_touching(mask1, direction=1-direction, min_size=min_size, max_size=max_size, canvas_size=canvas_size, pos=pos1, min_distance=1)
            if pos_new is None:
                continue
            image, mask2, pos2, _ = get_line(image, direction=None, pos=pos_new, min_size=min_size, max_size=max_size, color_avail=color_avail)
        
        # Randomly permute the mask
        if np.random.choice([0,1]) == 1:
            mask1, mask2 = mask2, mask1
            pos1, pos2 = pos2, pos1

        info = {}
        info["id_object_mask"] = {0: mask1, 1: mask2}
        info["id_object_pos"] = {0: pos1, 1: pos2}
        info["obj_spec"] = [(("obj_0", "line"), "Attr"), (("obj_1", "line"), "Attr")]
        info["node_id_map"] = {"obj_0": 0, "obj_1": 1}
        obj_idx = 2

        # Add distractors with position and color:
        max_n_distractors = max_n_objs - min_n_objs
        if max_n_distractors > 0:
            n_distractors = np.random.randint(1, max_n_distractors+1)
            mask = (image != 0).float()
            for k in range(n_distractors):
                mask = (image != 0).float()
                pos_new = get_pos_new_not_touching(mask, direction=np.random.choice([0,1]), min_size=min_size, max_size=max_size, canvas_size=canvas_size, min_distance=0)
                image, obj_mask, obj_pos, _ = get_line(image, direction=None, pos=pos_new, color_avail=color_avail)
                # Update info
                info["id_object_mask"][obj_idx] = obj_mask
                info["id_object_pos"][obj_idx] = obj_pos
                info["obj_spec"].append(((f"obj_{obj_idx}", "line"), "Attr"))
                info["node_id_map"][f"obj_{obj_idx}"] = obj_idx
                obj_idx += 1
        
        # Get all relations
        relations = []
        n_objs = len(info["id_object_mask"])
        for id1 in range(n_objs):
            for id2 in range(id1 + 1, n_objs):
                rel = get_chosen_line_rel(info["id_object_pos"][id1], info["id_object_pos"][id2])
                relations.append((id1, id2, rel))
        info["relations"] = relations

        data.append(
            (to_one_hot(image)[0],
             (mask1, mask2),
             relation,
             Dictionary(info),
            )
        )
        if len(data) >= n_examples:
            break
        if isplot:
            visualize_matrices(image)
            p.print(relation)
            plot_matrices([mask1.squeeze(), mask2.squeeze()])
            print('\n\n')
    return data

def get_line(image, direction=None, pos=None, min_size=2, max_size=10, color_avail=[1,2]):
    """
    Direction: 0: horizontal; 1: vertical.
    """
    canvas_size = image.shape[-1]
    mask = torch.zeros(image.shape)
    if pos is None:
        assert direction is not None
        for i in range(10):
            if direction == -1:
                direction_core = np.random.choice([0,1])
            else:
                direction_core = direction
            if direction_core == 0:
                # horizontal:
                h = 1
                w = np.random.randint(min_size, max_size+1)
            elif direction_core == 1:
                h = np.random.randint(min_size, max_size+1)
                w = 1
            row_start = 0
            row_end = canvas_size - h
            if row_end <= row_start:
                continue
            row = np.random.randint(row_start, row_end+1)

            col_start = 0
            col_end = canvas_size - w
            if col_end <= col_start:
                continue
            col = np.random.randint(col_start, col_end+1)
            pos = (row, col, h, w)
    else:
        row, col, h, w = pos

    color = np.random.choice(color_avail)
    image[..., row: row+h, col: col+w] = color
    mask[..., row: row+h, col: col+w] = 1
    return image, mask, pos, direction


def get_pos_new_mid(pos, direction, min_size, canvas_size):
    pos_mid = (pos[0] + pos[2]//2, pos[1] + pos[3]//2)
    pos_new = None
    for k in range(10):
        orientation = np.random.choice([0,1])
        if direction == 1:
            # second line is horizontal:
            if orientation == 0:
                # second line is on the right:
                if canvas_size-pos_mid[1] <= min_size:
                    continue
                pos_new = (pos_mid[0], pos_mid[1]+1, 1, np.random.randint(min_size, canvas_size-pos_mid[1]+1))
            elif orientation == 1:
                # second line is on the left:
                if pos[1] <= min_size:
                    continue
                w_mid = np.random.randint(min_size, pos[1]+1)
                pos_new = (pos_mid[0], pos_mid[1] - w_mid, 1, w_mid)
        elif direction == 0:
            # second line is vertical:
            if orientation == 0:
                # second line is on the bottom:
                if canvas_size-pos_mid[0] <= min_size:
                    continue
                pos_new = (pos_mid[0]+1, pos_mid[1], np.random.randint(min_size, canvas_size-pos_mid[0]+1), 1)
            elif orientation == 1:
                # second line is on the top:
                if pos[0] <= min_size:
                    continue
                h_mid = np.random.randint(min_size, pos[0]+1)
                pos_new = (pos_mid[0] - h_mid, pos_mid[1], h_mid, 1)
        if pos_new is not None:
            break
    return pos_new


def get_pos_new_edge(pos, direction, min_size, canvas_size):
    pos_new = None
    for k in range(10):
        orientation = np.random.choice([0,1])
        edge_side = np.random.choice([0,1])
        if direction == 1:
            # second line is horizontal:
            if orientation == 0:
                # second line is on the right:
                if canvas_size-pos[1] <= min_size:
                    continue
                if edge_side == 0:
                    pos_new = (pos[0], pos[1]+1, 1, np.random.randint(min_size, canvas_size-pos[1]+1))
                elif edge_side == 1:
                    pos_new = (pos[0]+pos[2]-1, pos[1]+1, 1, np.random.randint(min_size, canvas_size-pos[1]+1))
            elif orientation == 1:
                # second line is on the left:
                if pos[1] <= min_size:
                    continue
                w_mid = np.random.randint(min_size, pos[1]+1)
                if edge_side == 0:
                    pos_new = (pos[0], pos[1] - w_mid, 1, w_mid)
                elif edge_side == 1:
                    pos_new = (pos[0]+pos[2]-1, pos[1] - w_mid, 1, w_mid)
        elif direction == 0:
            # second line is vertical:
            if orientation == 0:
                # second line is on the bottom:
                if canvas_size-pos[0] <= min_size:
                    continue
                if edge_side == 0:
                    pos_new = (pos[0]+1, pos[1], np.random.randint(min_size, canvas_size-pos[0]+1), 1)
                elif edge_side == 1:
                    pos_new = (pos[0]+1, pos[1]+pos[3]-1, np.random.randint(min_size, canvas_size-pos[0]+1), 1)
            elif orientation == 1:
                # second line is on the top:
                if pos[0] <= min_size:
                    continue
                h_mid = np.random.randint(min_size, pos[0]+1)
                if edge_side == 0:
                    pos_new = (pos[0] - h_mid, pos[1], h_mid, 1)
                elif edge_side == 1:
                    pos_new = (pos[0] - h_mid, pos[1]+pos[3]-1, h_mid, 1)
        if pos_new is not None:
            break
    return pos_new


def get_pos_new_not_touching(mask1, direction, min_size, max_size, canvas_size, pos=None, min_distance=0):
    mask = deepcopy(mask1)
    if min_distance > 0:
        mask[...,max(0,pos[0]-min_distance):pos[0]+pos[2]+min_distance, max(0, pos[1]-min_distance):pos[1]+pos[3]+min_distance] = 1
    for k in range(30):
        if direction == 0:
            # horizontal:
            h = 1
            w = np.random.randint(min_size, max_size+1)
        elif direction == 1:
            h = np.random.randint(min_size, max_size+1)
            w = 1
        row_start = 0
        row_end = canvas_size - h
        if row_end <= row_start:
            continue
        row = np.random.randint(row_start, row_end)

        col_start = 0
        col_end = canvas_size - w
        if col_end <= col_start:
            continue
        col = np.random.randint(col_start, col_end)

        mask2 = torch.zeros(mask1.shape)
        mask2[..., row: row+h, col: col+w] = 1
        if (mask + mask2 == 2).any():
            continue
        else:
            break
    pos_new = (row, col, h, w)
    return pos_new



def mask_to_obj(data):
    """Transform the data with format 'image+mask' to format 'image+obj'."""
    def get_obj_from_mask(img, mask_ele):
        """
        image: [C, H, W], where the first dimension in C is for 0.
        mask:  [1, H, W]
        """
        assert len(img.shape) == len(mask_ele.shape) == 3
        obj_ele = torch.cat([img[:1] * (1 - mask_ele), img[1:] * mask_ele], 0)
        return obj_ele
    if data[0][1] is not None:
        data_new = []
        for data_item in data:
            image, mask, c_repr, info = data_item
            if isinstance(image, tuple):
                assert len(image) == len(mask) == 2
                obj = (get_obj_from_mask(image[0], mask[0]), get_obj_from_mask(image[1], mask[1]))
            else:
                # Concept:
                assert len(mask) == 1
                obj = (get_obj_from_mask(image, mask[0]),)
            data_new.append((image, obj, c_repr, info))
        return data_new
    else:
        return data


def get_simple_1D_dataset(
    n_tasks=1000,
    noise_std=0.1,
    image_size=(64,),
    in_channels=2,
    shape_types=["rectangle", "triangle"],
    max_n_shapes=4,
    max_shape_height=2,
    max_shape_width=5,
    is_expand_2d=True,
    isplot=False,
):
    """Generate a dataset on a 1D image, where there can be multiple different shapes and colors."""
    import matplotlib.pylab as plt
    def is_no_collision(shape_info_example, min_distance):
        locs = np.array([ele["loc"] for ele in shape_info_example])
        distance_matrix = np.abs(locs[None] - locs[:,None])
        rows, cols = np.triu_indices(len(distance_matrix), 1)
        all_distance = distance_matrix[rows, cols]
        return (all_distance > min_distance).all()

    def render_shape(shape_info_example, image_size):
        x = torch.zeros(in_channels, *image_size)
        for shape_info in shape_info_example:
            x_ele = torch.zeros(in_channels, *image_size)
            if shape_info["shape_type"] == "rectangle":
                x_ele[:, shape_info["loc"]-shape_info["width"]: shape_info["loc"]+shape_info["width"]+1] = torch.FloatTensor(shape_info["color"][:, None]) * shape_info["height"]
            elif shape_info["shape_type"] == "triangle":
                x_ele[:, shape_info["loc"]-shape_info["width"]: shape_info["loc"]+shape_info["width"]+1] = torch.cat(
                    [torch.linspace(0, shape_info["height"], shape_info["width"]+1),
                     torch.linspace(shape_info["height"], 0, shape_info["width"]+1)[1:]])[None] * shape_info["color"][:,None]
            else:
                pass
            x = x + x_ele
        return x


    # First, sample number of shapes:
    n_shapes_list = np.random.randint(1, max_n_shapes+1, size=n_tasks*3)
    min_shape_height = max_shape_height * 0.25
    min_shape_width = np.maximum(2, max_shape_width * 0.25)
    max_shape_width = np.round(max_shape_width)

    task_dict = {}
    jj = 0
    for i in range(n_tasks * 3):
        shape_info_example = []

        n_shapes = n_shapes_list[i]
        for k in range(n_shapes):
            shape_info = {}
            # Second, sample the shape types:
            shape_info["shape_type"] = np.random.choice(shape_types)

            # Third, sample the features of each shape except position:
            shape_info["height"] = np.random.rand() *(max_shape_height - min_shape_height) + min_shape_height
            shape_info["width"] = np.random.randint(min_shape_width, max_shape_width+1)
            shape_info["color"] = np.random.rand(2) * 0.5 + 1

            # Fourth, sample the position of each shape:
            shape_info["loc"] = np.random.randint(np.round(max_shape_width), np.round(image_size[0] - max_shape_width))

            # Append to shape_info_example:
            shape_info_example.append(shape_info)

        if is_no_collision(shape_info_example, min_distance=max_shape_width*2):
            x_example = render_shape(shape_info_example, image_size)
            if noise_std > 0:
                x_example = x_example + torch.randn(x_example.shape) * noise_std
            if is_expand_2d:
                task_dict[str(jj)] = [(x_example[...,None], None, None, {"shape_info": shape_info_example})]
            jj += 1
        else:
            continue
        if len(task_dict) >= n_tasks:
            break
    if len(task_dict) < n_tasks:
        p.print("Generated {} tasks, less than the required {} tasks".format(len(task_dict), n_tasks))
    if isplot:
        for i in range(10):
            plt.plot(task_dict[str(i)][0][0].T)
            plt.show()
    return task_dict


def get_dataset(args, n_examples=None, isplot=False, is_load=False, is_rewrite=False, verbose=True, is_save_inter=False):
    """Generate the dataset according to specifications.

    Args:
        is_load: if True, will load the previously-saved file if there is one, and write to file if there isn't.
        is_rewrite: if False, will obey the behavior of is_load. If True, will re-generate the data, and save to file.
    """
    if n_examples is None:
        n_examples = args.n_examples
    args.is_mask = False
    seed = args.seed
    if args.dataset.startswith("y-") and not args.use_seed_2d:
        seed = args.seed_3d
    if ((args.dataset.startswith("c-") or args.dataset.startswith("y-")) and args.max_n_distractors != 2) or args.dataset.startswith("h-"):
        dataset_2d_param = "ex_{}_seed_{}_cav_{}_rain_{}_color_{}_distr_{}".format(
            n_examples, seed,
            args.canvas_size if hasattr(args, "canvas_size") else None,
            args.rainbow_prob if hasattr(args, "rainbow_prob") else None,
            args.color_avail if hasattr(args, "color_avail") else None,
            args.max_n_distractors)
    else:
        dataset_2d_param = "ex_{}_seed_{}_cav_{}_rain_{}_color_{}".format(
            n_examples, seed,
            args.canvas_size if hasattr(args, "canvas_size") else None,
            args.rainbow_prob if hasattr(args, "rainbow_prob") else None,
            args.color_avail if hasattr(args, "color_avail") else None)
    if args.min_n_distractors != 0:
        dataset_2d_param += "_mindistr_{}".format(args.min_n_distractors)
    if args.allow_connect is False:
        dataset_2d_param += "_connect_{}".format(args.allow_connect)

    dataset_filename = REA_PATH + "/data/{}-{}.p".format(args.dataset if not args.dataset.startswith("y-") else "c" + args.dataset[1:], dataset_2d_param)
    dataset = None
    if is_load and os.path.isfile(dataset_filename) and not is_rewrite:
        dataset, args_load = pload(dataset_filename)
        if (len(dataset[0]) > 3 and "id_object_mask" in dataset[0][3]) or not args.dataset.startswith("c-"):
            if verbose and not args.dataset.startswith("y-"):
                p.print("Dataset loaded from {}.".format(dataset_filename))
            args.n_classes = args_load.n_classes
            args.image_size = args_load.image_size
            args.in_channels = args_load.in_channels
            args.is_mask = args_load.is_mask
            if args.dataset.startswith("c-"):
                args.concept_collection = dataset.concept_collection
            if args.dataset.startswith("u-"):
                args.concept_collection = args_load.concept_collection
            if not args.dataset.startswith("y-"):
                return dataset, args
        else:
            if verbose:
                p.print("Dataset's info is an old version, regenerate.")
    if is_load and not is_rewrite and not os.path.isfile(dataset_filename):
        p.print("Do not find dataset {}.".format(dataset_filename))
    set_seed(seed)
    if args.dataset in ["cifar10"]:
        """Standard datasets"""
        dataset = datasets.CIFAR10(get_root_dir() + '/datasets/', download=True, transform=transforms.ToTensor())
        args.in_channels = dataset[0][0].shape[0]
        args.image_size = dataset[0][0].shape[-2:]
        args.concept_collection = None
        args.n_classes = len(dataset.classes)
    elif args.dataset.startswith("c-"):
        """BabyARC single concept/relation/operator dataset with ground-truth masks."""
        mode = args.dataset.split("-")[1]
        if "^" in mode:
            mode, focus_type = mode.split("^")
        else:
            focus_type = None
        dataset = ConceptDataset(
            mode=mode,
            n_examples=n_examples,
            canvas_size=args.canvas_size,
            rainbow_prob=args.rainbow_prob,
            w_type=args.w_type,
            max_n_distractors=args.max_n_distractors,
            min_n_distractors=args.min_n_distractors,
            color_avail=args.color_avail,
            allow_connect=args.allow_connect,
            parsing_check=args.parsing_check if hasattr(args, "parsing_check") else False,
            focus_type=focus_type,
            save_filename=dataset_filename[:-2] + "_inter.p" if is_save_inter else None,
        )
        args.in_channels = 10
        args.n_classes = 1
        if isinstance(dataset[0][0], tuple):
            args.image_size = dataset[0][0][0].shape[-2:]
        else:
            args.image_size = dataset[0][0].shape[-2:]
        args.concept_collection = dataset.concept_collection
        args.is_mask = True
    elif args.dataset.startswith("u-"):
        """
        args.dataset = "u-concept-Red+Green+Blue+Cube+Cylinder+Large+Small"
        """
        mode = args.dataset[2:]
        if mode.startswith("concept"):
            if set(mode.split("concept-")[1].split("-")) == set("Red+Green+Blue+Cube+Cylinder+Large+Small".split("-")):
                data_list_1 = pload("/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-saved/data_list_canvas_64_ex_25000_1.p")
                data_list_2 = pload("/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-saved/data_list_canvas_64_ex_30000_2.p")
                data_list_3 = pload("/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-saved/data_list_canvas_64_ex_11000_3.p")
                data_list = data_list_1 + data_list_2 + data_list_3 
            elif set(mode.split("concept-")[1].split("-")) == set("Red+Cube+Large".split("-")):
                data_list_1 = pload("/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-saved/data_list_canvas_Red+Cube+Large_64_ex_20000_1.p")
                data_list_2 = pload("/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-saved/data_list_canvas_Red+Cube+Large_64_ex_20000_2.p")
                data_list_3 = pload("/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-saved/data_list_canvas_Red+Cube+Large_64_ex_4000_3.p")
                data_list = data_list_1 + data_list_2 + data_list_3 
            else:
                raise
            # data_list = pload("/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-saved/data_list_canvas_64_ex_440.p")
        elif mode.startswith("relation"):
            if set(mode.split("relation-")[1].split("-")) == set("SameColor+SameShape+SameSize".split("-")):
                data_list_1 = pload("/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-saved/data_list_canvas_relation_64_ex_25000_1.p")
                data_list_2 = pload("/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-saved/data_list_canvas_relation_64_ex_30000_2.p")
                data_list_3 = pload("/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-saved/data_list_canvas_relation_64_ex_11000_3.p")
                data_list = data_list_1 + data_list_2 + data_list_3 
            else:
                raise
            # data_list = pload("/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-saved/data_list_canvas_relation_64_ex_440.p")
        elif mode.startswith("graph"):
            if set(mode.split("graph-")[1].split("-")) == set("Graph1+Graph2+Graph3".split("-")):
                data_list = pload("/dfs/user/tailin/.results/CLEVR_relation/clevr-concept-relation-saved/data_list_canvas_graph_Graph1+Graph2+Graph3_64_ex_200_4.p") # Need to fix
            else:
                raise
        else:
            raise
        data_list = data_list[:n_examples]
        dataset = ConceptClevrDataset(
            data=data_list,
        )
        args.concept_collection = mode.split("-")[1].split("+")
        args.in_channels = 3
        args.image_size = dataset[0][0].shape[-2:]
        args.n_classes = 1
        args.is_mask = True
    elif args.dataset.startswith("y-"):
        dataset_3d_param = "seed_{}_proc_{}_sz_{}_{}_color_{}_thxy_{}_{}_thz_{}_{}".format(
                            args.seed_3d, args.num_processes_3d, args.image_size_3d[0], args.image_size_3d[1],
                            args.color_map_3d[:4], args.add_thick_surf[0],  args.add_thick_surf[1],
                            args.add_thick_depth[0], args.add_thick_depth[1])
        dataset_filename = REA_PATH + "/data/{}-3d_{}_2d_{}.p".format(args.dataset, dataset_3d_param, dataset_2d_param)
        if os.path.isfile(dataset_filename) and not is_rewrite:
            dataset, args_load = pload(dataset_filename)
            if verbose:
                p.print("Dataset loaded from {}.".format(dataset_filename))
            args.n_classes = args_load.n_classes
            args.image_size = args_load.image_size
            args.in_channels = args_load.in_channels
            args.is_mask = args_load.is_mask
            args.concept_collection = args_load.concept_collection
            return dataset, args
        if dataset is None:
            args_2d = deepcopy(args)
            args_2d.seed = seed
            args_2d.dataset = "c" + args.dataset[1:]
            dataset, _ = get_dataset(args_2d, n_examples=n_examples, isplot=isplot, is_load=is_load, 
                                     is_rewrite=is_rewrite, verbose=verbose)
        args.concept_collection = dataset.concept_collection
        dataset = ConceptDataset3D(data=convert_babyarc(dataset, args))
        args.in_channels = 3
        args.image_size = args.image_size_3d
        args.n_classes = 1
        args.is_mask = True
    elif args.dataset.startswith("pc-") or args.dataset.startswith("pg-"):
        dataset = generate_fewshot_dataset(args, n_shot=1, n_queries_per_class=args.n_queries_per_class)
        args.in_channels = 10
        args.image_size = dataset[0][0][0][0].shape[-2:]
        args.n_classes = 1
        args.is_mask = True
    elif args.dataset.startswith("yc-"):
        dataset = generate_fewshot_dataset(args, n_shot=1, n_queries_per_class=args.n_queries_per_class)
        args.in_channels = 3
        args.image_size = dataset[0][1][0][0].shape[-2:]
        args.n_classes = 1
        args.is_mask = True
    elif args.dataset.startswith("h-"):
        """
        BabyARC composition dataset.

        args.dataset:
            h-c^(1,2):Eshape+Ashape-d^1:RandShape :
                only concept with concept_avail=["Eshape","Ashape"], n_concepts_range=(1,2), max_n_distractors=1, additional_shape="Randshape"
            h-r^2a+2ai+3a+3b+3bi:SameShape+SameColor(Line+Rect+RectSolid+Lshape)-d^1:Randshape :
                relation_structure="2a+2ai+3a+3b+3bi",
                relation_avail=["SameShape","SameColor"],
                concept_avail=["Line","Rect","RectSolid","Lshape"]
                additional_concepts=["Randshape"]
                max_n_distractors=1
        """
        mode_split = args.dataset[2:].split("-")
        settings = {}
        for content in mode_split:
            if content.startswith("c^"):
                settings["n_concepts_range"] = eval(content.split(":")[0][2:])
                settings["concept_avail"] = content.split(":")[1].split("+")
            elif content.startswith("r^"):
                settings["relation_structure"] = content.split(":")[0][2:]
                settings["relation_avail"] = content.split(":")[1].split("(")[0].split("+")
                settings["concept_avail"] = content.split(":")[1][:-1].split("(")[1].split("+")
            elif content.startswith("d^"):
                settings["max_n_distractors"] = eval(content.split(":")[0][2:])
                settings["additional_concepts"] = content.split(":")[1].split("+") if len(content.split(":")) > 1 else None
            else:
                raise
        if hasattr(args, "max_n_distractors"):
            settings["max_n_distractors"] = args.max_n_distractors
        if hasattr(args, "min_n_distractors"):
            settings["min_n_distractors"] = args.min_n_distractors
        dataset = ConceptCompositionDataset(
            canvas_size=args.canvas_size,
            n_examples=n_examples,
            rainbow_prob=args.rainbow_prob,
            color_avail=args.color_avail,
            concept_avail=settings["concept_avail"] if "concept_avail" in settings else None,
            relation_avail=settings["relation_avail"] if "relation_avail" in settings else None,
            relation_structure=settings["relation_structure"] if "relation_structure" in settings else "None",
            additional_concepts=settings["additional_concepts"] if "additional_concepts" in settings else None,
            n_concepts_range=settings["n_concepts_range"] if "n_concepts_range" in settings else 2,
            min_n_distractors=settings["min_n_distractors"] if "min_n_distractors" in settings else 0,
            max_n_distractors=settings["max_n_distractors"] if "max_n_distractors" in settings else 0,
            n_examples_per_task=5 if "c^" in args.dataset else 6,
        )
        if isinstance(dataset[0][0][0], tuple):
            args.image_size = dataset[0][0][0][0].shape[-2:]
        else:
            args.image_size = dataset[0][0][0].shape[-2:]
        args.in_channels = 10
        args.n_classes = 1
        args.is_mask = True
    elif args.dataset.startswith("arc"):
        """ARC image dataset"""
        dataset = ARCDataset(n_examples=n_examples, output_mode="energy")
        args.in_channels = 10
        args.n_classes = 1
        args.image_size = dataset[0][0].shape[-2:]
    elif args.dataset.startswith("v-"):
        """CLEVR dataset:"""
        parts = args.dataset.split("-")
        split_name = parts[1]
        start_idx = 0
        if len(parts) > 2:
            start_idx = int(parts[2])
        if "interpolate_mode" not in args.__dict__:
            interpolate_mode = "bilinear" if "train" in split_name else "nearest"
        else:
            interpolate_mode = args.interpolate_mode
        # Process the images the same way as slot attention. This 
        # rescales RGB values to the passed in range
        min_val, max_val = args.image_value_range.split(",") if "image_value_range" in args.__dict__ else args.selector_image_value_range.split(",")
        min_val = float(min_val)
        max_val = float(max_val)
        rgb_std = 1.0 / (max_val - min_val)
        rgb_mean = -(rgb_std * min_val)

        if split_name.startswith("CLEVR6") or split_name.startswith("CLEVR10"):
            processor = ClevrImagePreprocessor((128, 128), crop=(29, 221, 64, 256), rgb_mean=rgb_mean, rgb_std=rgb_std)
            dataset = CLEVR(root="/dfs/user/tailin/.results/CLEVR_concept/CLEVR_with_masks", 
                            split_name=split_name)
            i = start_idx
            n_tasks = n_examples if n_examples != None and n_examples > 0 else len(dataset)
            end_idx = min(start_idx + n_tasks, len(dataset))
            task_dict = {}
            while i < end_idx:
                img, info = dataset[i]
                img_name = info['image_name']
                obj_masks = info['mask']
                # Make sure to process image the same as in slot attention
                true_img = processor(img.unsqueeze(0), interpolate_mode=interpolate_mode).squeeze(0)
                task_dict[img_name] = [(true_img, None, None, {'obj_masks': obj_masks})]
                i += 1
            args.image_size = true_img.shape[-2:]
        elif split_name.startswith("CLEVRRelation"):
            resolution = (88, 128)
            crop = (32, 208, 32, 288)
            processor = ClevrImagePreprocessor(resolution, crop=crop, rgb_mean=rgb_mean, rgb_std=rgb_std)

            "split_name is 'CLEVRRelation:{dataset_split}' where dataset_split is either train, val, or test"
            dataset_split = split_name.split(":")[-1]
            train_set, val_set, test_set = create_easy_dataset(output_type="mask-only")
            dataset_dct = {"train": train_set, "val": val_set, "test": test_set}
            dataset = dataset_dct[dataset_split]
            
            i = start_idx
            n_tasks = n_examples if n_examples != None and n_examples > 0 else len(dataset)
            end_idx = min(start_idx + n_tasks, len(dataset))
            task_dict = {}
            while i < end_idx:
                task = dataset[i]
                task_dict[i] = []
                # Iterate through the task example pairs.
                for idx in range(5):
                    # Take the RGB channels
                    input_img = processor(task["inputs"][idx]["image"][:3].unsqueeze(0)).squeeze(0)
                    target_mask = task["outputs"][idx].unsqueeze(0)
                    target_mask = target_mask[..., crop[0]:crop[1], crop[2]:crop[3]] if crop else target_mask
                    target_mask = F.interpolate(target_mask.float(), resolution, mode="nearest").round().squeeze(0)
                    task_dict[i].append(((input_img, target_mask), None, None, {'obj_masks': task["outputs_mask_only"][idx]}))
                # Test example:
                test_img = processor(task["test_input"]["image"][:3].unsqueeze(0)).squeeze(0)
                test_mask = task["test_output"].unsqueeze(0)
                test_mask = test_mask[..., crop[0]:crop[1], crop[2]:crop[3]] if crop else test_mask
                test_mask = F.interpolate(test_mask.float(), resolution, mode="nearest").round().squeeze(0)
                task_dict[i].append(((test_img, test_mask), None, None, {}))

                i += 1
            args.image_size = target_mask.shape[-2:]
        else:
            raise NotImplementedError
        args.in_channels = 3
        args.n_classes = 1
        dataset = task_dict
    elif args.dataset.startswith("m-") or args.dataset.startswith("t-"):
        """Multi-Dsprites and Tetrominoes datasets"""
        
        parts = args.dataset.split("-")
        split_name = parts[1]
        if args.dataset.startswith("m-"):
            dataset = MultiDsprites(root="/dfs/user/tailin/.results/CLEVR_concept/multi_dsprites", split_name=split_name)
        else:
            dataset = Tetrominoes(root="/dfs/user/tailin/.results/CLEVR_concept/tetrominoes", split_name=split_name)
        # Process the images the same way as slot attention. This 
        # rescales RGB values to the passed in range
        min_val, max_val = args.image_value_range.split(",") if "image_value_range" in args.__dict__ else args.selector_image_value_range.split(",")
        min_val = float(min_val)
        max_val = float(max_val)
        rgb_std = 1.0 / (max_val - min_val)
        rgb_mean = -(rgb_std * min_val)
        args.image_size = dataset[0][0].shape[-2:]
        # Perform normalization, but not cropping / resizing
        processor = ClevrImagePreprocessor(args.image_size, rgb_mean=rgb_mean, rgb_std=rgb_std)    

        start_idx = 0
        if len(parts) > 2:
            start_idx = int(parts[2])
        n_tasks = n_examples if n_examples != None and n_examples > 0 else len(dataset)
        end_idx = min(start_idx + n_tasks, len(dataset))
        task_dict = {} 
        i = start_idx
        while i < end_idx:
            img, info = dataset[i]
            img_name = info['image_name']
            obj_masks = info['mask']
            # Make sure to process image the same as in slot attention
            true_img = processor(img.unsqueeze(0)).squeeze(0)
            task_dict[img_name] = [(true_img, None, None, {'obj_masks': obj_masks})]
            i += 1
        args.in_channels = 3
        args.n_classes = 1
        dataset = task_dict
    elif args.dataset.startswith("s-"):
        """Simple 1D dataset:"""
        n_tasks = n_examples if n_examples is not None else 1000
        max_n_shapes_str, max_n_shapes, width_str, width, noise_std_str, noise_std = args.dataset.split("-")[1:]
        assert max_n_shapes_str == "nsh"
        assert width_str == "width"
        assert noise_std_str == "noise"
        dataset = get_simple_1D_dataset(
            n_tasks=n_tasks,
            noise_std=eval(noise_std),
            image_size=(args.canvas_size,),
            in_channels=2,
            shape_types=["rectangle", "triangle"],
            max_n_shapes=eval(max_n_shapes),
            max_shape_height=2,
            max_shape_width=eval(width),
            isplot=isplot,
        )
        args.in_channels = 2
        args.n_classes = 1
        args.image_size = dataset["0"][0][0].shape[2:]
    else:
        raise Exception("dataset '{}' is not recognized!".format(args.dataset))
    if is_rewrite or is_load:
        pdump((dataset, args), dataset_filename)
        if verbose:
            p.print("dataset saved at {}.".format(dataset_filename))
    return dataset, args


# In[ ]:


def generate_discriminative_task(
    concepts,
    n_examples_per_concept,
    n_examples_per_task,
    n_tasks,
    canvas_size=8,
    isplot=False,
):
    data_dict = {}
    for key in concepts:
        args = init_args({
            "dataset": "c-{}".format(key),
            "seed": 1,
            "n_examples": n_examples_per_concept,
            "canvas_size": canvas_size,
            "rainbow_prob": 0.,
            "color_avail": "-1",
            "w_type": "mask",
            "max_n_distractors": 0,
        })
        dataset, args = get_dataset(args, is_load=True)
        data_dict[key] = dataset.data

    # Generate task:
    task_list = []
    for i in range(n_tasks):
        # randomly sample a concept:
        concept_chosen = np.random.choice(concepts)
        example_ids = np.random.choice(len(data_dict[concept_chosen]), replace=False, size=n_examples_per_task+1)
        concept_examples = torch.stack([data_dict[concept_chosen][i][0] for i in example_ids[:-1]])

        concept_others = deepcopy(concepts)
        concept_others.remove(concept_chosen)
        other_ids_all = list(itertools.product(concept_others, range(n_examples_per_concept)))
        other_ids = np.random.choice(len(other_ids_all), replace=False, size=n_examples_per_task+1)
        other_keys = [other_ids_all[id] for id in other_ids]
        other_examples = torch.stack([data_dict[other_key][id][0] for other_key, id in other_keys[:-1]])

        is_concept = np.random.choice([True, False])
        if is_concept:
            test_example = data_dict[concept_chosen][example_ids[-1]][0][None]
        else:
            test_example = data_dict[other_keys[-1][0]][other_keys[-1][1]][0][None]
        task_list.append(((concept_examples, other_examples), (test_example, is_concept)))

    if isplot:
        p.print("concept: {}".format(concept_chosen))
        visualize_matrices(concept_examples.argmax(1), use_color_dict=True)
        visualize_matrices(other_examples.argmax(1), use_color_dict=True)
        p.print("is_concept: {}".format(is_concept))
        visualize_matrices(test_example.argmax(1), use_color_dict=True)
    return task_list


# ### 1.5 Datasets for paper:

# In[ ]:


is_load_example_dataset = False


# #### 1.5.0 CLEVR-Concept:

# ##### CLEVR-Concept, training:

# In[ ]:


if is_load_example_dataset:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
    from zeroc.train import get_dataset, ConceptDataset, ConceptClevrDataset
    from zeroc.concept_library.util import init_args

    args = init_args({
        "dataset": "u-concept-Red+Cube+Large",
        "seed": 3,
        "n_examples": 66000,
        "color_avail": "1,2",
        "canvas_size": 64,
        "min_n_distractors": 0,
        "allow_connect": True,
        "rainbow_prob": 0.0,
    })
    concept_dataset, _ = get_dataset(args, is_load=True)
    # concept_dataset.draw(range(100))


# In[ ]:


# concept_dataset.draw(range(40,80))


# In[ ]:


# # Red
# data_c1 = concept_dataset[5]
# visualize_matrices([data_c1[0]], use_color_dict=False, filename="clevr_img/c1_img.pdf")
# plot_matrices(data_c1[1][0], images_per_row=6, no_xlabel=True, filename="clevr_img/c1_mask.pdf")


# In[ ]:


# # Cube
# data_c2 = concept_dataset[54]
# visualize_matrices([data_c2[0]], use_color_dict=False, filename="clevr_img/c2_img.pdf")
# plot_matrices(data_c2[1][0], images_per_row=6, no_xlabel=True, filename="clevr_img/c2_mask.pdf")


# In[ ]:


# # Large
# data_c3 = concept_dataset[59]
# visualize_matrices([data_c3[0]], use_color_dict=False, filename="clevr_img/c3_img.pdf")
# plot_matrices(data_c3[1][0], images_per_row=6, no_xlabel=True, filename="clevr_img/c3_mask.pdf")


# In[ ]:


# relation_dataset.draw(range(40))


# In[ ]:


# relation_dataset.draw(range(40, 80))


# In[ ]:


# # SameColor
# data_r1 = relation_dataset[30]
# visualize_matrices([data_r1[0]], use_color_dict=False, filename="clevr_img/r1_img.pdf")
# plot_matrices(data_r1[1][0], images_per_row=6, no_xlabel=True, filename="clevr_img/r1_mask_1.pdf")
# plot_matrices(data_r1[1][1], images_per_row=6, no_xlabel=True, filename="clevr_img/r1_mask_2.pdf")


# In[ ]:


# # SameShape:
# data_r2 = relation_dataset[24]
# visualize_matrices([data_r2[0]], use_color_dict=False, filename="clevr_img/r2_img.pdf")
# plot_matrices(data_r2[1][0], images_per_row=6, no_xlabel=True, filename="clevr_img/r2_mask_1.pdf")
# plot_matrices(data_r2[1][1], images_per_row=6, no_xlabel=True, filename="clevr_img/r2_mask_2.pdf")


# In[ ]:


# # SameSize:
# data_r3 = relation_dataset[49]
# visualize_matrices([data_r3[0]], use_color_dict=False, filename="clevr_img/r3_img.pdf")
# plot_matrices(data_r3[1][0], images_per_row=6, no_xlabel=True, filename="clevr_img/r3_mask_1.pdf")
# plot_matrices(data_r3[1][1], images_per_row=6, no_xlabel=True, filename="clevr_img/r3_mask_2.pdf")


# In[ ]:





# In[ ]:


# # Graph2:
# data_g1 = graph_dataset[33]
# visualize_matrices([data_g1[0]], use_color_dict=False, filename="clevr_img/g1_img.pdf")
# # plot_matrices(data_g2[1][0], images_per_row=6, no_xlabel=True, filename="clevr_img/g2_mask_1.pdf")
# # plot_matrices(data_g2[1][1], images_per_row=6, no_xlabel=True, filename="clevr_img/g2_mask_2.pdf")


# In[ ]:


# # Graph2:
# data_g2 = graph_dataset[12]
# visualize_matrices([data_g2[0]], use_color_dict=False, filename="clevr_img/g2_img.pdf")
# # plot_matrices(data_g2[1][0], images_per_row=6, no_xlabel=True, filename="clevr_img/g2_mask_1.pdf")
# # plot_matrices(data_g2[1][1], images_per_row=6, no_xlabel=True, filename="clevr_img/g2_mask_2.pdf")


# In[ ]:


# # Graph3:
# data_g3 = graph_dataset[77]
# visualize_matrices([data_g3[0]], use_color_dict=False, filename="clevr_img/g3_img.pdf")
# # plot_matrices(data_g2[1][0], images_per_row=6, no_xlabel=True, filename="clevr_img/g2_mask_1.pdf")
# # plot_matrices(data_g2[1][1], images_per_row=6, no_xlabel=True, filename="clevr_img/g2_mask_2.pdf")


# ##### CLEVR-Relation, training:

# In[ ]:


if is_load_example_dataset:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
    from zeroc.train import get_dataset, ConceptDataset, ConceptClevrDataset
    from zeroc.concept_library.util import init_args

    args = init_args({
        "dataset": "u-relation-SameColor+SameShape+SameSize",
        "seed": 3,
        "n_examples": 66000,
        "color_avail": "1,2",
        "canvas_size": 64,
        "min_n_distractors": 0,
        "allow_connect": True,
        "rainbow_prob": 0.0,
    })
    relation_dataset, _ = get_dataset(args, is_load=True)
    # relation_dataset.draw(range(100))


# ##### CLEVR-graph, inference:

# In[ ]:


if is_load_example_dataset:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
    from zeroc.train import get_dataset, ConceptDataset, ConceptClevrDataset
    from zeroc.concept_library.util import init_args

    args = init_args({
        "dataset": "u-graph-Graph1+Graph2+Graph3",
        "seed": 2,
        "n_examples": 400,
        "canvas_size": 32,
        "rainbow_prob": 0.,
        "w_type": "image+mask",
        "color_avail": "1,2",
        "min_n_distractors": 0,
        "max_n_distractors": 0,
        "allow_connect": True,
        "parsing_check": False,
    })
    graph_dataset, _ = get_dataset(args, is_load=True)
    # dataset.draw(range(100))


# #### 1.5.1 HD-Letter:

# ##### HD-Letter, training concept:

# In[ ]:


if is_load_example_dataset:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
    from zeroc.train import get_dataset, ConceptDataset, ConceptClevrDataset
    from zeroc.concept_library.util import init_args

    args = init_args({
        "dataset": "c-Line",
        "seed": 1,
        "n_examples": 44000,
        "canvas_size": 16,
        "rainbow_prob": 0.,
        "w_type": "image+mask",
        "color_avail": "1,2",
        "max_n_distractors": 2,
        "min_n_distractors": 0,
        "allow_connect": True,
        "parsing_check": False,
    })
    dataset, _ = get_dataset(args, is_load=True)
    dataset.draw(range(100))


# ##### HD-Letter, training relation:

# In[ ]:


if is_load_example_dataset:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
    from zeroc.train import get_dataset, ConceptDataset, ConceptClevrDataset
    from zeroc.concept_library.util import init_args

    args = init_args({
        "dataset": "c-Parallel+VerticalMid+VerticalEdge",
        "seed": 1,
        "n_examples": 44000,
        "canvas_size": 16,
        "rainbow_prob": 0.,
        "w_type": "image+mask",
        "color_avail": "1,2",
        "max_n_distractors": 3,
        "min_n_distractors": 0,
        "allow_connect": True,
        "parsing_check": False,
    })
    dataset, _ = get_dataset(args, is_load=True)
    dataset.draw(range(100))


# ##### HD-Letter, classification at inference:

# In[ ]:


if is_load_example_dataset:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
    from zeroc.train import get_dataset, ConceptDataset, ConceptClevrDataset
    from zeroc.concept_library.util import init_args

    args = init_args({
        "dataset": "c-Eshape+Fshape+Ashape",
        "seed": 2,
        "n_examples": 400,
        "canvas_size": 16,
        "rainbow_prob": 0.,
        "w_type": "image+mask",
        "color_avail": "1,2",
        "max_n_distractors": 0,
        "min_n_distractors": 0,
        "allow_connect": True,
        "parsing_check": False,
    })
    dataset, _ = get_dataset(args, is_load=True)
    dataset.draw(range(100))


# ##### HD-Letter, detection at inference:

# ##### Eshape:

# In[ ]:


# Eshape:
if is_load_example_dataset:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
    from zeroc.train import get_dataset, ConceptDataset, ConceptClevrDataset
    from zeroc.concept_library.util import init_args

    args = init_args({
        "dataset": "c-Eshape[6,8]+Cshape+Lshape+Tshape+Rect+RectSolid^Eshape[6,8]",
        "seed": 2,
        "n_examples": 400,
        "canvas_size": 16,
        "rainbow_prob": 0.,
        "w_type": "image+mask",
        "color_avail": "1,2",
        "max_n_distractors": 2,
        "min_n_distractors": 1,
        "allow_connect": False,
        "parsing_check": True,
    })
    dataset, _ = get_dataset(args, is_load=True)
    dataset.draw(range(100))


# ##### Fshape:

# In[ ]:


# Fshape:
if is_load_example_dataset:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
    from zeroc.train import get_dataset, ConceptDataset, ConceptClevrDataset
    from zeroc.concept_library.util import init_args

    args = init_args({
        "dataset": "c-Fshape[6,8]+Cshape+Lshape+Tshape+Rect+RectSolid^Fshape[6,8]",
        "seed": 2,
        "n_examples": 400,
        "canvas_size": 16,
        "rainbow_prob": 0.,
        "w_type": "image+mask",
        "color_avail": "1,2",
        "max_n_distractors": 2,
        "min_n_distractors": 1,
        "allow_connect": False,
        "parsing_check": True,
    })
    dataset, _ = get_dataset(args, is_load=True)
    dataset.draw(range(100))


# ##### Ashape:

# In[ ]:


# Ashape:
if is_load_example_dataset:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
    from zeroc.train import get_dataset, ConceptDataset, ConceptClevrDataset
    from zeroc.concept_library.util import init_args

    args = init_args({
        "dataset": "c-Fshape[6,8]+Cshape+Lshape+Tshape+Rect+RectSolid^Fshape[6,8]",
        "seed": 2,
        "n_examples": 400,
        "canvas_size": 16,
        "rainbow_prob": 0.,
        "w_type": "image+mask",
        "color_avail": "1,2",
        "max_n_distractors": 2,
        "min_n_distractors": 1,
        "allow_connect": False,
        "parsing_check": True,
    })
    dataset, _ = get_dataset(args, is_load=True)
    dataset.draw(range(100))


# #### 1.5.2 HD-Concept:

# ##### HD-Concept, training concept:

# In[ ]:


if is_load_example_dataset:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
    from zeroc.train import get_dataset, ConceptDataset, ConceptClevrDataset
    from zeroc.concept_library.util import init_args

    args = init_args({
        "dataset": "c-Rect[4,16]+Eshape[3,10]",
        "seed": 1,
        "n_examples": 44000,
        "canvas_size": 20,
        "rainbow_prob": 0.,
        "w_type": "image+mask",
        "color_avail": "1,2",
        "max_n_distractors": 2,
        "min_n_distractors": 0,
        "allow_connect": True,
        "parsing_check": False,
    })
    dataset, _ = get_dataset(args, is_load=True)
    dataset.draw(range(100))


# ##### HD-Concept, training relation:

# In[ ]:


if is_load_example_dataset:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
    # BabyARC-fewshot dataset for classification:
    from zeroc.train import get_dataset, ConceptDataset, ConceptClevrDataset
    from zeroc.concept_library.util import init_args

    args = init_args({
        "dataset": "c-IsNonOverlapXY+IsInside+IsEnclosed(Rect[4,16]+Randshape[3,8]+Lshape[3,10]+Tshape[3,10])",
        "seed": 1,
        "n_examples": 44000,
        "canvas_size": 20,
        "rainbow_prob": 0.,
        "w_type": "image+mask",
        "color_avail": "1,2",
        "max_n_distractors": 1,
        "min_n_distractors": 0,
        "allow_connect": True,
        "parsing_check": False,
    })
    dataset, _ = get_dataset(args, is_load=True)
    dataset.draw(range(100))


# ##### HD-Concept, classification at inference:

# In[ ]:


if is_load_example_dataset:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
    from zeroc.train import get_dataset, ConceptDataset, ConceptClevrDataset
    from zeroc.concept_library.util import init_args

    args = init_args({
        "dataset": "c-RectE1a+RectE2a+RectE3a",
        "seed": 2,
        "n_examples": 200,
        "canvas_size": 20,
        "rainbow_prob": 0.,
        "w_type": "image+mask",
        "color_avail": "1,2",
        "max_n_distractors": 0,
        "min_n_distractors": 0,
        "allow_connect": True,
        "parsing_check": False,
    })
    dataset, _ = get_dataset(args, is_load=True)
    dataset.draw(range(100))


# ##### HD-Concept, detection at inference:

# ##### RectE1a:

# In[ ]:


if is_load_example_dataset:
# RectE1a:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
    from zeroc.train import get_dataset, ConceptDataset, ConceptClevrDataset
    from zeroc.concept_library.util import init_args

    args = init_args({
        "dataset": "c-RectE1a+Cshape+Lshape+Tshape+Rect+RectSolid^RectE1a",
        "seed": 2,
        "n_examples": 200,
        "canvas_size": 20,
        "rainbow_prob": 0.,
        "w_type": "image+mask",
        "color_avail": "1,2",
        "max_n_distractors": 1,
        "min_n_distractors": 1,
        "allow_connect": False,
        "parsing_check": True,
    })
    dataset, _ = get_dataset(args, is_load=True)
    dataset.draw(range(100))


# ##### RectE2a:

# In[ ]:


# RectE2a:
if is_load_example_dataset:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
    from zeroc.train import get_dataset, ConceptDataset, ConceptClevrDataset
    from zeroc.concept_library.util import init_args

    args = init_args({
        "dataset": "c-RectE2a+Cshape+Lshape+Tshape+Rect+RectSolid^RectE2a",
        "seed": 2,
        "n_examples": 200,
        "canvas_size": 20,
        "rainbow_prob": 0.,
        "w_type": "image+mask",
        "color_avail": "1,2",
        "max_n_distractors": 1,
        "min_n_distractors": 1,
        "allow_connect": False,
        "parsing_check": True,
    })
    dataset, _ = get_dataset(args, is_load=True)
    dataset.draw(range(100))


# ##### RectE3a:

# In[ ]:


# RectE3a:
if is_load_example_dataset:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
    from zeroc.train import get_dataset, ConceptDataset, ConceptClevrDataset
    from zeroc.concept_library.util import init_args

    args = init_args({
        "dataset": "c-RectE3a+Cshape+Lshape+Tshape+Rect+RectSolid^RectE3a",
        "seed": 2,
        "n_examples": 200,
        "canvas_size": 20,
        "rainbow_prob": 0.,
        "w_type": "image+mask",
        "color_avail": "1,2",
        "max_n_distractors": 1,
        "min_n_distractors": 1,
        "allow_connect": True,
        "parsing_check": True,
    })
    dataset, _ = get_dataset(args, is_load=True)
    dataset.draw(range(100))


# #### 1.5.3 Acquiring concepts between domains:

# In[ ]:


# RectE3a:
if is_load_example_dataset:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
    sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
    from zeroc.train import get_dataset, ConceptDataset, ConceptClevrDataset
    from zeroc.concept_library.util import init_args

    args = init_args({
        "dataset": "yc-Eshape[5,9]+Fshape[5,9]+Ashape[5,9]",
        "seed_3d": 42,
        "n_examples": 200,
        "num_processes_3d": 5,
        "n_queries_per_class": 1,
        # 2D examples
        "seed": 102,
        "use_seed_2d": True,
        "canvas_size": 16,
        "rainbow_prob": 0.,
        "w_type": "image+mask",
        "color_avail": "1,2",
        "max_n_distractors": 0,
        "min_n_distractors": 0,
        "parsing_check": False,
        "allow_connect": True,
    })
    dataset, _ = get_dataset(args, is_load=True)
    dataset.draw(range(100))


# ### 1.6 Dataset test:

# In[ ]:


# Test get_chosen_rel
def test_line_rel(pos1, pos2):
    image = torch.zeros(1, 16, 16)
    image, mask1, pos1, _ = get_line(image, direction=None, pos=pos1, min_size=3, max_size=None, color_avail=[1])
    image, mask2, pos2, _ = get_line(image, direction=None, pos=pos2, min_size=3, max_size=None, color_avail=[2])
    print(image)
    print(get_chosen_line_rel(pos1, pos2)) # Vertical mid
    print()

# test_line_rel((9, 4, 1, 7), (0, 3, 9, 1))

# # Rotated T-shape tests
# test_line_rel((9, 4, 1, 7), (1, 3, 9, 1)) # Test bottom edge
# test_line_rel((8, 4, 1, 7), (1, 3, 9, 1))
# test_line_rel((1, 4, 1, 7), (1, 3, 9, 1)) # Test upper edge
# test_line_rel((2, 4, 1, 7), (1, 3, 9, 1))

# test_line_rel((9, 1, 1, 5), (1, 6, 9, 1)) # Test bottom edge
# test_line_rel((8, 1, 1, 5), (1, 6, 9, 1)) 
# test_line_rel((1, 1, 1, 5), (1, 6, 9, 1)) 
# test_line_rel((2, 1, 1, 5), (1, 6, 9, 1)) 


# # T-shape tests
# test_line_rel((2, 9, 6, 1), (1, 2, 1, 8)) # Test right edge
# test_line_rel((2, 8, 6, 1), (1, 2, 1, 8))
# test_line_rel((2, 2, 6, 1), (1, 2, 1, 8)) # Test left edge
# test_line_rel((2, 3, 6, 1), (1, 2, 1, 8))
# test_line_rel((2, 5, 6, 1), (1, 2, 1, 8)) # Test middle

# test_line_rel((1, 9, 6, 1), (7, 2, 1, 8)) # Test right edge
# test_line_rel((1, 8, 6, 1), (7, 2, 1, 8)) 
# test_line_rel((1, 2, 6, 1), (7, 2, 1, 8)) 
# test_line_rel((1, 3, 6, 1), (7, 2, 1, 8)) 


# In[ ]:


# import sys, os
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
# # BabyARC-fewshot dataset for classification:
# from zeroc.concept_library.train import get_dataset, ConceptDataset, ConceptFewshotDataset
# from zeroc.concept_library.util import init_args

# args = init_args({
#     "dataset": "c-RectE3a+Eshape+Rect+Tshape+Fshape+Ashape^RectE3a",
#     "seed": 2,
#     "n_examples": 200,
#     "canvas_size": 20,
#     "rainbow_prob": 0.,
#     "w_type": "image+mask",
#     "color_avail": "1,2",
#     "max_n_distractors": 1,
#     "min_n_distractors": 1,
#     "allow_connect": True,
#     "parsing_check": False,
# })
# dataset, _ = get_dataset(args, is_load=True, is_save_inter=True)
# dataset.draw(range(40))


# In[ ]:


# import sys, os
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
# # BabyARC-fewshot dataset for classification:
# from zeroc.concept_library.train import get_dataset, ConceptDataset, ConceptFewshotDataset
# from zeroc.concept_library.util import init_args

# args = init_args({
#     "dataset": "c-RectE3a+Cshape[2,5]+Lshape[2,5]+Tshape[2,5]+Rect[2,5]+RectSolid[2,5]^RectE3a",
#     "seed": 2,
#     "n_examples": 200,
#     "canvas_size": 20,
#     "rainbow_prob": 0.,
#     "w_type": "image+mask",
#     "color_avail": "1,2",
#     "max_n_distractors": 1,
#     "min_n_distractors": 1,
#     "allow_connect": True,
#     "parsing_check": False,
# })
# dataset, _ = get_dataset(args, is_load=True, is_save_inter=True)
# dataset.draw(range(40))


# In[ ]:


# import sys, os
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
# # BabyARC-fewshot dataset for classification:
# from zeroc.concept_library.train import get_dataset, ConceptDataset, ConceptFewshotDataset
# from zeroc.concept_library.util import init_args

# args = init_args({
#     "dataset": "c-Cshape[2,5]+Lshape[2,5]+Tshape[2,5]+Rect[2,5]+RectSolid[2,5]^Cshape[2,5]",
#     "seed": 2,
#     "n_examples": 200,
#     "canvas_size": 20,
#     "rainbow_prob": 0.,
#     "w_type": "image+mask",
#     "color_avail": "1,2",
#     "max_n_distractors": 1,
#     "min_n_distractors": 1,
#     "allow_connect": False,
#     "parsing_check": True,
# })
# dataset, _ = get_dataset(args, is_load=True, is_rewrite=True, is_save_inter=True)
# dataset.draw(range(40))


# In[ ]:


# # BabyARC-relation dataset:
# relation_args = init_args({
#     "dataset": "c-IsNonOverlapXY+IsInside+IsEnclosed(Rect[4,16]+Randshape[3,8]+Lshape[3,10]+Tshape[3,10])",
#     "seed": 1,
#     "n_examples": 44000,
#     "canvas_size": 20,
#     "rainbow_prob": 0.,
#     "color_avail": "1,2",
#     "w_type": "image+mask",
#     "max_n_distractors": 1,
#     "min_n_distractors": 0,
#     "allow_connect": True,
#     "parsing_check": True,
# })
# relation_dataset, args = get_dataset(relation_args, is_load=True)
# relation_dataset.draw(range(40))


# In[ ]:


# from zeroc.concepts_shapes import SameShape, SameColor, SameAll, SameRow, SameCol, SubsetOf, IsInside, IsNonOverlapXY
# for data in relation_dataset:
#     pos_id = data[2]
#     info = data[3]
#     relations = get_arc_relations(data[0], info)
#     relations = [ele[2] for ele in relations]
#     print(pos_id, relations)
#     print(data[2])
#     plot_matrices([data[1][0].squeeze(), data[1][1].squeeze()])
#         # raise


# In[ ]:


# concept_args = init_args({
#     "dataset": "c-Parallel+VerticalMid+VerticalEdge",
#     "seed": 1,
#     "n_examples": 44000,
#     "canvas_size": 16,
#     "rainbow_prob": 0.,
#     "color_avail": "1,2",
#     "w_type": "image+mask",
#     "max_n_distractors": 3,
#     "min_n_distractors": 0,
#     "allow_connect": True,
# })
# concept_dataset, args = get_dataset(concept_args, is_load=True, is_rewrite=True)


# In[ ]:


# import sys, os
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
# # BabyARC-fewshot dataset for classification:
# from zeroc.concept_library.train import get_dataset, ConceptDataset, ConceptFewshotDataset
# from zeroc.concept_library.util import init_args

# args = init_args({
#     "dataset": "c-Eshape+Ashape^Eshape",
#     "seed": 2,
#     "n_examples": 40,
#     "canvas_size": 16,
#     "rainbow_prob": 0.,
#     "w_type": "image+mask",
#     "color_avail": "1,2",
#     "max_n_distractors": 1,
#     "min_n_distractors": 1,
#     "allow_connect": True,
#     "parsing_check": True,
# })
# dataset, _ = get_dataset(args, is_load=True, is_rewrite=True)
# dataset.draw(range(40))


# In[ ]:


# import sys, os
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
# # BabyARC-fewshot dataset for classification:
# from zeroc.concept_library.train import get_dataset, ConceptDataset, ConceptFewshotDataset
# from zeroc.concept_library.util import init_args

# args = init_args({
#     "dataset": "c-IsNonOverlapXY+IsInside+IsEnclosed(Rect[4,16]+Randshape[3,8]+Lshape[3,10]+Tshape[3,10])",
#     "seed": 2,
#     "n_examples": 400,
#     "canvas_size": 16,
#     "rainbow_prob": 0.,
#     "w_type": "image+mask",
#     "color_avail": "1,2",
#     "min_n_distractors": 0,
#     "max_n_distractors": 1,
#     "parsing_check": True,
#     "allow_connect": True,
# })
# dataset, _ = get_dataset(args, is_load=True)
# dataset.draw(range(40))


# In[ ]:


# import sys, os
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
# # BabyARC-fewshot dataset for classification:
# from zeroc.concept_library.train import get_dataset, ConceptDataset, ConceptFewshotDataset
# from zeroc.concept_library.util import init_args

# args = init_args({
#     "dataset": "c-RectE1a+Lshape^RectE1a",
#     "seed": 2,
#     "n_examples": 40,
#     "canvas_size": 16,
#     "rainbow_prob": 0.,
#     "w_type": "image+mask",
#     "color_avail": "1,2",
#     "min_n_distractors": 1,
#     "max_n_distractors": 1,
#     "parsing_check": True,
# })
# dataset, _ = get_dataset(args, is_load=False)
# dataset.draw(range(40))


# In[ ]:


# import sys, os
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
# # BabyARC-fewshot dataset for classification:
# from zeroc.concept_library.train import get_dataset, ConceptDataset, ConceptFewshotDataset
# from zeroc.concept_library.util import init_args

# args = init_args({
#     "dataset": "pc-RectE1b+RectE1c+RectE2b+RectE2c+RectE3b+RectE3c",
#     "seed": 2,
#     "n_examples": 100,
#     "canvas_size": 16,
#     "rainbow_prob": 0.,
#     "w_type": "image+mask",
#     "color_avail": "1,2",
#     "max_n_distractors": 0,
#     "parsing_check": True,
# })
# dataset, _ = get_dataset(args, is_load=True)
# dataset.draw(range(10))


# In[ ]:


# import sys, os
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
# # BabyARC-fewshot dataset for classification:
# from zeroc.concept_library.train import get_dataset, ConceptDataset, ConceptFewshotDataset
# from zeroc.concept_library.util import init_args

# args = init_args({
#     "dataset": "pc-RectE1a+RectE2a+RectE3a",
#     "seed": 2,
#     "n_examples": 400,
#     "canvas_size": 16,
#     "rainbow_prob": 0.,
#     "w_type": "image+mask",
#     "color_avail": "1,2",
#     "max_n_distractors": 0,
#     "parsing_check": True,
# })
# dataset, _ = get_dataset(args, is_load=True, is_rewrite=True)
# dataset.draw(range(10))


# In[ ]:


# # import sys, os
# # sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
# # sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
# # # BabyARC-fewshot dataset for classification:
# from zeroc.concept_library.train import get_dataset, ConceptDataset, ConceptFewshotDataset
# from zeroc.concept_library.util import init_args

# args = init_args({
#     "dataset": "c-Rect[4,16]+Eshape[3,10]",
#     "seed": 1,
#     "n_examples": 44000,
#     "canvas_size": 20,
#     "rainbow_prob": 0.,
#     "w_type": "image+mask",
#     "color_avail": "1,2",
#     "max_n_distractors": 2,
#     "min_n_distractors": 0,
#     "allow_connect": True,
#     "parsing_check": False,
# })
# dataset, _ = get_dataset(args, is_load=True)
# dataset.draw(range(40))


# In[ ]:


# # import sys, os
# # sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
# # sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
# # # BabyARC-fewshot dataset for classification:
# from zeroc.concept_library.train import get_dataset, ConceptDataset, ConceptFewshotDataset
# from zeroc.concept_library.util import init_args

# args = init_args({
#     "dataset": "c-IsNonOverlapXY+IsInside+IsEnclosed(Rect[4,16]+Randshape[3,8]+Lshape[3,10]+Tshape[3,10])",
#     "seed": 1,
#     "n_examples": 44000,
#     "canvas_size": 20,
#     "rainbow_prob": 0.,
#     "w_type": "image+mask",
#     "color_avail": "1,2",
#     "max_n_distractors": 1,
#     "min_n_distractors": 0,
#     "allow_connect": True,
#     "parsing_check": False,
# })
# dataset, _ = get_dataset(args, is_load=True)
# dataset.draw(range(40))


# In[ ]:


# import sys, os
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
# # BabyARC-fewshot dataset for classification:
# from zeroc.concept_library.train import get_dataset, ConceptDataset, ConceptFewshotDataset
# from zeroc.concept_library.util import init_args

# args = init_args({
#     "dataset": "c-Rect[4,15]+Eshape[5,12]",
#     "seed": 2,
#     "n_examples": 40,
#     "canvas_size": 16,
#     "rainbow_prob": 0.,
#     "w_type": "image+mask",
#     "color_avail": "1,2",
#     "max_n_distractors": 1,
#     "parsing_check": True,
# })
# dataset, _ = get_dataset(args, is_load=True)
# dataset.draw(range(40))


# In[ ]:


# import sys, os
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
# # BabyARC-fewshot dataset for classification:
# from zeroc.concept_library.train import get_dataset, ConceptDataset, ConceptFewshotDataset
# from zeroc.concept_library.util import init_args

# args = init_args({
#     "dataset": "pc-Cshape+Eshape+Fshape+Ashape+Hshape+Rect",
#     "seed": 102,
#     "n_examples": 400,
#     "canvas_size": 16,
#     "rainbow_prob": 0.,
#     "w_type": "image+mask",
#     "color_avail": "1,2",
#     "max_n_distractors": 0,
#     "parsing_check": False,
# })
# dataset, _ = get_dataset(args, is_load=True, is_rewrite=True)
# dataset.draw(range(10))


# In[ ]:


# import sys, os
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
# # BabyARC-fewshot dataset for classification:
# from zeroc.concept_library.train import get_dataset, ConceptDataset, ConceptFewshotDataset
# from zeroc.concept_library.util import init_args

# args = init_args({
#     "dataset": "pg-Cshape+Lshape+Tshape+Rect^Eshape+Fshape+Ashape",
#     "seed": 12,
#     "n_examples": 400,
#     "canvas_size": 16,
#     "rainbow_prob": 0.,
#     "w_type": "image+mask",
#     "color_avail": "1,2",
#     "max_n_distractors": 0,
#     "parsing_check": False,
# })
# dataset, _ = get_dataset(args, is_load=True, is_rewrite=False)
# dataset.draw(range(10))


# In[ ]:


# import sys, os
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
# # BabyARC-fewshot dataset for classification:
# from zeroc.concept_library.train import get_dataset, ConceptDataset, ConceptFewshotDataset
# from zeroc.concept_library.util import init_args

# args = init_args({
#     "dataset": "yc-Eshape[5,9]+Fshape[5,9]+Ashape[5,9]",
#     "seed_3d": 42,
#     "n_examples": 200,
#     "num_processes_3d": 5,
#     "n_queries_per_class": 1,
#     # 2D examples
#     "seed": 102,
#     "use_seed_2d": True,
#     "canvas_size": 16,
#     "rainbow_prob": 0.,
#     "w_type": "image+mask",
#     "color_avail": "1,2",
#     "max_n_distractors": 0,
#     "min_n_distractors": 0,
#     "parsing_check": False,
#     "allow_connect": True,
# })
# dataset, _ = get_dataset(args, is_load=True)
# dataset.draw(range(10))


# In[ ]:


# composite_args = init_args({
#     "dataset": "c-Eshape+Cshape+Lshape+Tshape+Rect+RectSolid^Eshape",
#     "seed": 2,
#     "n_examples": 400,
#     "canvas_size": 16,
#     "rainbow_prob": 0.,
#     "w_type": "image+mask",
#     "color_avail": "1,2",
#     "max_n_distractors": 4,
#     "parsing_check": True,
# })
# dataset, composite_args = get_dataset(composite_args, is_rewrite=True, is_load=True)


# In[ ]:


# # BabyARC-relation dataset, 3D:
# relation_args = init_args({
#     "dataset": "y-Parallel+VerticalMid+VerticalEdge",
#     "seed": 2,
#     "n_examples": 44000,
#     "canvas_size": 16,
#     "rainbow_prob": 0.,
#     "color_avail": "1,2",
#     "w_type": "image+mask",
#     "color_map_3d": "same",
#     "add_thick_surf": (0, 0.5),
#     "add_thick_depth": (0, 0.5),
#     "max_n_distractors": 2,
#     "seed_3d": 42,
#     "num_processes_3d": 20,
#     "image_size_3d": (256,256),
# })
# relation_dataset, args = get_dataset(relation_args, is_rewrite=False, is_load=True)
# relation_dataset.draw(range(40))


# In[ ]:


# # import sys, os
# # sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
# # sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
# # # BabyARC-concept dataset:
# from zeroc.concept_library.train import get_dataset, ConceptDataset, ConceptFewshotDataset
# from zeroc.concept_library.util import init_args
# concept_args = init_args({
#     "dataset": "c-Line",
#     "seed": 1,
#     "n_examples": 44000,
#     "canvas_size": 16,
#     "rainbow_prob": 0.,
#     "color_avail": "1,2",
#     "w_type": "image+mask",
#     "max_n_distractors": 0,
# })
# concept_dataset, args = get_dataset(concept_args, is_load=True)


# In[ ]:


# import sys, os
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
# # BabyARC-concept dataset:
# from zeroc.concept_library.train import get_dataset, ConceptDataset
# from zeroc.concept_library.util import init_args
# concept_args = init_args({
#     "dataset": "c-Line",
#     "seed": 1,
#     "n_examples": 44000,
#     "canvas_size": 8,
#     "rainbow_prob": 0.,
#     "color_avail": "1,2",
#     "w_type": "image+mask",
#     "max_n_distractors": 2,
# })
# concept_dataset, args = get_dataset(concept_args, is_load=True)

# BabyARC-relation dataset:
# relation_args = init_args({
#     "dataset": "c-Parallel+Vertical",
#     "seed": 1,
#     "n_examples": 44000,
#     "canvas_size": 16,
#     "rainbow_prob": 0.,
#     "color_avail": "1,2",
#     "w_type": "image+mask",
#     "max_n_distractors": 2,
# })
# relation_dataset, args = get_dataset(relation_args, is_rewrite=True, is_load=True)


# In[ ]:


# import sys, os
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
# sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
# # BabyARC-concept dataset:
# from zeroc.concept_library.train import get_dataset, ConceptDataset
# from zeroc.concept_library.util import init_args
# concept_args = init_args({
#     "dataset": "c-Line+Lshape+Rect+RectSolid",
#     "seed": 1,
#     "n_examples": 44000,
#     "canvas_size": 16,
#     "rainbow_prob": 0.,
#     "color_avail": "1,2",
#     "w_type": "image+mask",
#     "max_n_distractors": 2,
# })
# concept_dataset, args = get_dataset(concept_args, is_load=False)

# # BabyARC-relation dataset:
# relation_args = init_args({
#     "dataset": "c-SameShape+SameColor+IsInside(Line+Lshape+Rect+RectSolid)",
#     "seed": 1,
#     "n_examples": 44000,
#     "canvas_size": 16,
#     "rainbow_prob": 0.,
#     "color_avail": "1,2",
#     "w_type": "image+mask",
#     "max_n_distractors": 2,
# })
# relation_dataset, args = get_dataset(relation_args, is_load=True)


# In[ ]:


# if __name__ == "__main__":
#     args = init_args({
#         "dataset": "c-Eshape+Fshape+Ashape",
#         "seed": 1,
#         "n_examples":100,
#         "canvas_size": 16,
#         "rainbow_prob": 0.,
#         "color_avail": "1,2",
#         "w_type": "mask",
#         "max_n_distractors": 0,
#     })
#     dataset, args = get_dataset(args, is_load=True)


# In[ ]:


# if __name__ == "__main__":
#     args = init_args({
#         "dataset": "c-Line+Lshape+Rect+RectSolid",
#         "seed": 1,
#         "n_examples": 20,
#         "canvas_size": 8,
#         "rainbow_prob": 0.,
#         "color_avail": "-1",
#         "w_type": "mask",
#         "max_n_distractors": 0,
#     })
#     dataset, args = get_dataset(args, is_load=True)


# In[ ]:


# # Get dataset:
# concepts = ["Lshape", "Tshape", "Eshape", "Hshape", "Cshape", "Ashape", "Fshape"]
# n_examples_per_concept = 10
# n_examples_per_task = 5
# n_tasks = 4

# task_list = generate_discriminative_task(
#     concepts,
#     n_examples_per_concept,
#     n_examples_per_task,
#     n_tasks,
#     isplot=True,
# )


# In[ ]:


# if __name__ == "__main__":
#     # CLEVR-relation dataset easy:
#     args = init_args({
#         "dataset": "v-CLEVRRelation:train",
#         "seed": 1,
#         "n_examples": 3000,
#         "canvas_size": 8,
#         "rainbow_prob": 0.,
#         "color_avail": "-1",
#         "selector_image_value_range": "-1,1",
#     })
#     dataset, args = get_dataset(args, is_load=False, is_rewrite=True)
#     for i in range(50):
#         visualize_matrices([(dataset[i][0][0][0] + 1)/2], use_color_dict=False)
#         plot_matrices(dataset[i][0][0][1], images_per_row=5)


# In[ ]:


# if __name__ == "__main__":
#     # BabyARC-relation dataset easy:
#     args = init_args({
#         "dataset": "h-r^2ai+2a+3ai+3a+3b:SameShape+SameColor(Line+Rect+RectSolid+Lshape)-d^1:Line+Randshape",
#         "seed": 1,
#         "n_examples": 40,
#         "canvas_size": 16,
#         "rainbow_prob": 0.,
#         "color_avail": "-1",
#         "max_n_distractors": 2,
#         "min_n_distractors": 1,
#         "allow_connect": False,
#     })
#     dataset, args = get_dataset(args, is_load=True)


# In[ ]:


# if __name__ == "__main__":
#     # BabyARC-relation dataset easy validation:
#     args = init_args({
#         "dataset": "h-r^2ai+2a+3ai+3a+3b:SameShape+SameColor(Line+Rect+RectSolid+Lshape)",
#         "seed": 1,
#         "n_examples": 3000,
#         "canvas_size": 16,
#         "rainbow_prob": 0.,
#         "color_avail": "1,2",
#     })
#     dataset, args = get_dataset(args, is_load=False, is_rewrite=True)


# In[ ]:


# if __name__ == "__main__":
#     # BabyARC-relation dataset easy validation:
#     args = init_args({
#         "dataset": "h-r^2ai+2a+3ai+3a+3b+4ai+4a+4b:SameShape+SameColor(Line+Rect+RectSolid+Lshape)",
#         "seed": 1,
#         "n_examples": 3000,
#         "canvas_size": 16,
#         "rainbow_prob": 0.,
#         "color_avail": "1,2",
#     })
#     dataset, args = get_dataset(args, is_load=False, is_rewrite=True)


# In[ ]:


# if __name__ == "__main__":
#     args = init_args({
#         "seed": 1,
#         "dataset": "s-nsh-4-noise-0.01",
#         "n_examples": 1000,
#     })
#     dataset = get_dataset(args, isplot=True, is_load=True)


# In[ ]:


# if __name__ == "__main__":
#     args = init_args({
#         "dataset": "h-c^(1,2):Eshape+Ashape-d^1:Randshape",
#         "seed": 1,
#         "n_examples": 10,
#         "canvas_size": 8,
#         "rainbow_prob": 0.,
#         "color_avail": "-1",
#     })
#     dataset, args = get_dataset(args, is_load=True)


# In[ ]:


# if __name__ == "__main__":
#     # BabyARC-concept dataset:
#     from zeroc.concept_library.train import get_dataset, ConceptDataset
#     from zeroc.concept_library.util import init_args
#     args = init_args({
#         "dataset": "h-c^(1,2):Line+Lshape+Rect+RectSolid",
#         "seed": 1,
#         "n_examples": 40000,
#         "canvas_size": 8,
#         "rainbow_prob": 0.,
#         "color_avail": "1,2",
#     })
#     dataset, args = get_dataset(args, is_load=True)


# In[ ]:


# if __name__ == "__main__":
#     # BabyARC-concept dataset:
#     args = init_args({
#         "dataset": "h-c^(1,2):Line+Lshape+Rect+RectSolid",
#         "seed": 1,
#         "n_examples": 40000,
#         "canvas_size": 8,
#         "rainbow_prob": 0.,
#         "color_avail": "1,2",
#     })
#     dataset, args = get_dataset(args, is_load=True)


# In[ ]:


# if __name__ == "__main__":
#     args = get_pdict()(
#         # dataset="c-Image",
# #         dataset="c-Line+RectSolid",
# #         dataset="c-IsInside(ARCshape)",
#         dataset="c-SameAll+SameShape+SameColor+SameRow+SameCol+IsInside(Line+Rect+RectSolid+Lshape+Randshape+ARCshape)",
#         # dataset="c-arc^Line+RectSolid+Rect",
#         canvas_size=8, n_examples=100, rainbow_prob=0,
#         max_n_distractors=-1,
#         color_avail="-1",
#     )
#     dataset, args = get_dataset(args)
# #     dataset.draw(range(10))


# In[ ]:


# if __name__ == "__main__":
#     args = get_pdict()(
#         dataset="c-RotateA+RotateB+RotateC+hFlip+vFlip+DiagFlipA+DiagFlipB+Move(Line+Rect+RectSolid+Lshape+Randshape+ARCshape)",
#         canvas_size=8, n_examples=100, rainbow_prob=0,
#         max_n_distractors=-1,
#         color_avail="2",
#     )
#     dataset, args = get_dataset(args)
#     dataset.draw(range(100))


# ## 2. Helper functions:

# ### 2.1 Data Transformations:

# In[ ]:


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4])
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indices = np.where(m.any(dim=0))[0]
        vertical_indices = np.where(m.any(dim=1))[0]
        if horizontal_indices.shape[0]:
            x1, x2 = horizontal_indices[[0, -1]]
            y1, y2 = vertical_indices[[0, -1]]
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([x1, y1, x2, y2])
    return boxes.astype(np.float32)


class BabyARC2DPad(nn.Module):
    """ 
    Pad 10 channel BabyARC images by padding the 0 channel with one's and the rest
    with zero's
    """
    def __init__(self, pad_size):
        super(BabyARC2DPad, self).__init__()
        self.pad_size = pad_size

    def forward(self, img):
        assert img.shape[-3] == 10
        one_pad = nn.ConstantPad2d(self.pad_size, 1)
        zero_pad = nn.ConstantPad2d(self.pad_size, 0)
        zero_channel = one_pad(img[0:1, :, :])
        remainder = zero_pad(img[1:, :, :])
        new_img = torch.cat((zero_channel, remainder))
        return new_img


class PermuteChannels(nn.Module):
    """
    Permutes all channels following some starting channel
    """
    def __init__(self, start_channel, num_channels):
        super(PermuteChannels, self).__init__()
        self.start_channel = start_channel
        self.num_channels = num_channels
    
    def forward(self, sample):
        # Assume sample is a tuple of (negative data, positive mask)
        perm = torch.randperm(self.num_channels - self.start_channel)
        perm = perm + self.start_channel
        indices = torch.cat((torch.tensor(list(range(0, self.start_channel))) , perm))
        permuted = sample[0][0][indices, ...]
        new_sample = ((permuted, sample[0][1], sample[0][2], sample[0][3]), sample[1])
        return new_sample


class BabyARC2DColorTransform(PermuteChannels):
    """
    Permutes all channels following some starting channel
    """
    def __init__(self):
        super().__init__(start_channel=1, num_channels=10)
        
        
class BabyARC3DColorTransform(nn.Module):
    """
    Performs random color jitter and random grayscale
    """
    def __init__(self, s=0.5):
        # s is the strength of color distortion.
        super().__init__()
        self.s = s
        
    def get_color_distortion(self):
        color_jitter = transforms.ColorJitter(0.8*self.s, 0.8*self.s, 0.8*self.s, 0.4*self.s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([
            rnd_color_jitter,
            rnd_gray])
        return color_distort
    
    def forward(self, sample):
        color_transform = self.get_color_distortion()
        new_img = color_transform(sample[0][0])
        new_sample = ((new_img, sample[0][1], sample[0][2], sample[0][3]), sample[1])
        return new_sample


class GenericRandomResizedCrop(nn.Module):
    """
    Crops an area with fixed aspect ratio that fully includes 
    ground truth mask
    """
    def __init__(self, size, pad, interpolation=InterpolationMode.NEAREST):
        super().__init__()
        self.size = size # Size of the result, which is equivalent to the canvas_size
        self.pad = pad
        self.mask_pad = nn.ConstantPad2d(size // 2, 0)
        self.interpolation = interpolation
        self.MAX_ITER = 2
    
    def __call__(self, sample):
        # Assume sample is a tuple of (negative data, positive mask)
        neg_data, pos_mask = sample
        half_canvas = int(neg_data[0].shape[-2] / 2)
        assert self.size == max(*neg_data[0].shape[-2:])
        # Important: get the bounding box of the masks in the image
        total_mask = pos_mask[0] if len(pos_mask) == 1 else pos_mask[0] + pos_mask[1]
        # Top left, bottom right corners
        y1, x1, y2, x2 = extract_bboxes(self.mask_pad(total_mask.squeeze()).unsqueeze(-1))[0].astype(np.int32)
        box_h = x2 - x1
        box_w = y2 - y1
        min_crop_size = max(box_h, box_w)
        mask_sums = torch.tensor([0] * len(pos_mask))
        i = 0
        # Make sure both positive masks are non-empty
        while torch.any(mask_sums == 0) and i < self.MAX_ITER:
            # Randomly sample crop sizes
            crop_size = torch.randint(low=min_crop_size, high=min_crop_size + half_canvas, size=()).item()
            # Sample the top left corner. Make sure that the crop area is within the padded canvas
            bottom = neg_data[0].shape[-2] + 2 * half_canvas
            right = neg_data[0].shape[-1] + 2 * half_canvas
            x1_sample = torch.randint(low=max(x1 - (crop_size - box_h ), 0), high=min(x1+1, bottom - crop_size + 1), size=()).item()
            y1_sample = torch.randint(low=max(y1 - (crop_size - box_w ), 0), high=min(y1+1, right - crop_size + 1), size=()).item()
            assert x1_sample >= 0 and x1_sample + crop_size <= bottom                 and y1_sample >= 0 and y1_sample + crop_size <= right
            # Performing cropping 
            new_img = F_tr.resized_crop(self.pad(neg_data[0]), x1_sample, y1_sample, crop_size, crop_size, self.size, self.interpolation)
            assert new_img.shape[-2:] == neg_data[0].shape[-2:]
            new_neg_masks = [F_tr.resized_crop(self.mask_pad(mask), x1_sample, y1_sample, crop_size, crop_size, self.size, self.interpolation) for mask in neg_data[1]]
            new_pos_mask = [F_tr.resized_crop(self.mask_pad(mask), x1_sample, y1_sample, crop_size, crop_size, self.size, self.interpolation) for mask in pos_mask] 
            new_sample = ((new_img, tuple(new_neg_masks), neg_data[2], neg_data[3]), new_pos_mask)
            mask_sums = torch.tensor([torch.sum(mask) for mask in new_pos_mask])
            i += 1
        if torch.any(mask_sums == 0):
            return sample
        return new_sample


class BabyARC2DRandomResizedCrop(GenericRandomResizedCrop):
    """
    Crops an area with fixed aspect ratio that fully includes 
    ground truth mask
    """
    def __init__(self, size):
        if not isinstance(size, int):
            size = size[-1]
        super().__init__(size=size, pad=BabyARC2DPad(size // 2))


class BabyARC3DRandomResizedCrop(GenericRandomResizedCrop):
    """
    Crops an area with fixed aspect ratio that fully includes 
    ground truth mask
    """
    def __init__(self, size):
        super().__init__(size=size, pad=nn.ConstantPad2d(size // 2, 0))


class GenericRandomFlip(nn.Module):
    """flip the given image randomly (horizontally or vertically) with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions
    """

    def __init__(self):
        super().__init__()

    def forward(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped (Identity, hFlip, vFlip).

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        neg_data, pos_mask = sample
        funcs = [
            nn.Identity(),
            F_tr.hflip,
            F_tr.vflip,
        ]
        func = np.random.choice(funcs)
        new_neg_data = (func(neg_data[0]),
                        tuple([func(mask) for mask in neg_data[1]]), neg_data[2], neg_data[3])
        new_pos_mask = tuple([func(mask) for mask in pos_mask])
        return (new_neg_data, new_pos_mask)


class GenericRandomRotate(nn.Module):
    """Randomly rotate with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions
    """

    def __init__(self):
        super().__init__()

    def forward(self, sample):
        """
        Args:
            img (PIL Image or Tensor): Image to be rotated (Identity, RotateA, RotateB, RotateC).

        Returns:
            PIL Image or Tensor: Randomly flipped image.
        """
        neg_data, pos_mask = sample
        funcs = [
            nn.Identity(),
            partial(F_tr.rotate, angle=90),
            partial(F_tr.rotate, angle=180),
            partial(F_tr.rotate, angle=270),
        ]
        func = np.random.choice(funcs)
        new_neg_data = (func(neg_data[0]),
                        tuple([func(mask) for mask in neg_data[1]]), neg_data[2], neg_data[3])
        new_pos_mask = tuple([func(mask) for mask in pos_mask])
        return (new_neg_data, new_pos_mask)


class AddRandomImgPixels(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, sample):
        neg_data, pos_mask = sample
        neg_img, neg_mask, neg_id, neg_info = neg_data
        assert not isinstance(neg_img, tuple) and not isinstance(neg_img, list)
        assert neg_img.shape[-3] == 10
        neg_img_argmax = neg_img.argmax(-3, keepdims=True)
        remainder = (neg_img_argmax == 0)  # [1, H, W]
        p = (neg_img_argmax!=0).float().mean().item()  # non_empty probability
        p = torch.rand(1)[0] * 1.5 * p + 0.5 * p
        size = (self.size, self.size) if isinstance(self.size, Number) else self.size 
        new_neg_img = torch.randint(10, size=(1, *size))  # [B, H, W]
        mask_pixels = ((torch.rand(1, *size) < p) & remainder).float()
        pixels = new_neg_img * mask_pixels
        new_neg_img = neg_img_argmax + pixels
        new_neg_img = to_one_hot(new_neg_img[0], n_channels=10)
        new_neg_data = (new_neg_img, neg_mask, neg_id, neg_info)
        return (new_neg_data, pos_mask)


class AddRandomImgPatch(nn.Module):
    def __init__(self, size, color_avail):
        """Add random image patch to the place where there is no mask."""
        super().__init__()
        self.size = size
        self.color_avail = color_avail

    def forward(self, sample):
        neg_data, pos_mask = sample
        neg_img, neg_mask, neg_id, neg_info = neg_data
        neg_img = deepcopy(neg_img)
        assert not isinstance(neg_img, tuple) and not isinstance(neg_img, list)
        assert neg_img.shape[-3] == 10
        """
        1. First, sample one or two patches
        2. Randomly permute the color, and rotate the patch
        3. Randomly put into a place with no mask.
        """
        neg_img_argmax = neg_img.argmax(-3)
        canvas_h, canvas_w = neg_img.shape[-2:]
        num = np.random.choice([1,2])
        for _ in range(num):
            patch0, pos0 = shrink(neg_img_argmax.float())
            row0, col0, h0, w0 = pos0
            # Sample a section of the patch:
            h = torch.randint(low=1, high=max(2, h0+1), size=(1,))[0]
            w = torch.randint(low=1, high=max(2, w0+1), size=(1,))[0]
            row = torch.randint(low=0, high=max(1,h0-h+1), size=(1,))[0]
            col = torch.randint(low=0, high=max(1,w0-w+1), size=(1,))[0]
            patch = patch0[row:row+h, col:col+w]
            # Sample direction:
            rand_dir = np.random.choice([0,1])
            if rand_dir == 1:
                patch = torch.rot90(patch, dims=[0,1])
            # Permute color:
            patch = to_one_hot(patch)
            if self.color_avail == "-1":
                color_avail = "1,2,3,4,5,6,7,8,9"
            else:
                color_avail = self.color_avail
            n_colors = len(color_avail.split(","))
            permute_idx = torch.randperm(n_colors) + 1
            permute_idx = torch.cat([torch.tensor([0]), permute_idx, torch.arange(n_colors+1, 10)])
            patch = patch[permute_idx]
            # Randomly elongate the image:
            h1, w1 = patch.shape[-2:]
            is_elongate = np.random.choice([0,1])
            if is_elongate == 1:
                elongate_ratio = np.random.rand() + 1
                if h1 > w1:
                    h1_new = min(canvas_h, int(np.round(h1 * elongate_ratio)))
                    patch = F.interpolate(patch[None], size=(h1_new, w1), mode="nearest")[0]
                elif w1 > h1:
                    w1_new = min(canvas_w, int(np.round(w1 * elongate_ratio)))
                    patch = F.interpolate(patch[None], size=(h1, w1_new), mode="nearest")[0]

            # Add to the current image:
            h1, w1 = patch.shape[-2:]
            for j in range(10):
                row1 = torch.randint(low=0, high=max(1, canvas_h-h1+1), size=(1,))[0]
                col1 = torch.randint(low=0, high=max(1, canvas_w-w1+1), size=(1,))[0]
                if neg_img_argmax[row1:row1+h1, col1:col1+w1].sum() == 0:
                    neg_img[:, row1:row1+h1, col1:col1+w1] = patch
                    break
                else:
                    continue
            neg_img_argmax = neg_img.argmax(-3)
        new_neg_data = (neg_img, neg_mask, neg_id, neg_info)
        return new_neg_data, pos_mask


def get_augment(transforms_str, canvas_size, is_rgb=False, color_avail=None):
    transforms_lst = []
    assert ":" not in transforms_str
    if "color" in transforms_str.split("+"):
        transforms_lst.append(BabyARC2DColorTransform() if not is_rgb else BabyARC3DColorTransform())
    if "flip" in transforms_str.split("+"):
        transforms_lst.append(GenericRandomFlip())
    if "rotate" in transforms_str.split("+"):
        transforms_lst.append(GenericRandomRotate())
    if "resize" in transforms_str.split("+"):
        transforms_lst.append(BabyARC2DRandomResizedCrop(canvas_size) if not is_rgb else BabyARC3DRandomResizedCrop(canvas_size))
    if "rand" in transforms_str.split("+"):
        transforms_lst.append(AddRandomImgPixels(canvas_size))
    if "randpatch" in transforms_str.split("+"):
        transforms_lst.append(AddRandomImgPatch(canvas_size, color_avail=color_avail))
    return transforms.Compose(transforms_lst)


def transform_pos_data(pos_data, transforms_str, color_avail):
    """Transforms a batch of positive data, a tuple of (img, masks, label, info)
    Args:
        pos_data: (img, tuple(masks), pos_ids, pos_infos)
            img: has shape of [B, C, H, W]
            tuple(masks) is a tuple of masks, each of which has shape of [B, 1, H, W]
            pos_ids and pos_infos will remain as they are.
        transforms_str: has format of e.g. "color+flip+rotate+resize:0.7", which means that
            the img and masks will have 0.7 probability of making the transform 
            (and 0.3 probability of remaining at the current status), and when transform
            happens, will peform transforms of color, flip, rotate, resize.

    Returns:
        transformed_pos_data: (transformed_img, tuple(transformed_masks), pos_ids, pos_infos)
    """
    transforms_str_split = transforms_str.split(":")
    if len(transforms_str_split) == 1:
        p_transforms = 1
        transforms_str_core = transforms_str
    else:
        p_transforms = eval(transforms_str_split[1])
        transforms_str_core = transforms_str_split[0]
    if np.random.rand() > p_transforms:
        return pos_data
    pos_imgs, pos_masks, pos_ids, pos_infos = pos_data
    is_rgb = True if pos_imgs.shape[1] == 3 else False
    canvas_size = pos_imgs.shape[-2:]
    augment = get_augment(transforms_str_core, canvas_size, is_rgb=is_rgb, color_avail=color_avail)
    transformed_img = torch.zeros(pos_imgs.shape)
    transformed_masks = [torch.zeros(mask.shape) for mask in pos_masks] 
    for idx in range(pos_imgs.shape[0]):
        ex_mask = tuple(mask[idx] for mask in pos_masks)
        transformed_tup, new_pos_masks = augment(((pos_imgs[idx], ex_mask, None, None), ex_mask))
        transformed_img[idx, ...] = transformed_tup[0]
        for mask_idx in range(len(transformed_masks)):
            transformed_masks[mask_idx][idx, ...] = new_pos_masks[mask_idx]
    return (transformed_img, tuple(transformed_masks), pos_ids, pos_infos)


def rescale_data(pos_data, rescaled_size, rescale_mode="nearest"):
    """Rescale the data to some given size.

    Args:
        rescaled_size: Choose from "None", or e.g. "16,16" (size of (16,16))
    """
    if rescaled_size == "None":
        return pos_data
    assert len(rescaled_size.split(",")) == 2
    rescaled_size = eval(rescaled_size)
    pos_imgs, pos_masks, pos_ids, pos_infos = pos_data
    if isinstance(pos_imgs, tuple):
        assert len(pos_imgs) == 2
        pos_imgs = (F.interpolate(pos_imgs[0], size=rescaled_size, mode=rescale_mode),
                    F.interpolate(pos_imgs[1], size=rescaled_size, mode=rescale_mode))
    else:
        pos_imgs = F.interpolate(pos_imgs, size=rescaled_size, mode=rescale_mode)
    pos_masks = tuple(F.interpolate(mask_ele, size=rescaled_size, mode="nearest") for mask_ele in pos_masks)
    return pos_imgs, pos_masks, pos_ids, pos_infos


def rescale_tensor(tensor, rescaled_size):
    """Rescale the tensor to some given size.

    Args:
        rescaled_size: Choose from "None", or e.g. "16,16" (size of (16,16))
    """
    if rescaled_size == "None":
        return tensor
    assert len(rescaled_size.split(",")) == 2
    rescaled_size = eval(rescaled_size)
    if len(tensor.shape) == 3:
        tensor = tensor[None]
        is_size_3 = True
    else:
        is_size_3 = False
    tensor = F.interpolate(tensor, size=rescaled_size, mode="nearest")
    if is_size_3:
        tensor = tensor[0]
    return tensor


# In[ ]:


# if __name__ == "__main__":
#     args = init_args({
#         "dataset": "c-Eshape+Cshape+Fshape+Tshape^Fshape",
#         "seed": 1,
#         "n_examples":40,
#         "canvas_size": 16,
#         "rainbow_prob": 0.,
#         "color_avail": "1,2",
#         "w_type": "mask",
#         "max_n_distractors": 2,
#     })
#     dataset, args = get_dataset(args, is_load=True)
#     # Test color permutation
#     permute_transform = BabyARC2DColorTransform()
#     example = (dataset[0][0], (torch.zeros(1, 16, 16),), dataset[0][2], dataset[0][3])
#     new_example, pos_mask = permute_transform((example, dataset[0][1]))
#     visualize_matrices([dataset[0][0].argmax(0)])
#     visualize_matrices([new_example[0].argmax(0)])
#     visualize_matrices(torch.cat(pos_mask))
    
#     # Test Hflip
#     flip_transform = GenericRandomFlip()
#     example = (dataset[0][0], (torch.zeros(1, 16, 16),), dataset[0][2], dataset[0][3])
#     new_example, pos_mask = flip_transform((example, dataset[0][1]))
#     visualize_matrices([dataset[0][0].argmax(0)])
#     visualize_matrices([new_example[0].argmax(0)])
#     visualize_matrices(new_example[1][0])
#     visualize_matrices(torch.cat(pos_mask))

    
#     # Test ResizeCrop
#     crop_transform = BabyARC2DRandomResizedCrop(size=dataset[0][0].shape[-1])
#     example = (dataset[0][0], (torch.zeros(1, 16, 16),), dataset[0][2], dataset[0][3])
#     new_example, pos_mask = crop_transform((example, dataset[0][1]))
#     visualize_matrices([dataset[0][0].argmax(0)])
#     visualize_matrices([new_example[0].argmax(0)])
#     visualize_matrices(new_example[1][0])
#     visualize_matrices(torch.cat(pos_mask))
    
#     # Test composition
#     composed = transforms.Compose([BabyARC2DColorTransform(), GenericRandomFlip(), BabyARC2DRandomResizedCrop(size=dataset[0][0].shape[-1])])
#     example = (dataset[0][0], (torch.zeros(1, 16, 16),), dataset[0][2], dataset[0][3])
#     new_example, pos_mask = composed((example, dataset[0][1]))
#     visualize_matrices([dataset[0][0].argmax(0)])
#     visualize_matrices([new_example[0].argmax(0)])
#     visualize_matrices(new_example[1][0])
#     visualize_matrices(torch.cat(pos_mask))
    
#     composed = transforms.Compose([BabyARC2DColorTransform(), GenericRandomFlip(), BabyARC2DRandomResizedCrop(size=dataset[0][0].shape[-1])])
#     example = (dataset[1][0], (torch.zeros(1, 16, 16),), dataset[1][2], dataset[1][3])
#     new_example, pos_mask = composed((example, dataset[1][1]))
#     visualize_matrices([dataset[1][0].argmax(0)])
#     visualize_matrices([new_example[0].argmax(0)])
#     visualize_matrices(new_example[1][0])
#     visualize_matrices(torch.cat(pos_mask))


# In[ ]:


# if __name__ == "__main__":
#     relation_args = init_args({
#         "dataset": "y-Parallel+VerticalMid+VerticalEdge",
#         "seed": 2,
#         "n_examples": 40,
#         "canvas_size": 16,
#         "rainbow_prob": 0.,
#         "color_avail": "1,2",
#         "w_type": "image+mask",
#         "color_map_3d": "same",
#         "add_thick_surf": (0, 0.5),
#         "add_thick_depth": (0, 0.5),
#         "max_n_distractors": 2,
#         "seed_3d": 42,
#         "num_processes_3d": 10,
#         "image_size_3d": (256,256),
#         "use_seed_2d": False,
#     })
#     relation_dataset, args = get_dataset(relation_args, is_rewrite=False, is_load=True)
#     # Test color permutation
#     permute_transform = BabyARC3DColorTransform(s=0.5)
#     example = (relation_dataset[0][0], (relation_dataset[0][1][1],relation_dataset[0][1][0]), relation_dataset[0][2], relation_dataset[0][3])
#     new_example, pos_mask = permute_transform((example, relation_dataset[0][1]))
#     visualize_matrices([relation_dataset[0][0]], use_color_dict=False)
#     visualize_matrices([new_example[0]], use_color_dict=False)
#     visualize_matrices(torch.cat(new_example[1]))
#     visualize_matrices(torch.cat(pos_mask))
    
#     new_example, pos_mask = permute_transform((example, relation_dataset[0][1]))
#     visualize_matrices([relation_dataset[0][0]], use_color_dict=False)
#     visualize_matrices([new_example[0]], use_color_dict=False)
    
#     new_example, pos_mask = permute_transform((example, relation_dataset[0][1]))
#     visualize_matrices([relation_dataset[0][0]], use_color_dict=False)
#     visualize_matrices([new_example[0]], use_color_dict=False)
    
#     new_example, pos_mask = permute_transform((example, relation_dataset[0][1]))
#     visualize_matrices([relation_dataset[0][0]], use_color_dict=False)
#     visualize_matrices([new_example[0]], use_color_dict=False)

#     # Test Hflip
#     flip_transform = GenericRandomFlip()
#     example = (relation_dataset[0][0], (relation_dataset[0][1][1],relation_dataset[0][1][0]), relation_dataset[0][2], relation_dataset[0][3])
#     new_example, pos_mask = flip_transform((example, relation_dataset[0][1]))
#     visualize_matrices([relation_dataset[0][0]], use_color_dict=False)
#     visualize_matrices([new_example[0]], use_color_dict=False)
#     visualize_matrices(torch.cat(new_example[1]))
#     visualize_matrices(torch.cat(pos_mask))
    
#     # Test ResizeCrop
#     crop_transform = BabyARC3DRandomResizedCrop(size=relation_dataset[0][0].shape[-1])
#     example = (relation_dataset[0][0], (relation_dataset[0][1][0],torch.zeros(1, 256, 256)), relation_dataset[0][2], relation_dataset[0][3])
#     new_example, pos_mask = crop_transform((example, relation_dataset[0][1]))
#     visualize_matrices([relation_dataset[0][0]], use_color_dict=False)
#     visualize_matrices([new_example[0]], use_color_dict=False)
#     visualize_matrices(torch.cat(new_example[1]))
#     visualize_matrices(torch.cat(pos_mask))
    
#     example = (relation_dataset[0][0], (torch.zeros(1, 256, 256),relation_dataset[0][1][0]), relation_dataset[0][2], relation_dataset[0][3])
#     new_example, pos_mask = crop_transform((example, relation_dataset[0][1]))
#     visualize_matrices([relation_dataset[0][0]], use_color_dict=False)
#     visualize_matrices([new_example[0]], use_color_dict=False)
#     visualize_matrices(torch.cat(new_example[1]))
#     visualize_matrices(torch.cat(pos_mask))
    
#     new_example, pos_mask = crop_transform((example, relation_dataset[0][1]))
#     visualize_matrices([relation_dataset[0][0]], use_color_dict=False)
#     visualize_matrices([new_example[0]], use_color_dict=False)
#     visualize_matrices(torch.cat(new_example[1]))
#     visualize_matrices(torch.cat(pos_mask))

#     # Test composed
#     composed = transforms.Compose([BabyARC3DColorTransform(s=0.5), GenericRandomFlip(), BabyARC3DRandomResizedCrop(size=relation_dataset[0][0].shape[-1])])
#     example = (relation_dataset[0][0], (torch.zeros(1, 256, 256),relation_dataset[0][1][0]), relation_dataset[0][2], relation_dataset[0][3])
#     new_example, pos_mask = composed((example, relation_dataset[0][1]))
#     visualize_matrices([relation_dataset[0][0]], use_color_dict=False)
#     visualize_matrices([new_example[0]], use_color_dict=False)
#     visualize_matrices(torch.cat(new_example[1]))
#     visualize_matrices(torch.cat(pos_mask))
    
#     new_example, pos_mask = composed((example, relation_dataset[0][1]))
#     visualize_matrices([relation_dataset[0][0]], use_color_dict=False)
#     visualize_matrices([new_example[0]], use_color_dict=False)
#     visualize_matrices(torch.cat(new_example[1]))
#     visualize_matrices(torch.cat(pos_mask))


# ### 2.2 Sample buffer:

# In[ ]:


class SampleBuffer(object):
    def __init__(
        self,
        is_mask=False,
        is_two_branch=False,
        max_samples=10000,
    ):
        self.max_samples = max_samples
        self.is_mask = is_mask
        self.is_two_branch = is_two_branch
        self.mask_arity = None
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def push(self, samples, masks=None, class_ids=None, c_reprs=None, infos=None):
        if self.is_two_branch:
            samples = (samples[0].detach().to('cpu'), samples[1].detach().to('cpu'))
        else:
            samples = samples.detach().to('cpu')
        if self.is_mask:
            masks = tuple(masks[k].detach().to('cpu') for k in range(len(masks)))
            if self.mask_arity is None:
                self.mask_arity = len(masks)
            else:
                assert self.mask_arity == len(masks)
            c_reprs = c_reprs.detach().to('cpu')
        else:
            if isinstance(class_ids, torch.Tensor):
                class_ids = class_ids.detach().to('cpu')

        if self.is_mask:
            assert class_ids is None
            for i in range(len(c_reprs)):
                if self.is_two_branch:
                    assert len(samples) == 2
                    sample = (samples[0][i], samples[1][i])
                else:
                    sample = samples[i]
                mask = tuple(masks[k][i] for k in range(len(masks)))
                c_repr = c_reprs[i]
                info = infos[i]

                self.buffer.append((sample, mask, c_repr, info))

                if len(self.buffer) > self.max_samples:
                    self.buffer.pop(0)
        else:
            assert c_reprs is None
            for sample, class_id in zip(samples, class_ids):
                self.buffer.append((sample.detach(), class_id))

                if len(self.buffer) > self.max_samples:
                    self.buffer.pop(0)

    def get(self, n_samples, device='cuda'):
        def stack_general(List, dim=0, device=device):
            if isinstance(List[0], torch.Tensor):
                return torch.stack(List, dim=dim).to(device)
            elif isinstance(List[0], str):
                return List
            elif isinstance(List[0], Dictionary):
                return List
            else:
                raise
        items = random.choices(self.buffer, k=n_samples)
        if self.is_mask:
            """
            Zip(*[(('a',2), 1), (('b',3), 2), (('c',3), 3), (('d',2), 4)], function=function)
                ==> [[function(['a', 'b', 'c', 'd']), function([2, 3, 3, 2])], function([1, 2, 3, 4])]
            """
            samples, masks, c_reprs, infos = Zip(*items, function=stack_general)
            return samples, masks, c_reprs, infos
        else:
            samples, class_ids = Zip(*items, function=stack_general)
            return samples, class_ids


def sample_buffer(buffer, in_channels, n_classes, image_size, batch_size=128, p=0.95, is_mask=True, is_two_branch=False, w_type="image+mask", mask_arity=None, device='cuda'):
    if isinstance(image_size, Number):
        image_size = (image_size, image_size)
    if is_mask:
        w_dim = 1 if "mask" in w_type else in_channels
        if len(buffer) < 1:
            if buffer.mask_arity is None:
                if mask_arity is None:
                    buffer.mask_arity = 2 if is_two_branch else 1
                else:
                    buffer.mask_arity = mask_arity
            if is_two_branch:
                return (
                    (torch.rand(batch_size, in_channels, *image_size, device=device), torch.rand(batch_size, in_channels, *image_size, device=device)),
                    tuple(torch.rand(batch_size, w_dim, *image_size, device=device) for _ in range(buffer.mask_arity)),
                    torch.rand(batch_size, REPR_DIM, device=device),
                    [Dictionary()] * batch_size,
                )
            else:
                return (
                    torch.rand(batch_size, in_channels, *image_size, device=device),
                    tuple(torch.rand(batch_size, w_dim, *image_size, device=device) for _ in range(buffer.mask_arity)),
                    torch.rand(batch_size, REPR_DIM, device=device),
                    [Dictionary()] * batch_size,
                )

        n_replay = (np.random.rand(batch_size) < p).sum()

        replay_sample, replay_mask, replay_repr, replay_info = buffer.get(n_replay, device=device)
        if is_two_branch:
            random_sample = (torch.rand(batch_size - n_replay, in_channels, *image_size, device=device), torch.rand(batch_size - n_replay, in_channels, *image_size, device=device))
        else:
            random_sample = torch.rand(batch_size - n_replay, in_channels, *image_size, device=device)
        random_mask = tuple(torch.rand(batch_size - n_replay, w_dim, *image_size, device=device) for _ in range(buffer.mask_arity))
        random_repr = torch.rand(batch_size - n_replay, REPR_DIM, device=device)
        random_info = [Dictionary()] * (batch_size - n_replay)

        combined_mask = tuple(torch.cat([replay_mask[k], random_mask[k]], 0) for k in range(buffer.mask_arity))

        if is_two_branch:
            return (
                (torch.cat([replay_sample[0], random_sample[0]], 0), torch.cat([replay_sample[1], random_sample[1]], 0)),
                combined_mask,
                torch.cat([replay_repr, random_repr], 0),
                replay_info + random_info,
            )
        else:
            return (
                torch.cat([replay_sample, random_sample], 0),
                combined_mask,
                torch.cat([replay_repr, random_repr], 0),
                replay_info + random_info,
            )
    else:
        if len(buffer) < 1:
            return (
                torch.rand(batch_size, in_channels, *image_size, device=device),
                torch.randint(0, n_classes, (batch_size,), device=device),
            )

        n_replay = (np.random.rand(batch_size) < p).sum()

        replay_sample, replay_id = buffer.get(n_replay, device=device)
        random_sample = torch.rand(batch_size - n_replay, in_channels, *image_size, device=device)
        random_id = torch.randint(0, n_classes, (batch_size - n_replay,), device=device)

        return (
            torch.cat([replay_sample, random_sample], 0),
            torch.cat([replay_id, random_id], 0),
        )


# In[ ]:


class SampleBuffer_Conditional(object):
    def __init__(
        self,
        is_two_branch=False,
        max_samples=10000,
    ):
        """
        self.buffer is a dictionary, and has the form of 
        {
            (img_hash, class_id, ebm_target): (img, mask, c_repr, info),
            ...
        }
            When img_hash is not None, it means that img == pos_img, and mask and c_repr have at least 
                one negative examples, and ebm_target should not have "image" (e.g. "mask", "repr", "mask+repr").
            When class_id is not None, it means that c_repr == pos_repr, and both img and mask are negative
                examples, and the ebm_target == "image+mask".
        """
        self.max_samples = max_samples
        self.is_two_branch = is_two_branch
        self.mask_arity = None
        self.buffer = {}

    def __len__(self):
        return sum([len(self.buffer[key]) for key in self.buffer])

    def get_length(self, ebm_target):
        return sum([len(value) for key, value in self.buffer.items() if key[2] == ebm_target])

    def push(self, imgs=None, masks=None, c_reprs=None, class_ids=None, infos=None, ebm_target=None):
        # Check validity:
        if isinstance(imgs, tuple) or isinstance(imgs, list):
            is_image_tuple = True
            assert isinstance(imgs[0], torch.Tensor)
        else:
            is_image_tuple = False
            assert isinstance(imgs, torch.Tensor)
        # Detach and move to cpu:
        imgs = to_device_recur(imgs, "cpu", is_detach=True)
        masks = to_device_recur(masks, "cpu", is_detach=True)
        if self.mask_arity is None:
            self.mask_arity = len(masks)
        else:
            assert self.mask_arity == len(masks)
        c_reprs = c_reprs.detach().to("cpu")

        for i in range(len(c_reprs)):
            # (img_hash, class_id, ebm_target): (img, mask, c_repr)
            if is_image_tuple:
                img = (imgs[0][i], imgs[1][i])
            else:
                img = imgs[i]
            img_hash = get_image_hashing(img) if ebm_target != "image+mask" else None
            class_id = class_ids[i] if ebm_target == "image+mask" else None
            mask = tuple(masks[k][i] for k in range(len(masks)))
            c_repr = c_reprs[i]
            info = infos[i]
            record_data(self.buffer, [(img, mask, c_repr, info)], [(img_hash, class_id, ebm_target)], recent_record=self.max_samples)

    def get(
        self,
        pos_data,
        p=1,
        ebm_target=None,
        transforms="None",
        color_avail=None,
        device='cuda',
    ):
        """
        Given the positive examples, will sample the negative examples according to the ebm_target.
            If ebm_target == "image+mask", the neg_examples will have the same c_repr as the positive examples.
            Otherwise, will first sample from the key that has the same img_hash and ebm_target, and if not available,
            sample random instance that has the same ebm_target.

        Zip(*[(('a',2), 1), (('b',3), 2), (('c',3), 3), (('d',2), 4)], function=function)
            ==> [[function(['a', 'b', 'c', 'd']), function([2, 3, 3, 2])], function([1, 2, 3, 4])]

        Args:
            p: probability of using the samples from buffer. Otherwise use random samples.

        Returns:
            neg_data: contains (neg_imgs, neg_masks, neg_reprs, neg_infos).
        """
        def stack_general(List, dim=0, device=device):
            if isinstance(List[0], torch.Tensor):
                return torch.stack(List, dim=dim).to(device)
            elif isinstance(List[0], str):
                return List
            elif isinstance(List[0], Dictionary):
                return List
            else:
                raise
        assert ebm_target is not None
        pos_imgs, pos_masks, pos_ids, pos_infos = pos_data
        neg_imgs, neg_masks, neg_reprs, neg_infos = [], [], [], []
        if isinstance(pos_imgs, tuple) or isinstance(pos_imgs, list):
            is_image_tuple = True
            assert isinstance(pos_imgs[0], torch.Tensor)
        else:
            is_image_tuple = False
            assert isinstance(pos_imgs, torch.Tensor)
        if ebm_target == "image+mask":
            # The c_repr is the same as that of the pos_data:
            for i, pos_id in enumerate(pos_ids):
                if np.random.rand() < p:
                    # From buffer:
                    lst = buffer.buffer[(None, pos_id, ebm_target)]
                    id = np.random.choice(len(lst))
                    # Apply transformations
                    if transforms == "None":
                        img, mask, c_repr, info = lst[id]
                    else:
                        transforms_split = transforms.split(":")
                        if len(transforms_split) == 1:
                            p_transforms = 1
                            transforms_core = transforms
                        else:
                            p_transforms = eval(transforms_split[1])
                            transforms_core = transforms_split[0]
                        if np.random.rand() > p_transforms:
                            img, mask, c_repr, info = lst[id]
                        else:
                            pos_mask = tuple(pos_masks[k][i] for k in range(len(pos_masks)))
                            is_rgb = True if pos_imgs[0].shape[0] == 3 else False
                            augment = get_augment(transforms_core, pos_imgs[0].shape[-1], is_rgb=is_rgb, color_avail=color_avail)
                            img, mask, c_repr, info = augment((lst[id], pos_mask))[0]
                    if i == 0:
                        pos_repr = id_to_tensor([pos_id], CONCEPTS=CONCEPTS, OPERATORS=OPERATORS)[0]
                        assert (c_repr == pos_repr).all()
                    info["src"] = "buffer-same"
                else:
                    # From random:
                    img = (torch.rand_like(pos_imgs[0][0]), torch.rand_like(pos_imgs[1][0])) if is_image_tuple else torch.rand_like(pos_imgs[0])
                    mask = tuple(torch.rand_like(pos_masks[0][0]) for _ in range(len(pos_masks)))
                    c_repr = id_to_tensor([pos_id], CONCEPTS=CONCEPTS, OPERATORS=OPERATORS)[0]
                    info = Dictionary({"src": "random"})
                neg_imgs.append(img)
                neg_masks.append(mask)
                neg_reprs.append(c_repr)
                neg_infos.append(info)
        else:
            # The neg_img is the same as that of the pos_data:
            for i in range(len(pos_ids)):
                if is_image_tuple:
                    img = (pos_imgs[0][i], pos_imgs[1][i])
                else:
                    img = pos_imgs[i]
                if np.random.rand() < p:
                    # From buffer:
                    img_hash = get_image_hashing(img)
                    key = (img_hash, None, ebm_target)
                    if key in buffer.buffer:
                        # Find the items that has the same img_hash and ebm_target as the pos_data:
                        lst = buffer.buffer[key]
                    else:
                        # Find the items that has the same ebm_target as the given ebm_target:
                        lst = list(itertools.chain.from_iterable([
                            item for key, item in buffer.buffer.items() if key[2] == ebm_target]))
                    id = np.random.choice(len(lst))
                    # Apply transformations
                    if transforms == "None":
                        img_load, mask, c_repr, info = lst[id]
                        if i == 0 and key in buffer.buffer:
                            assert img_hash == get_image_hashing(img_load)
                    else:
                        transforms_split = transforms.split(":")
                        if len(transforms_split) == 1:
                            p_transforms = 1
                            transforms_core = transforms
                        else:
                            p_transforms = eval(transforms_split[1])
                            transforms_core = transforms_split[0]
                        if np.random.rand() > p_transforms:
                            img_load, mask, c_repr, info = lst[id]
                        else:
                            pos_mask = tuple(pos_masks[k][i] for k in range(len(pos_masks)))
                            is_rgb = True if pos_imgs[0].shape[0] == 3 else False
                            augment = get_augment(transforms_core, pos_imgs[0].shape[-1], is_rgb=is_rgb, color_avail=color_avail)
                            img, mask, c_repr, info = augment((lst[id], pos_mask))[0]
                    if key in buffer.buffer:
                        info["src"] = "buffer-same"
                    else:
                        info["src"] = "buffer-diff"
                else:
                    # From random:
                    if "mask" in ebm_target:
                        mask = tuple(torch.rand_like(pos_masks[0][0]) for _ in range(len(pos_masks)))
                    else:
                        # If "mask" is not in ebm_target, then use the ground-truth:
                        mask = tuple(pos_masks[k][i] for k in range(len(pos_masks)))
                    if "repr" in ebm_target:
                        c_repr = torch.rand(REPR_DIM)
                    else:
                        # If "repr" is not in ebm_target, then use the ground-truth:
                        c_repr = id_to_tensor([pos_ids[i]], CONCEPTS=CONCEPTS, OPERATORS=OPERATORS)[0]
                    info = Dictionary({"src": "random"})
                neg_imgs.append(img)
                neg_masks.append(mask)
                neg_reprs.append(c_repr)
                neg_infos.append(info)
        if is_image_tuple:
            neg_imgs = Zip(*neg_imgs, function=stack_general)
        else:
            neg_imgs = torch.stack(neg_imgs).to(device)
        neg_masks = Zip(*neg_masks, function=stack_general)
        neg_reprs = torch.stack(neg_reprs).to(device)
        neg_data = (neg_imgs, neg_masks, neg_reprs, neg_infos)
        return neg_data

    def __repr__(self):
        return "SampleBuffer_Conditional(keys={}, examples={})".format(len(self.buffer), len(self))


def sample_buffer_conditional(
    buffer,
    pos_data,
    ebm_target,
    in_channels,
    image_size,
    batch_size=128,
    p=0.95,
    is_two_branch=False,
    is_image_tuple=False,
    w_type="image+mask",
    transforms="None",
    color_avail="-1",
    device='cuda',
):
    """
    Given the positive examples, will sample the negative examples according to the ebm_target.
    E.g. if ebm_target == "image+mask", the neg_examples will have the same c_repr as the positive examples.
    """
    if isinstance(image_size, Number):
        image_size = (image_size, image_size)
    w_dim = 1 if "mask" in w_type else in_channels
    if buffer.get_length(ebm_target) <= len(pos_data[2]):
        if buffer.mask_arity is None:
            buffer.mask_arity = 2 if is_two_branch else 1
        random_img = (torch.rand(batch_size, in_channels, *image_size, device=device),
                      torch.rand(batch_size, in_channels, *image_size, device=device)) if is_image_tuple else \
                    torch.rand(batch_size, in_channels, *image_size, device=device)
        return (
            random_img,
            tuple(torch.rand(batch_size, w_dim, *image_size, device=device) for _ in range(buffer.mask_arity)),
            torch.rand(batch_size, REPR_DIM, device=device),
            [Dictionary()] * batch_size,
        )
    else:
        return buffer.get(pos_data, p=p, ebm_target=ebm_target, transforms=transforms, color_avail=color_avail, device=device)


# In[ ]:


# if __name__ == "__main__":
#     imgs=neg_img
#     masks=neg_mask
#     class_ids=["Hshape"] * 128
#     c_reprs=neg_repr
#     infos=neg_info
#     ebm_target="image+mask"

#     buffer = SampleBuffer_Conditional()
#     buffer.push(imgs, masks, c_reprs, class_ids, infos, ebm_target)

#     sample = buffer.get(pos_data, p=0.9)


# ### 2.3 Negative examples:

# In[ ]:


def generate_neg_examples(pos_img, pos_mask, pos_repr, neg_mode):
    """Generate negative examples according to the neg_mode.

    Args:
        neg_mode: choose from
                    "addrand" (add random pixels from the image),
                    "permlabel" (randomly permute the pos_repr),
                    "addallrand" (add random pixels on the empty place of the positive masks),
                    or using "+" to combine any subset of them.

    Returns:
        neg_mask_gen: generated negative masks
        neg_repr_gen: generated (permuted c_repr)
        neg_gen_valid: valid negative examples
    """
    if "+" in neg_mode:
        # multiple modes, randomly assign one mode for each example:
        neg_mode_split = neg_mode.split("+")
        is_two_branch = isinstance(pos_img, tuple) or isinstance(pos_img, list)
        device = pos_repr.device
        length = len(pos_img[0]) if is_two_branch else len(pos_img)
        id_assign = torch.randint(len(neg_mode_split), size=(length,))
        neg_mask_gen = tuple(-torch.ones_like(pos_mask[k]).to(device)
                             for k in range(len(pos_mask)))
        neg_repr_gen = -torch.ones_like(pos_repr).to(device)
        neg_gen_valid = -torch.ones(length, 1).to(device)

        for i in range(len(neg_mode_split)):
            id_assign_i = id_assign == i
            if is_two_branch:
                pos_img_ele = (pos_img[0][id_assign_i],
                               pos_img[1][id_assign_i])
            else:
                pos_img_ele = pos_img[id_assign_i]
            pos_mask_ele = tuple(pos_mask[k][id_assign_i]
                                 for k in range(len(pos_mask)))
            pos_repr_ele = pos_repr[id_assign_i]
            neg_mask_gen_i, neg_repr_gen_i, neg_gen_valid_i = generate_neg_examples(
                pos_img_ele,
                pos_mask_ele,
                pos_repr_ele,
                neg_mode=neg_mode_split[i],  # use the assigned mode
            )
            for k in range(len(neg_mask_gen)):
                neg_mask_gen[k][id_assign_i] = neg_mask_gen_i[k]
            neg_repr_gen[id_assign_i] = neg_repr_gen_i
            neg_gen_valid[id_assign_i] = neg_gen_valid_i

        # Make sure that all examples are operated on:
        for k in range(len(neg_mask_gen)):
            assert (neg_mask_gen[k] > -1).all()
        if "softmax" not in args.c_repr_mode and args.is_pos_repr_learnable is False:
            assert (neg_repr_gen > -1).all()
        assert (neg_gen_valid > -1).all()
        # Returns:
        return neg_mask_gen, neg_repr_gen, neg_gen_valid

    else:
        # Single mode:
        if neg_mode == "addrand":
            # Randomly add pixels to the other places of the image where there is object:
            if len(pos_mask) > 1:
                if not (isinstance(pos_img, tuple) or isinstance(pos_img, list)):
                    pos_img = (pos_img, pos_img)
                assert pos_img[0].shape[1] == 10, "if channel_size is 10, then cannot use neg_mask_mode of 'addrand'."
                # [B, 1, H, W]; pos_mask[0]: [B, 1, H, W]
                pos_non_zero = (pos_img[0][:, :1] != 1, pos_img[1][:, :1] != 1)
                remainder = (pos_non_zero[0] & (
                    ~pos_mask[0].bool()), pos_non_zero[1] & (~pos_mask[1].bool()))
                # valid if the remainder is not all-zero:
                neg_gen_valid = ((remainder[0].sum((1, 2, 3)) > 0) | (
                    remainder[1].sum((1, 2, 3)) > 0)).unsqueeze(1).float()
                remainder_random = (remainder[0]*torch.randint(2, size=remainder[0].shape).to(
                    remainder[0].device), remainder[1]*torch.randint(2, size=remainder[0].shape).to(remainder[1].device))
                neg_mask_gen = (
                    pos_mask[0]+remainder_random[0], pos_mask[1]+remainder_random[1])
                neg_repr_gen = pos_repr
            else:
                assert pos_img.shape[1] == 10, "if channel_size is 10, then cannot use neg_mask_mode of 'addrand'."
                assert len(pos_mask) == 1
                # [B, 1, H, W]; pos_mask: [B, 1, H, W]
                pos_non_zero = pos_img[:, :1] != 1
                remainder = pos_non_zero & (~pos_mask[0].bool())
                # valid if the remainder is not all-zero:
                neg_gen_valid = (remainder.sum((1, 2, 3))
                                 > 0).unsqueeze(1).float()
                remainder_random = remainder *                     torch.randint(2, size=remainder.shape).to(remainder.device)
                neg_mask_gen = (pos_mask[0] + remainder_random,)
                neg_repr_gen = pos_repr
        elif neg_mode == "addallrand":
            # Randomly add pixels to the other places of the image where there is no object:
            if len(pos_mask) > 1:
                if not (isinstance(pos_img, tuple) or isinstance(pos_img, list)):
                    pos_img = (pos_img, pos_img)
                # [B, 1, H, W]; pos_mask[0]: [B, 1, H, W]
                remainder = (~pos_mask[0].bool(), ~pos_mask[1].bool())
                p0, p1 = pos_mask[0].mean().item(), pos_mask[1].mean().item()
                remainder_random = ((torch.rand(pos_mask[0].shape, device=pos_mask[0].device) < p0).float(),
                                    (torch.rand(pos_mask[1].shape, device=pos_mask[1].device) < p1).float())
                remainder_random = (remainder[0]*remainder_random[0], remainder[1]*remainder_random[1])

                # valid if the remainder is not all-zero:
                neg_gen_valid = ((remainder_random[0].sum((1, 2, 3)) > 0) | (
                    remainder_random[1].sum((1, 2, 3)) > 0)).unsqueeze(1).float()
                neg_mask_gen = (
                    pos_mask[0]+remainder_random[0], pos_mask[1]+remainder_random[1])
                neg_repr_gen = pos_repr
            else:
                assert len(pos_mask) == 1
                # [B, 1, H, W]; pos_mask: [B, 1, H, W]
                remainder = ~pos_mask[0].bool()
                p0 = pos_mask[0].mean().item()
                # valid if the remainder is not all-zero:
                remainder_random = (torch.rand(pos_mask[0].shape, device=pos_mask[0].device) < p0).float() * remainder
                neg_gen_valid = (remainder_random.sum((1, 2, 3))
                                 > 0).unsqueeze(1).float()
                neg_mask_gen = (pos_mask[0] + remainder_random,)
                neg_repr_gen = pos_repr
        elif neg_mode == "delrand":
            if len(pos_mask) > 1:
                remove_random = (pos_mask[0]*torch.randint(2, size=pos_mask[0].shape).to(pos_mask[0].device),
                                 pos_mask[1]*torch.randint(2, size=pos_mask[0].shape).to(pos_mask[1].device))
                # valid if the remove_random is not all-zero:
                neg_gen_valid = ((remove_random[0].sum((1, 2, 3)) > 0) | (
                    remove_random[1].sum((1, 2, 3)) > 0)).unsqueeze(1).float()
                neg_mask_gen = (
                    pos_mask[0]-remove_random[0], pos_mask[1]-remove_random[1])
                neg_repr_gen = pos_repr
            else:
                assert len(pos_mask) == 1
                # pos_mask: [B, 1, H, W]
                remove_random = pos_mask[0]*torch.randint(2, size=pos_mask[0].shape).to(pos_mask[0].device)

                # valid if the remainder is not all-zero:
                neg_gen_valid = (remove_random.sum((1, 2, 3)) > 0).unsqueeze(1).float()
                neg_mask_gen = (pos_mask[0] - remove_random,)
                neg_repr_gen = pos_repr
        elif neg_mode == "permlabel":
            # Permute pos_repr:
            id_rand = torch.randperm(len(pos_repr))
            neg_repr_gen = pos_repr[id_rand]
            # Valid if the c_repr is different:
            neg_gen_valid = (neg_repr_gen != pos_repr).any(
                1, keepdims=True).float()
            neg_mask_gen = pos_mask
        else:
            raise Exception("neg_mode '{}' is not supported!".format(neg_mode))
        return neg_mask_gen, neg_repr_gen, neg_gen_valid


# ### 2.4 Loss function:

# In[ ]:


def get_loss_core(args, pos_out, neg_out, emp_out=None, neg_out_gen=None):
    """
    Compute the core loss from pos_out, neg_out, emp_out, based on the energy_mode.
    If the loss involves emp_out, will only be valid if is_emp_loss(args) is True.

    Args:
        pos_out, neg_out, emp_out, neg_out_gen: tensors of the same shape, typically [B, 1].
        energy_mode:
            "standard:0.3": (E_pos - E_neg) * 0.3
            "margin^0.2:0.3": max(0, 0.3 + E_pos - E_neg) * 0.2
            "mid^0.2:0.3": (max(0, 0.2 + E_pos - E_empty) + max(0, 0.2 + E_empty - E_neg)) * 0.3
            "mid^0.2^adapt:0.3":
                (max(0, gamma + E_pos - E_empty) + max(0, gamma + E_empty - E_neg)) * 0.3
                    where gamma = max(0, StopGrad(E_neg - E_pos)/2) + 0.2
            "standard:0.5+mid^0.2^adapt:0.3":
                (E_pos - E_neg) * 0.5 + (max(0, gamma + E_pos - E_empty) + max(0, gamma + E_empty - E_neg)) * 0.3,
                    where gamma = max(0, StopGrad(E_neg - E_pos)/2) + 0.2
            "standard+center^stop": (E_pos - E_neg) * 1 + ((E_pos+E_neg).detach()/2 - E_empty).abs()
                "stop": stop gradient, and each empty loss is computed per example
                "stopgen": similar to "stop", but the negative energy is the mean of neg_out and neg_out_gen, per example.
                "stopmean": stop gradient, and each empty loss is computed per minibatch
                "stopgenmean": similar to "stopmean", but the negative energy is the mean of neg_out and neg_out_gen, per minibatch.
    Returns:
        loss_core: total loss, in the same shape as pos_out, etc.
        loss_core_info: dictionary of each of the average loss for the component of the energy_mode.
    """
    loss_core = 0
    loss_core_info = {}
    device = pos_out.device
    for energy_mode_ele in args.energy_mode.split("+"):
        if len(energy_mode_ele.split(":")) > 1:
            coef = eval(energy_mode_ele.split(":")[-1])
        else:
            coef = 1
        energy_mode_ele = energy_mode_ele.split(":")[0]
        if energy_mode_ele.startswith("standard"):
            loss_ele = (pos_out - neg_out) * coef
        elif energy_mode_ele.startswith("margin"):
            gamma = eval(energy_mode_ele.split("^")[1])
            loss_ele = torch.maximum(torch.tensor(0).to(device), pos_out - neg_out + gamma) * coef
        elif energy_mode_ele.startswith("mid"):
            if not is_emp_loss(args):
                continue
            gamma = eval(energy_mode_ele.split("^")[1])
            is_adapt = "adapt" in energy_mode_ele
            if is_adapt:
                gamma = gamma + max(0, (neg_out - pos_out).mean().item() / 2)
            loss_ele = torch.maximum(torch.tensor(0).to(device), pos_out - emp_out + gamma) * coef +                        torch.maximum(torch.tensor(0).to(device), emp_out - neg_out + gamma) * coef
        elif energy_mode_ele.startswith("center"):
            if not is_emp_loss(args):
                continue
            grad_type = energy_mode_ele.split("^")[1] if len(energy_mode_ele) > 1 else "None"
            if grad_type == "None":
                loss_ele = ((pos_out + neg_out) / 2 - emp_out).abs() * coef
            elif grad_type == "stop":
                loss_ele = ((pos_out + neg_out).detach() / 2 - emp_out).abs() * coef
            elif grad_type == "stopsq":
                loss_ele = ((pos_out + neg_out).detach() / 2 - emp_out).square() * coef
            elif grad_type == "stopgen":
                if neg_out_gen is not None:
                    loss_ele = ((pos_out + (neg_out + neg_out_gen)/2).mean() / 2 - emp_out).abs() * coef
                else:
                    loss_ele = ((pos_out + neg_out).mean() / 2 - emp_out).abs() * coef
            elif grad_type == "stopmean":
                loss_ele = ((pos_out + neg_out).detach().mean() / 2 - emp_out).abs() * coef
            elif grad_type == "stopsqmean":
                loss_ele = ((pos_out + neg_out).detach().mean() / 2 - emp_out).square() * coef
            elif grad_type == "stopgenmean":
                if neg_out_gen is not None:
                    loss_ele = ((pos_out + (neg_out + neg_out_gen)/2).detach().mean() / 2 - emp_out).abs() * coef
                else:
                    loss_ele = ((pos_out + neg_out).detach().mean() / 2 - emp_out).abs() * coef
            else:
                raise
        else:
            raise
        loss_core = loss_core + loss_ele
        loss_core_info[energy_mode_ele.split("^")[0]] = loss_ele.mean().item()
    return loss_core, loss_core_info


# ### 2.5 Helper functions:

# In[ ]:


def get_image_hashing(img):
    """Get the hashing of an image."""
    if isinstance(img, tuple) or isinstance(img, list):
        hashing = get_hashing(str(torch.stack(img).long().view(-1)), length=10)
    else:
        hashing = get_hashing(str(img.long().view(-1)), length=10)
    return hashing


def get_concept_embeddings(CONCEPTS, OPERATORS):
    concept_embeddings = {key: to_np_array(CONCEPTS[key].get_node_repr()) for key in CONCEPTS}
    operator_embeddings = {key: to_np_array(OPERATORS[key].get_node_repr()) for key in OPERATORS}
    concept_embeddings.update(operator_embeddings)
    return concept_embeddings


def unittest(model, args, device, CONCEPTS, OPERATORS):
    """Make sure that the loaded model is the same as the original model."""
    if isinstance(model, nn.parallel.DataParallel):
        model = model.module
    model.eval()
    buffer = SampleBuffer()
    neg_data = sample_buffer(
        buffer,
        in_channels=args.in_channels,
        n_classes=args.n_classes,
        image_size=args.image_size if args.rescaled_size == "None" else eval(args.rescaled_size),
        batch_size=args.batch_size,
        is_mask=args.is_mask,
        is_two_branch=args.is_two_branch,
        w_type=args.w_type,
        p=args.p_buffer,
        device=device,
    )
    model2 = load_model_energy(model.model_dict, device=device)
    model2.eval()
    if args.is_mask:
        neg_img, neg_mask, neg_repr, neg_info = neg_data
        neg_out = model(neg_img, mask=neg_mask, c_repr=neg_repr)
        neg_out2 = model2(neg_img, mask=neg_mask, c_repr=neg_repr)
    else:
        neg_img, neg_id = neg_data
        neg_out = model(neg_img, neg_id)
        neg_out2 = model2(neg_img, neg_id)
    diff_max = (neg_out2 - neg_out).abs().max().item()

    if diff_max < 8e-6:
        p.print("The largest diff for sample neg_img is {}, smaller than 8e-6. Unittest passed!".format(diff_max))
    else:
        raise Exception("The largest diff for sample neg_img is {}, greater than 8e-6. Check model loading and saving!".format(diff_max))


def get_filename(args, short_str_dict, is_local_path):
    filename_short = get_filename_short(
        short_str_dict.keys(),
        short_str_dict,
        args_dict=args.__dict__,
    )
    dirname = EXP_PATH + "/{}_{}/".format(args.exp_id, args.date_time)
    if args.exp_name != "None":
        # If args.exp_name != "None", the experiments are saved under "{exp_id}_{date_time}/{exp_name}/"
        dirname += "{}/".format(args.exp_name)
    filename = dirname + filename_short[:-2] + "_{}.p".format(get_machine_name())
    make_dir(filename)
    p.print(filename, banner_size=100)
    return dirname, filename


def get_ebm_target(args):
    if args.ebm_target_mode.startswith("r-"):
        # randomly sample mode:
        ebm_str = args.ebm_target_mode.split("-")[1]
        ebm_str_dict = {"r": "repr", "m": "mask", "b": "mask+repr", "x": "image+mask"}
        ebm_target_collection = [ebm_str_dict[key] for key in ebm_str]
        args.ebm_target = np.random.choice(ebm_target_collection)
    elif args.ebm_target_mode == "None":
        pass
    else:
        raise
    return args


def is_emp_loss(args):
    """Return True if the current args.ebm_target belongs to the args.emp_target."""
    if args.emp_target_mode == "all":
        emp_target_mode = args.ebm_target_mode.split("-")[1]
    else:
        assert args.emp_target_mode.startswith("r-")
        emp_target_mode = args.emp_target_mode.split("-")[1]
    ebm_str_reverse_dict = {"repr": "r", "mask": "m", "mask+repr": "b", "image+mask": "x"}
    current_str = ebm_str_reverse_dict[args.ebm_target]
    return current_str in emp_target_mode


# In[ ]:


def init_concepts(n_concepts, n_relations):
    from zeroc.concept_library.concepts import Concept, Graph, Placeholder, Tensor
    CONCEPTS = OrderedDict()
    OPERATORS = OrderedDict()
    IS_CUDA = False
    num_colors = 10
    CONCEPTS["Image"] = Concept(name="Image",
        repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
        inherit_to=[f"c{i}" for i in range(n_concepts)],
        value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
    )

    CONCEPTS["Bool"] = Concept(name="Bool",
        repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
        value=Placeholder(Tensor(dtype="bool", shape=(1,), range=[True, False])),
    )

    for i in range(n_concepts):
        CONCEPTS[f"c{i}"] = Concept(name=f"c{i}",
            repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
            inherit_from=["Image"],
            value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
        )

    for i in range(n_relations):
        OPERATORS[f"r{i}"] = Graph(name=f"r{i}",
            repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
            forward={"args": [Placeholder("Image"), Placeholder("Image")],
                     "output": Placeholder("Bool"),
                     "fun": identity_fun,
                    })
    return CONCEPTS, OPERATORS


def init_concepts_with_repr(concept_repr_dict=None, relation_repr_dict=None):
    from zeroc.concept_library.concepts import Concept, Graph, Placeholder, Tensor
    CONCEPTS = OrderedDict()
    OPERATORS = OrderedDict()
    IS_CUDA = False
    num_colors = 10
    CONCEPTS["Image"] = Concept(name="Image",
        repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
        inherit_to=list(concept_repr_dict.keys()),
        value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
    )

    CONCEPTS["Bool"] = Concept(name="Bool",
        repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
        value=Placeholder(Tensor(dtype="bool", shape=(1,), range=[True, False])),
    )

    if concept_repr_dict is not None:
        for c_str, c_repr in concept_repr_dict.items():
            CONCEPTS[c_str] = Concept(name=c_str,
                repr=c_repr,
                inherit_from=["Image"],
                value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
            )
    if relation_repr_dict is not None:
        for c_str, c_repr in relation_repr_dict.items():
            OPERATORS[c_str] = Graph(name=c_str,
                repr=c_repr,
                forward={"args": [Placeholder("Image"), Placeholder("Image")],
                         "output": Placeholder("Bool"),
                         "fun": identity_fun,
                        }
            )
    return CONCEPTS, OPERATORS


# ## 3. Test_acc:

# In[ ]:


def get_arc_relations(inp_img, info):
    """Returns a list of relations in the format of (id1, id2, relation_type),
    where relation_type is SameShape, SameColor, IsInside, etc.
    """
    # Convert masks into objects
    obj_lst = []
    for i in range(2):
        obj_id = f"obj_{i}"
        mask = info['id_object_mask'][info['node_id_map'][obj_id]]
        obj_value, obj_pos = shrink(get_obj_from_mask(inp_img.argmax(0), mask))
        obj_lst.append((i, CONCEPTS[DEFAULT_OBJ_TYPE].copy().set_node_value(obj_value).set_node_value(obj_pos, "pos")))

    relations = []
    for idx1 in range(len(obj_lst)):
        for idx2 in range(idx1 + 1, len(obj_lst)):
            # Use functions defined in concepts_ARC2
            rel_types = []
            id1, obj1 = obj_lst[idx1]
            id2, obj2 = obj_lst[idx2]
            if SameShape(obj1, obj2):
                relations.append((id1, id2, "SameShape"))
            if SameColor(obj1, obj2):
                relations.append((id1, id2, "SameColor"))
            if SameAll(obj1, obj2):
                relations.append((id1, id2, "SameAll"))
            if SameRow(obj1, obj2):
                relations.append((id1, id2, "SameRow"))
            if SameCol(obj1, obj2):
                relations.append((id1, id2, "SameCol"))
            if IsNonOverlapXY(obj1, obj2):
                relations.append((id1, id2, "IsNonOverlapXY"))
            # Uncommutative relations
            if SubsetOf(obj1, obj2):
                relations.append((id1, id2, "SubsetOf"))
            if SubsetOf(obj2, obj1):
                relations.append((id2, id1, "SubsetOf"))
            if IsInside(obj1, obj2):
                relations.append((id1, id2, "IsInside"))
            if IsInside(obj2, obj1):
                relations.append((id1, id2, "IsEnclosed"))
    return relations


def get_all_masks(pos_img, pos_id, pos_info, args):
    # For each example in the batch, get all masks that match the pos_id
    # max_num_occur is the maximum number of occurrences of a concept type in an example
    concept_str_mapping = {
        "line": "Line",
        "rectangle": "Rect",
        "rectangleSolid": "RectSolid",
        "Lshape": "Lshape",
        "Tshape": "Tshape",
        "Eshape": "Eshape",
        "Hshape": "Hshape",
        "Cshape": "Cshape",
        "Ashape": "Ashape",
        "Fshape": "Fshape",
        "randomShape": "Randshape",
        "arcShape": "ARCshape",
        "Red": "Red",
        "Blue": "Blue",
        "Green": "Green",
        "Cube": "Cube",
        "Cylinder": "Cylinder",
        "Large": "Large",
        "Small": "Small",
    }
    concept_str_reverse_mapping = {item: key for key, item in concept_str_mapping.items()}
    batch_masks  = []
    # Go through the batch
    for idx, info in enumerate(pos_info):
        ex_masks = torch.zeros(args.max_num_occur, 2 if args.is_two_branch else 1, *(args.image_size if args.rescaled_size == "None" else eval(args.rescaled_size)))
        if args.is_two_branch:
            found_relation = False
            ex_ind = 0
            relations = info["relations"] if "relations" in info else get_arc_relations(pos_img[idx], info)
            for tup in relations:
                if tup[2] == pos_id[idx]:
                    # Create the mask mask dimension
                    if args.rescaled_size == "None":
                        if tup[2] != "IsInside":
                            ex_masks[ex_ind,] = torch.stack((info['id_object_mask'][tup[0]].squeeze(),
                                                             info['id_object_mask'][tup[1]].squeeze()))
                        else:
                            ex_masks[ex_ind,] = torch.stack((info['id_object_mask'][tup[1]].squeeze(),
                                                             info['id_object_mask'][tup[0]].squeeze()))
                    else:
                        if tup[2] != "IsInside":
                            ex_masks[ex_ind] = torch.cat((rescale_tensor(info['id_object_mask'][tup[0]], rescaled_size=args.rescaled_size),
                                                          rescale_tensor(info['id_object_mask'][tup[1]], rescaled_size=args.rescaled_size)))
                        else:
                            ex_masks[ex_ind] = torch.cat((rescale_tensor(info['id_object_mask'][tup[1]], rescaled_size=args.rescaled_size),
                                                          rescale_tensor(info['id_object_mask'][tup[0]], rescaled_size=args.rescaled_size)))
                    found_relation = True
                    ex_ind += 1
            assert found_relation, "Should be at least one relation matching the passed in pos_id"
        else:
            found_concept = False
            ex_ind = 0
            """
            Example for obj_spec: 
            [[('obj_0', 'line_[-1,1,-1]'), 'Attr'],
             [('obj_1', 'line_[-1,1,-1]'), 'Attr'],
            ]
            """
            for obj_lst in info["obj_spec"]:
                # Get obj name
                tup = obj_lst[0]  # tup: ('obj_0', 'line_[-1,1,-1]')
                obj_name, obj_type = tup[0], tup[1].split("_")[0].split("+")  # obj_name: 'obj_0', obj_type: ['line']
                if concept_str_reverse_mapping[pos_id[idx]] in obj_type:
                    if args.rescaled_size == "None":
                        ex_masks[ex_ind] = info["id_object_mask"][info["node_id_map"][obj_name]].unsqueeze(0)
                    else:
                        ex_masks[ex_ind] = rescale_tensor(
                            info["id_object_mask"][info["node_id_map"][obj_name]].unsqueeze(0),
                            rescaled_size=args.rescaled_size)
                    found_concept = True
                    ex_ind += 1
            assert found_concept, "Should be at least one concept matching the passed in pos_id"
        ex_masks = ex_masks.unsqueeze(2) # Create the channel dimension
        batch_masks.append(ex_masks)
    return torch.stack(batch_masks) # Create the batch dimension


def test_acc_mask(
    pos_data,
    args,
    neg_mask,
    CONCEPTS,
    OPERATORS,
    reduction="mean",
    device="cpu",
):
    uncommunitable_relation = ['IsInside']
    result_dict = {}
    pos_img, pos_mask, pos_id, pos_info = pos_data
    pos_all_masks = get_all_masks(pos_img, pos_id, pos_info, args)
    pos_repr = id_to_tensor(pos_id, CONCEPTS=CONCEPTS, OPERATORS=OPERATORS).to(device) # vector of len 4
    pos_img, pos_mask, pos_all_masks = to_device_recur([pos_img, pos_mask, pos_all_masks], device)

    # Now we have pos_mask and neg_mask, given c_repr. We calculate the accuracy for neg_mask:
    if args.is_two_branch:
        pos_img = (pos_img, pos_img) if not args.is_image_tuple else pos_img
        is_commutable = torch.Tensor([0 if ele in uncommunitable_relation else 1 for ele in pos_id]).unsqueeze(1).to(device) # [B*1], True or False. 
        is_uncommutable = 1 - is_commutable # [B*1], True or False.

        # Use mask_iou:
        neg_repeated = repeat(torch.stack(neg_mask, 1), "b m c h w -> b p m c h w", p=args.max_num_occur)
        neg_repeated = rearrange(neg_repeated, "b p m c h w -> (b p) m c h w")
        pos_all_masks = rearrange(pos_all_masks, "b p m c h w -> (b p) m c h w")
        mask_acc_0_original = mask_iou_score(neg_repeated[:, 0, ...], pos_all_masks[:, 0, ...])
        mask_acc_0_commute = mask_iou_score(neg_repeated[:, 0, ...], pos_all_masks[:, 1, ...])
        mask_acc_1_original = mask_iou_score(neg_repeated[:, 1, ...], pos_all_masks[:, 1, ...])
        mask_acc_1_commute = mask_iou_score(neg_repeated[:, 1, ...], pos_all_masks[:, 0, ...])
        orig_sum = (mask_acc_0_original + mask_acc_1_original) / 2
        alt_sum = (mask_acc_0_commute + mask_acc_1_commute) / 2
        # Rearrange from [B * P, ...] to [B, P, ...]
        orig_sum = rearrange(orig_sum, "(b p) -> b p", p=args.max_num_occur)
        alt_sum = rearrange(alt_sum, "(b p) -> b p", p=args.max_num_occur)
        # Get the best match among all possible pairs 
        values1, best_ind1 = orig_sum.max(1)
        values2, best_ind2 = alt_sum.max(1)
        # For each batch element, whether to use original or commute
        bool_val = torch.logical_or(values1 > values2, is_uncommutable.squeeze())
        if reduction == "mean":
            mask_acc = to_np_array(torch.where(bool_val, values1, values2).mean())
        elif reduction == "none":
            mask_acc = to_np_array(torch.where(bool_val, values1, values2))
        else:
            raise
        batch_size = pos_img[0].shape[0]
        mask_acc_0_orig_max = rearrange(mask_acc_0_original, "(b p) -> b p", p=args.max_num_occur)[torch.arange(batch_size), best_ind1]
        mask_acc_1_orig_max = rearrange(mask_acc_1_original, "(b p) -> b p", p=args.max_num_occur)[torch.arange(batch_size), best_ind1]
        mask_acc_0_com_max = rearrange(mask_acc_0_commute, "(b p) -> b p", p=args.max_num_occur)[torch.arange(batch_size), best_ind2]
        mask_acc_1_com_max = rearrange(mask_acc_1_commute, "(b p) -> b p", p=args.max_num_occur)[torch.arange(batch_size), best_ind2]
        mask_acc_0 = torch.where(bool_val, mask_acc_0_orig_max, mask_acc_0_com_max)
        mask_acc_1 = torch.where(bool_val, mask_acc_1_orig_max, mask_acc_1_com_max)
        result_dict["mask_acc_0"] = to_np_array(mask_acc_0)
        result_dict["mask_acc_1"] = to_np_array(mask_acc_1)
    else:
        assert len(pos_mask) == 1 and len(neg_mask) == 1
        # Use mask_iou
        neg_repeated = repeat(torch.stack(neg_mask, 1), "b m c h w -> b p m c h w", p=args.max_num_occur)
        neg_repeated = rearrange(neg_repeated, "b p m c h w -> (b p) m c h w")
        pos_all_masks = rearrange(pos_all_masks, "b p m c h w -> (b p) m c h w")
        mask_acc = mask_iou_score(neg_repeated[:, 0, ...], pos_all_masks[:, 0, ...])
        mask_acc = rearrange(mask_acc, "(b p) -> b p", p=args.max_num_occur)
        mask_acc = to_np_array(reduce_tensor(mask_acc.max(1)[0], reduction))
    result_dict["mask_acc"] = mask_acc
    return result_dict


def test_acc(
    model,
    args,
    dataloader,
    device,
    test_aspects=None,
    CONCEPTS=None,
    OPERATORS=None,
    reduction="mean",
    suffix="",
):
    """Test the accuracy of the prediction of mask and c_repr.
    Always assume that the pos_img is given.
    """
    is_training = model.training
    model.eval()
    uncommunitable_relation = ['IsInside']

    if test_aspects is None:
        test_aspects = ["mask|c_repr", "c_repr|mask", "both|c", "both"]
    acc_dict = {}

    for test_aspect in test_aspects:
        if test_aspect == "mask|c_repr":
            # Given true c_repr, compute acc on mask: per pixel acc, c_repr
            args_core = deepcopy(args)
            args_core.ebm_target = "mask"
            mask_acc_list = []
            neg_out_list = []
            if args.is_two_branch:
                mask_acc_list_0 = []
                mask_acc_list_1 = []
            for i, pos_data in enumerate(dataloader):
                if args.rescaled_size != "None":
                    pos_data = rescale_data(pos_data, rescaled_size=args.rescaled_size, rescale_mode=args.rescale_mode)
                if args.transforms_pos != "None":
                    pos_data = transform_pos_data(pos_data, args.transforms_pos, color_avail=args.color_avail)
                pos_img, pos_mask, pos_id, pos_info = pos_data
                pos_img = to_device_recur(pos_img, device)
                pos_repr = id_to_tensor(pos_id, CONCEPTS=CONCEPTS, OPERATORS=OPERATORS).to(device)
                (_, neg_mask, _, _, _, _), info = neg_mask_sgd(model, pos_img, c_repr=pos_repr, args=args_core)
                neg_out = info["neg_out_list"][-1]  # [B]
                neg_out_list.append(reduce_tensor(neg_out, reduction))

                result_dict = test_acc_mask(
                    pos_data=pos_data,
                    args=args_core,
                    neg_mask=neg_mask,
                    reduction=reduction,
                    CONCEPTS=CONCEPTS,
                    OPERATORS=OPERATORS,
                    device=device,
                )
                mask_acc_list.append(result_dict["mask_acc"])
                if args.is_two_branch:
                    mask_acc_list_0.append(result_dict["mask_acc_0"])
                    mask_acc_list_1.append(result_dict["mask_acc_1"])
            if reduction == "mean":
                acc_dict["iou:mask|c_repr"+suffix] = np.mean(mask_acc_list)
                acc_dict["E:neg|c_repr"+suffix] = np.mean(neg_out_list)
            elif reduction == "none":
                acc_dict["iou:mask|c_repr"+suffix] = np.concatenate(mask_acc_list)
                acc_dict["E:neg|c_repr"+suffix] = np.concatenate(neg_out_list)
            if args.is_two_branch:
                if reduction == "mean":
                    acc_dict["iou:mask_0|c_repr"+suffix] = np.mean(mask_acc_list_0)
                    acc_dict["iou:mask_1|c_repr"+suffix] = np.mean(mask_acc_list_1)
                elif reduction == "none":
                    acc_dict["iou:mask_0|c_repr"+suffix] = np.concatenate(mask_acc_list_0)
                    acc_dict["iou:mask_1|c_repr"+suffix] = np.concatenate(mask_acc_list_1)
            del neg_mask

        elif test_aspect == "c_repr|mask":
            # Given true mask, compute acc on c_repr:
            repr_acc_list = []
            pos_out_list = []
            emp_out_list = []
            for i, pos_data in enumerate(dataloader):
                if args.rescaled_size != "None":
                    pos_data = rescale_data(pos_data, rescaled_size=args.rescaled_size, rescale_mode=args.rescale_mode)
                if args.transforms_pos != "None":
                    pos_data = transform_pos_data(pos_data, args.transforms_pos, color_avail=args.color_avail)
                pos_img, pos_mask, pos_id, pos_info = pos_data
                length = len(pos_id)
                pos_img, pos_mask = to_device_recur([pos_img, pos_mask], device)
                pos_repr = id_to_tensor(pos_id, CONCEPTS=CONCEPTS, OPERATORS=OPERATORS).to(device)
                pos_out = model(pos_img, mask=pos_mask, c_repr=pos_repr)
                pos_out_list.append(to_np_array(pos_out.squeeze()))
                emp_mask = tuple(torch.zeros_like(mask_ele) for mask_ele in pos_mask)
                emp_out = model(pos_img, mask=emp_mask, c_repr=pos_repr)
                emp_out_list.append(to_np_array(emp_out.squeeze()))
                c_repr_energy = []
                for j in range(len(args.concept_collection)):
                    # args.concept_collection is the collection of concept id that is parsed from input
                    c_repr = id_to_tensor([get_c_core(args.concept_collection[j])] * length, CONCEPTS=CONCEPTS, OPERATORS=OPERATORS).to(device) # len 4
                    neg_energy = model(pos_img, mask=pos_mask, c_repr=c_repr)
                    c_repr_energy.append(neg_energy)
                c_repr_energy = torch.cat(c_repr_energy, 1)
                c_repr_argmin = to_np_array(c_repr_energy.argmin(1))

                pos_repr_int = np.array([get_c_core(args.concept_collection).index(pos_id[k]) for k in range(length)])
                repr_acc = reduce_tensor((c_repr_argmin == pos_repr_int), reduction)
                repr_acc_list.append(repr_acc)

            if reduction == "mean":
                acc_dict["acc:c_repr|mask"+suffix] = np.mean(repr_acc_list)
                acc_dict["E:pos"+suffix] = np.mean(pos_out_list)
                acc_dict["E:emp"+suffix] = np.mean(emp_out_list)
            elif reduction == "none":
                acc_dict["acc:c_repr|mask"+suffix] = np.concatenate(repr_acc_list)
                acc_dict["E:pos"+suffix] = np.concatenate(pos_out_list)
                acc_dict["E:emp"+suffix] = np.concatenate(emp_out_list)
            acc_dict["E:pos_std"+suffix] = np.std(pos_out_list)
            acc_dict["E:emp_std"+suffix] = np.std(emp_out_list)

        elif test_aspect == "both|c":
            # For each c_repr, sgd -> mask and energy, find the lowest mask, compute acc on mask and c_repr
            args_core = deepcopy(args)
            args_core.ebm_target = "mask"
            c_repr_min_list = []
            repr_acc_list = []
            mask_acc_list = []
            if args.is_two_branch:
                mask_acc_list_0 = []
                mask_acc_list_1 = []

            for i, pos_data in enumerate(dataloader):
                if args.rescaled_size != "None":
                    pos_data = rescale_data(pos_data, rescaled_size=args.rescaled_size, rescale_mode=args.rescale_mode)
                if args.transforms_pos != "None":
                    pos_data = transform_pos_data(pos_data, args.transforms_pos, color_avail=args.color_avail)
                pos_img, pos_mask, pos_id, pos_info = pos_data
                pos_all_masks = get_all_masks(pos_img, pos_id, pos_info, args)
                pos_img, pos_mask, pos_all_masks = to_device_recur([pos_img, pos_mask, pos_all_masks], device)
                length = len(pos_id)

                c_repr_energy = []
                c_repr_masks = []
                for j in range(len(args.concept_collection)):
                    # args.concept_collection is the collection of concept id that is parsed from input
                    c_repr = id_to_tensor([get_c_core(args.concept_collection[j])] * length, CONCEPTS=CONCEPTS, OPERATORS=OPERATORS).to(device) # len 4
                    (_, c_repr_mask, _, _, _, _), info = neg_mask_sgd(model, pos_img, c_repr=c_repr, args=args_core)
                    neg_energy = info["neg_out_list"][-1]
                    c_repr_masks.append(c_repr_mask)
                    c_repr_energy.append(neg_energy)
                c_repr_energy = np.stack(c_repr_energy, 1)  # [batch_size, n_concepts]
                c_repr_min_list.append(reduce_tensor(c_repr_energy.min(1), reduction))
                c_repr_argmin = c_repr_energy.argmin(1)  # [batch_size]
                c_repr_argmin_onehot = torch.LongTensor(np.eye(len(args.concept_collection))[c_repr_argmin]).bool()
                
                if args.is_two_branch:
                    c_repr_masks_core = Zip(*c_repr_masks, function=lambda x: torch.stack(x, 1))  # ( (b, n_cs, c, h, w), (b, n_cs, c, h, w))
                    c_repr_masks_best = (c_repr_masks_core[0][c_repr_argmin_onehot], c_repr_masks_core[1][c_repr_argmin_onehot])  # ((b, c, h, w), (b, c, h, w))
                else:
                    c_repr_masks_core = Zip(*c_repr_masks, function=lambda x: torch.stack(x, 1))
                    c_repr_masks_best = (c_repr_masks_core[0][c_repr_argmin_onehot],)

                result_dict = test_acc_mask(
                    pos_data=pos_data,
                    args=args_core,
                    neg_mask=c_repr_masks_best,
                    reduction=reduction,
                    CONCEPTS=CONCEPTS,
                    OPERATORS=OPERATORS,
                    device=device,
                )
                mask_acc_list.append(result_dict["mask_acc"])
                if args.is_two_branch:
                    mask_acc_list_0.append(result_dict["mask_acc_0"])
                    mask_acc_list_1.append(result_dict["mask_acc_1"])

                # Calculate the accuracy for repr:
                pos_repr_int = np.array([get_c_core(args.concept_collection).index(pos_id[k]) for k in range(length)])
                repr_acc = reduce_tensor(c_repr_argmin == pos_repr_int, reduction)
                repr_acc_list.append(repr_acc)

            if reduction == "mean":
                acc_dict["acc:c_repr|c"+suffix] = np.mean(repr_acc_list)
                acc_dict["iou:mask|c"+suffix] = np.mean(mask_acc_list)
                acc_dict["E:neg|c"+suffix] = np.mean(c_repr_min_list)
            elif reduction == "none":
                acc_dict["acc:c_repr|c"+suffix] = np.concatenate(repr_acc_list)
                acc_dict["iou:mask|c"+suffix] = np.concatenate(mask_acc_list)
                acc_dict["E:neg|c"+suffix] = np.concatenate(c_repr_min_list)
            if args.is_two_branch:
                if reduction == "mean":
                    acc_dict["iou:mask_0|c"+suffix] = np.mean(mask_acc_list_0)
                    acc_dict["iou:mask_1|c"+suffix] = np.mean(mask_acc_list_1)
                elif reduction == "none":
                    acc_dict["iou:mask_0|c"+suffix] = np.concatenate(mask_acc_list_0)
                    acc_dict["iou:mask_1|c"+suffix] = np.concatenate(mask_acc_list_1)
            del c_repr_masks_best
            del c_repr_min_list
        elif test_aspect == "both":
            # Perform SGD on both mask and c_repr, and compute the acc on both:
            test_aspect_core = "mask"
            args_core = deepcopy(args)
            args_core.ebm_target = "mask+repr"
            mask_acc_list = []
            repr_acc_list = []
            neg_out_list = []
            if args.is_two_branch:
                mask_acc_list_0 = []
                mask_acc_list_1 = []
            c_reprs = id_to_tensor(get_c_core(args.concept_collection), CONCEPTS=CONCEPTS, OPERATORS=OPERATORS).to(device)  # [n_concepts, REPR_DIM]
            for i, pos_data in enumerate(dataloader):
                if args.rescaled_size != "None":
                    pos_data = rescale_data(pos_data, rescaled_size=args.rescaled_size, rescale_mode=args.rescale_mode)
                if args.transforms_pos != "None":
                    pos_data = transform_pos_data(pos_data, args.transforms_pos, color_avail=args.color_avail)
                pos_img, pos_mask, pos_id, pos_info = pos_data
                if i == 0:
                    c_reprs = c_reprs[None].expand(len(pos_id), *c_reprs.shape)  # [batch_size, n_concepts, REPR_DIM]
                pos_repr = id_to_tensor(pos_id, CONCEPTS=CONCEPTS, OPERATORS=OPERATORS).to(device) # vector of len 4
                pos_all_masks = get_all_masks(pos_img, pos_id, pos_info, args)
                pos_img, pos_mask, pos_all_masks = to_device_recur([pos_img, pos_mask, pos_all_masks], device)

                # Now we have pos_mask and neg_mask, given c_repr. We calculate the accuracy for neg_mask:
                (_, neg_mask, neg_repr, _, _, _), info = neg_mask_sgd(model, pos_img, args=args_core)
                neg_out_list.append(to_np_array(info["neg_out_list"][-1].squeeze()))
                
                result_dict = test_acc_mask(
                    pos_data=pos_data,
                    args=args_core,
                    neg_mask=neg_mask,
                    reduction=reduction,
                    CONCEPTS=CONCEPTS,
                    OPERATORS=OPERATORS,
                    device=device,
                )
                mask_acc_list.append(result_dict["mask_acc"])
                if args.is_two_branch:
                    mask_acc_list_0.append(result_dict["mask_acc_0"])
                    mask_acc_list_1.append(result_dict["mask_acc_1"])

                # The chosen repr is the one with the smallest distance to the ground-truth:
                c_repr_argmin = to_np_array((c_reprs - neg_repr[:, None]).square().sum(-1).argmin(-1))
                pos_repr_int = np.array([get_c_core(args.concept_collection).index(id) for id in pos_id])
                repr_acc_list.append(reduce_tensor(pos_repr_int == c_repr_argmin, reduction))
            if reduction == "mean":
                acc_dict["acc:c_repr"+suffix] = np.mean(repr_acc_list)
                acc_dict["iou:mask"+suffix] = np.mean(mask_acc_list)
                acc_dict["E:neg"+suffix] = np.mean(neg_out_list)
            elif reduction == "none":
                acc_dict["acc:c_repr"+suffix] = np.concatenate(repr_acc_list)
                acc_dict["iou:mask"+suffix] = np.concatenate(mask_acc_list)
                acc_dict["E:neg"+suffix] = np.concatenate(neg_out_list)
            acc_dict["E:neg_std"+suffix] = np.std(neg_out_list)
            if args.is_two_branch:
                if reduction == "mean":
                    acc_dict["iou:mask_0"+suffix] = np.mean(mask_acc_list_0)
                    acc_dict["iou:mask_1"+suffix] = np.mean(mask_acc_list_1)
                elif reduction == "none":
                    acc_dict["iou:mask_0"+suffix] = np.concatenate(mask_acc_list_0)
                    acc_dict["iou:mask_1"+suffix] = np.concatenate(mask_acc_list_1)
            del neg_mask
            del neg_repr
        else:
            raise Exception("Oh no! The test aspect '{}' is not valid!".format(test_aspect))
    acc_dict = {key: acc_dict[key] for key in sorted(acc_dict.keys())}
    if is_training:
        model.train()
    return acc_dict


# In[ ]:


# #test commutable
# if __name__ == "__main__":
#     dirname = "/dfs/user/tailin/.results/ebm_4-30/"
#     filenames = sorted(filter_filename(dirname, include=[".p"]))
#     #print(filenames)
    
#     set_seed(seed=1)
#     # Make sure that the initialization of CONCEPTS and OPERATORS (including their embedding) is after setting the seed
#     from zeroc.concepts_shapes import OPERATORS, CONCEPTS, load_task, seperate_concept
#     #filename = "c-RotateA+RotateB+RotateC(Lshape)_cz_8_model_CEBM_alpha_1_lambd_0.005_size_20.0_sams_60_et_mask_pl_False_neg_addrand+permlabel_nco_0.1_mask_mulcat_tbm_concat_p_0.2_id_1_Hash_7iTPXjV8_turing2.p"
#     #filename = "c-Parallel+Vertical_cz_8_model_CEBM_alpha_1_las_0.1_size_20.0_sams_60_et_mask_pl_False_neg_addrand+permlabel_nco_0.2_mask_concat_tbm_concat_cm_c2_cf_1_p_0.2_id_1_Hash_1iHtZRVo_turing3.p"
#     filename ='c-IsInside+IsTouch_cz_8_model_CEBM_alpha_1_las_0.1_size_20.0_sams_60_et_mask_pl_False_neg_addrand+permlabel_nco_0.2_mask_concat_tbm_concat_cm_c1_cf_0_p_0.2_id_1_Hash_Io3xVIYa_turing1.p'
#     device = "cuda:0"
#     data_record = pickle.load(open(dirname + filename, "rb"))
#     model = load_model_energy(data_record["model_dict"][5]).to(device)
#     set_seed(seed=2)
#     args = init_args(update_default_hyperparam(data_record["args"]))
#     args.n_examples = 500
#     dataset, args = get_dataset(args)
#     dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

#     set_seed(seed=1)
#     # Make sure that the initialization of CONCEPTS and OPERATORS (including their embedding) is after setting the seed
#     zeroc.concepts_shapes import OPERATORS, CONCEPTS, load_task, seperate_concept
#     val_acc_dict = test_acc(model, args, dataloader, device, CONCEPTS=CONCEPTS, OPERATORS=OPERATORS)
#     print(val_acc_dict)


# In[ ]:


# # test 
# if __name__ == "__main__":

#     # Make sure that the initialization of CONCEPTS and OPERATORS (including their embedding) is after setting the seed
#     zeroc.concepts_shapes import OPERATORS, CONCEPTS, load_task, seperate_concept
#     concept_embeddings = {key: to_np_array(CONCEPTS[key].get_node_repr()) for key in CONCEPTS}
#     operator_embeddings = {key: to_np_array(OPERATORS[key].get_node_repr()) for key in OPERATORS}

#     concept_embeddings.update(operator_embeddings)    
#     #device = get_device(args) 
#     device = 'cpu'
#     # Get dataset and dataloader:
#     # if is_two_branch is True, will use for operators, in which the image x and mask a 
#     #    fed to the EBM E(x; a; c) will actually be a tuple of x=(x_in, x_out) and mask=(mask_in, mask_out):


#     dirname = '/dfs/user/xyang23/results/ebm_4-4_model/'
#     filename = 'c-Parallel+Vertical_model_CEBM_alpha_1_lambd_0.005_size_50.0_sams_60_mask_mulcat_p_0.5_id_0_Hash_V6a3+T6S_turing3.p'
#     #filename = 'c-Lshape_model_CEBM_alpha_1_lambd_0.005_size_50.0_sams_60_mask_concat_p_0.2_id_0_Hash_7UaCu0QQ_turing3.p'

#     data_record = pickle.load(open(dirname + filename, "rb"))
#     args = init_args(update_default_hyperparam(data_record["args"]))
#     args.n_examples = 400
#     dataset, args = get_dataset(args)
#     args.n_examples = 400
#     dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, drop_last=True)

#     args.is_two_branch = isinstance(dataset[0][0], tuple)
#     print(args.concept_collection)
#     pp.pprint(args.__dict__)
#     model = load_model_energy(data_record["model_dict"][20], device=device)
#     test = test_acc_operator(model, args, dataloader, test_aspects=None)
#     print(test)


# ## 4. Training:

# In[ ]:


if __name__ == "__main__":
    # Obtain args and setting specific args for jupyter for easy testing:
    args = get_args_EBM()
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
        get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=4,5')
        # Experiment management:
#         args.exp_id = "ebm_test"
#         args.date_time = "{}-{}".format(datetime.now().month, datetime.now().day)
#         args.inspect_interval = 5
#         args.save_interval = 20
#         args.gpuid = "5"
#         args.seed = 1
#         args.id = "1"      # For differentiating experiments with different settings
#         # args.batch_size = 16

#         # Dataset:
#         # args.dataset = "c-Parallel+VerticalMid+VerticalEdge"
#         # args.dataset = "c-SameShape+SameColor+IsInside(Line+Rect+RectSolid+Lshape)"
# #         args.dataset = "c-arc^Line+RectSolid"  # choose from c-{}, cifar10. E.g., c-Line, c-Rect, c-Line+Rect, c-arc^Line, c-arc^Line+Rect
# #         args.dataset = "c-Parallel+Vertical"
# #         args.dataset = "y-Parallel+Vertical"
#         args.dataset = "c-Line"
#         args.dataset = "c-IsNonOverlapXY+IsInside+IsEnclosed(Rect[4,16]+Randshape[3,8]+Lshape[3,10]+Tshape[3,10])"
#         # args.dataset = "c-Rect[4,16]+Eshape[5,12]"
#         # args.dataset = "c-Eshape[5,12]+Rect[4,16]"
#         # args.dataset = "c-IsInside+SameColor(Rect+Lshape+Tshape)"
#         # args.dataset = "y-Line"
#         args.seed = 1
# #         args.dataset = "c-RotateA+RotateB+RotateC+hFlip+vFlip+DiagFlipA+DiagFlipB(Lshape+Line+Rect)"  # Full operator
#         # args.dataset = "c-Hshape+Lshape"
# #         args.dataset = "c-Image"
# #         args.dataset = "c-arc^RotateA+RotateB()"
# #         args.dataset = "cifar10"
# #         args.dataset="h-c^(1,2):Eshape+Ashape-d^1"
#         args.n_examples = 400   # Use 400 for quick testing, use 10000 for running actual experiments
#         args.rainbow_prob = 0.  # Probability of using rainbow color for BabyARC datatset
#         args.max_n_distractors = 0 #-1  # The maximum number of distractors besides objects related to the core concept.
#         args.color_avail = "1,2"
#         args.batch_size = 40
#         args.canvas_size = 16
#         # args.rescaled_size = "32,32"
        
#         # 3D dataset generation
#         args.num_processes_3d = 20

#         # Model setting:
#         # args.model_type = "CEBMLarge"  # "CEBM" (for all other args.dataset), "CEBMLarge", "IGEBM" (only for cifar10)
#         args.model_type = "CEBM"
# #         args.model_atom = "Line^Vertical+Parallel^AdaptRe"
#         args.w_type = "image+mask"
#         args.mask_mode = "concat"
#         args.channel_base = 64
#         args.two_branch_mode = "concat"
#         args.c_repr_mode = "c2"
#         args.c_repr_first = 2
#         args.c_repr_base = 2
#         args.z_mode = "None"
#         args.z_first = 2
#         args.z_dim = 4
#         args.n_workers = 0
#         args.aggr_mode = "max"
#         args.normalization_type = "gn-2"
#         args.is_spec_norm = "True"
#         args.dropout = 0
#         args.self_attn_mode = "None"
#         args.last_act_name = "None"
# #         args.transforms = "None"
#         args.transforms = "color+flip+rotate+resize:0.7"
#         args.transforms_pos = "randpatch"#"None" # "color+flip+rotate:0.5"

#         # EBM training setting:
#         args.train_mode = "cd"
#         # args.energy_mode = "standard+center^stopsq:0.1"
#         args.kl_all_step = False
#         args.kl_coef = 1
#         args.entropy_coef_mask = 0
#         args.entropy_coef_repr = 0
#         args.entropy_coef_img = 0
#         args.pos_consistency_coef = 0.1
#         args.neg_consistency_coef = 0.1
#         args.SGLD_mutual_exclusive_coef = 0
#         args.epsilon_ent = 1e-5
#         args.ebm_target_mode = "r-rmbx"
#         args.ebm_target = "mask+repr"
#         args.is_pos_repr_learnable = False
#         args.neg_mode = "addallrand+delrand"  # Choose from "None" (default), "addrand", "delrand", "addallrand", "permlabel"
#         args.neg_mode_coef = 0.2
#         args.lambd_start = 0.1
#         args.step_size = 20
#         args.step_size_repr = 2
#         args.step_size_img = -1
#         args.sample_step = 60
#         args.p_buffer = 0.2
#         args.epochs = 500
#         args.early_stopping_patience = 5
        
        
        # Using "u-"
        args.exp_id = "ebm_test"
        # args.dataset = "u-concept-Red+Green+Blue+Cube+Cylinder+Large+Small"
        args.dataset = "u-relation-SameColor+SameShape+SameSize"
        args.color_avail = "1,2"
        args.n_examples = 400
        args.canvas_size = 64
        args.train_mode = "cd"
        args.model_type = "CEBM"
        args.mask_mode = "concat"
        args.c_repr_mode = "c2"
        args.c_repr_first = 2
        args.kl_coef = 1
        args.entropy_coef_mask = 0
        args.entropy_coef_repr = 0
        args.ebm_target = "mask"
        args.ebm_target_mode = "r-rmbx"
        args.is_pos_repr_learnable = False
        args.sample_step = 60
        args.step_size = 30
        args.step_size_repr = 2
        args.lambd_start = 0.1
        args.p_buffer=0.2
        args.channel_base = 128
        args.two_branch_mode = "concat"
        args.neg_mode = "addallrand+delrand+permlabel"
        args.aggr_mode = "max"
        args.neg_mode_coef = 0.2
        args.epochs=200
        args.early_stopping_patience = -1
        args.n_workers = 0
        args.seed = 1
        args.self_attn_mode = "None"
        args.transforms = "color+flip+rotate+resize:0.5"
        args.pos_consistency_coef = 0.1
        args.is_res = True
        args.lr = 1e-4
        args.gpuid = "5"
        args.id = "0"
        args.act_name = "leakyrelu"
        args.rescaled_size = "32,32"
        args.rescale_mode = "nearest"
        args.energy_mode = "standard+center^stop:0.1"
        args.emp_target_mode = "r-mb"
        args.batch_size = 40
        args.max_num_occur = 20
        args.parallel_mode = "None"
        is_jupyter = True
    except:
        is_jupyter = False
    if args.step_size_img == -1:
        args.step_size_img = args.step_size
    if args.step_size_repr == -1:
        args.step_size_repr = args.step_size
    if args.step_size_z == -1:
        args.step_size_z = args.step_size

    set_seed(args.seed)
    # Make sure that the initialization of CONCEPTS and OPERATORS (including their embedding) is after setting the seed
    from zeroc.concepts_shapes import OPERATORS, CONCEPTS, load_task, seperate_concept
    from zeroc.concepts_shapes import SameShape, SameColor, SameAll, SameRow, SameCol, SubsetOf, IsInside, IsNonOverlapXY
    concept_embeddings = get_concept_embeddings(CONCEPTS, OPERATORS)

    # short_str_dict specifies the options to appear in the filename in their short name:
    short_str_dict = {
        "dataset": "",
        "canvas_size": "cz",
        "model_type": "model",
        "alpha": "alpha",
        "lambd_start": "las",
        "step_size": "size",
        "sample_step": "sams",
        "ebm_target_mode": "e",
        "ebm_target": "et",
        "is_pos_repr_learnable": "pl",
        "train_mode": "tm",
        "w_type": "w",
        "neg_mode_coef": "nco",
        "mask_mode": "mask",
        "two_branch_mode": "tbm",
        "c_repr_mode": "cm",
        "c_repr_first": "cf",
        "p_buffer": "p",
        "id": "id",
    }
    if len(args.dataset) > 60:
        short_str_dict.pop("dataset", None)
    _, filename = get_filename(args, short_str_dict, is_local_path=is_jupyter)
    # Get dataset and dataloader:
    dataset, args = get_dataset(args, n_examples=int(args.n_examples*1.1), is_load=True, is_rewrite=args.is_rewrite)
    n_train = int(len(dataset)*10/11)
    if is_jupyter:
        train_dataset = dataset[:400]
        val_dataset = dataset[400:440]
    else:
        train_dataset = dataset[:n_train]
        val_dataset = dataset[n_train:]
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=Batch(is_collate_tuple=True).collate(), num_workers=args.n_workers, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=Batch(is_collate_tuple=True).collate(), num_workers=args.n_workers, drop_last=True)
    # if is_two_branch is True, will use for operators, in which the image x and mask a 
    #    fed to the EBM E(x; a; c) will actually be a tuple of x=(x_in, x_out) and mask=(mask_in, mask_out):
    data_ex = dataset[0]
    args.is_two_branch = len(data_ex[1]) > 1
    args.is_image_tuple = isinstance(data_ex[0], tuple) or isinstance(data_ex[0], list)
    early_stopping = Early_Stopping(patience=args.early_stopping_patience, mode="max")

    # Load model. (IBGBM is original model Implicit Generation and Modeling with Energy-Based Models (Du and Mordatch, 2019),
    #   and ConceptEBM is our model that learns masks instead of images):
    model = get_model_energy(args)
    model, device = model_parallel(model, args)

    # Unittest to make sure that the loaded model is exactly the same with the original model:
    unittest(model, args, device, CONCEPTS=CONCEPTS, OPERATORS=OPERATORS)
    # Buffer contains 10000 examples of negative examples to provide as initial condition to perform gradient step
    #   (usually 95% from buffer, 5% from unit Gaussian noise):
    buffer = SampleBuffer_Conditional(is_two_branch=args.is_two_branch)

    all_params = [{'params': model.parameters()}]
    if args.is_pos_repr_learnable:
        all_params.append({"params": [CONCEPTS[key].get_node_repr() for key in CONCEPTS] + [OPERATORS[key].get_node_repr() for key in OPERATORS]})
    optimizer = torch.optim.Adam(all_params, lr=args.lr, betas=(0.0, 0.999))

    data_record = {"args": args.__dict__, "concept_embeddings": [concept_embeddings], "repr_epoch": [-1], "acc": {}}
    pp.pprint(args.__dict__)
    for epoch in range(args.epochs+1):
        # Initialize the recording list:
        loss_list = []
        if args.train_mode == "cd":
            pos_out_list = []
            neg_out_list = []
            loss_core_info_dict = {"L2": []}
            if "mid" in args.energy_mode:
                emp_out_list = []
                loss_core_info_dict["mid"] = []
            if "center" in args.energy_mode:
                emp_out_list = []
                loss_core_info_dict["center"] = []
            if "standard" in args.energy_mode:
                loss_core_info_dict["standard"] = []
            if "margin" in args.energy_mode:
                loss_core_info_dict["margin"] = []
            if args.neg_mode_coef > 0 and args.neg_mode != "None":
                neg_out_gen_list = []
                neg_out_gen_valid_list = []
            if args.pos_consistency_coef > 0:
                loss_pos_consistency_list = []
            if args.neg_consistency_coef > 0:
                loss_neg_consistency_list = []
            if args.emp_consistency_coef > 0:
                loss_emp_consistency_list = []
        elif args.train_mode == "sl":
            loss_img_list = []
            loss_mask_list = []
            loss_repr_list = []
        else:
            raise

        for i, pos_data_ori in enumerate(train_loader):
            if is_diagnose(loc="ebm:0", filename=filename):
                pdb.set_trace()
            args = get_ebm_target(args)  # get ebm_target

            # Sample negative examples (negative image for IGEBM and negative mask for CEBM,
            #    args.p_buffer fraction is from buffer, rest from unit Gaussian):
            if args.rescaled_size != "None":
                pos_data_ori = rescale_data(pos_data_ori, rescaled_size=args.rescaled_size, rescale_mode=args.rescale_mode)
            if args.transforms_pos != "None":
                pos_data = transform_pos_data(pos_data_ori, args.transforms_pos, color_avail=args.color_avail)
            else:
                pos_data = pos_data_ori
            neg_data = sample_buffer_conditional(
                buffer,
                pos_data=pos_data,
                ebm_target=args.ebm_target,
                in_channels=args.in_channels,
                image_size=args.image_size if args.rescaled_size == "None" else eval(args.rescaled_size),
                batch_size=args.batch_size,
                is_two_branch=args.is_two_branch,
                w_type=args.w_type,
                p=args.p_buffer,
                transforms=args.transforms,
                color_avail=args.color_avail,
                device=device,
            )

            if args.is_mask:
                pos_img, pos_mask, pos_id, pos_info = pos_data
                neg_img, neg_mask, neg_repr, neg_info = neg_data
                # if args.ebm_target == "mask":
                #     visualize_matrices([neg_img[0].argmax(0)])
                #     plot_matrices([neg_mask[0][0][0]])
                pos_repr = id_to_tensor(pos_id, CONCEPTS=CONCEPTS, OPERATORS=OPERATORS,
                                        requires_grad=args.is_pos_repr_learnable).to(device)
                # Make sure that neg_mask can be perform gradient descent on:
                pos_img, pos_mask = to_device_recur([pos_img, pos_mask], device)

                if args.ebm_target in ["image+mask"]:
                    if args.is_image_tuple:
                        # Operator:
                        neg_img[1].requires_grad = True
                    else:
                        # Concept or Relation:
                        neg_img.requires_grad = True
                else:
                    neg_img = deepcopy(pos_img)
                if args.ebm_target in ["mask", "mask+repr", "image+mask"]:
                    for k in range(model.mask_arity):
                        neg_mask[k].requires_grad = True
                else:
                    neg_mask = deepcopy(tuple(ele.detach() for ele in pos_mask))
                if args.ebm_target in ["repr", "mask+repr"]:
                    neg_repr = neg_repr.to(device)
                    neg_repr.requires_grad = True
                else:
                    neg_repr = deepcopy(pos_repr.detach())

                if args.train_mode == "cd":
                    # The for loop below obtains the negative examples (image/mask) using current EBM,
                    #   using gradient descent on fix model E(x; a; c) w.r.t. mask a:
                    if is_diagnose(loc="ebm:1", filename=filename):
                        pdb.set_trace()
                    if args.kl_coef == 0:
                        (neg_img, neg_mask, neg_repr, _, _, _), info = neg_mask_sgd(
                            model, neg_img, neg_mask=neg_mask, c_repr=neg_repr, args=args)
                    else:
                        (neg_img, neg_mask, neg_repr, _, _, _), (neg_img_kl, neg_mask_kl, neg_repr_kl, _, _, _), info = neg_mask_sgd_with_kl(
                            model, neg_img, neg_mask=neg_mask, c_repr=neg_repr, args=args)

                    # After obtaining negative examples, now perform gradient descent on fixed pos and neg examples w.r.t model parameter of EBM:
                    requires_grad(model.parameters(), True)
                    model.train()
                    optimizer.zero_grad()

                    pos_out = model(pos_img, mask=pos_mask, c_repr=pos_repr)
                    neg_out = model(neg_img, mask=neg_mask, c_repr=neg_repr)
                    if "mid" in args.energy_mode or "center" in args.energy_mode:
                        emp_mask = tuple(torch.zeros_like(mask_ele) for mask_ele in pos_mask)
                        emp_out = model(pos_img, mask=emp_mask, c_repr=pos_repr)
                    else:
                        emp_out = None
                    if args.neg_mode_coef > 0 and args.neg_mode != "None" and args.ebm_target not in ["image+mask"]:
                        neg_mask_gen, neg_repr_gen, neg_gen_valid = generate_neg_examples(pos_img, pos_mask, pos_repr, neg_mode=args.neg_mode)
                        neg_out_gen = model(pos_img, mask=neg_mask_gen, c_repr=neg_repr_gen)
                    else:
                        neg_out_gen = None

                    # The loss consists of L2 regularization on both pos and neg examples (to make the energy landscape less steep) and pos_out - neg_out:
                    loss = args.alpha * (pos_out ** 2 + neg_out ** 2)
                    loss_core, loss_core_info = get_loss_core(
                        args=args,
                        pos_out=pos_out,
                        neg_out=neg_out,
                        emp_out=emp_out,
                        neg_out_gen=neg_out_gen,
                    )
                    loss_core_info["L2"] = loss.mean().item()
                    loss = loss + loss_core
                    if args.neg_mode_coef > 0 and args.neg_mode != "None" and args.ebm_target not in ["image+mask"]:
                        loss = loss + args.neg_mode_coef * neg_gen_valid * (args.alpha * neg_out_gen ** 2 - neg_out_gen)

                    if args.kl_coef > 0:
                        # KL loss:
                        requires_grad(model.parameters(), False)
                        loss_kl = model(neg_img_kl if neg_img_kl is not None else neg_img,
                                        mask=neg_mask_kl if neg_mask_kl is not None else neg_mask,
                                        c_repr=neg_repr_kl if neg_repr_kl is not None else neg_repr)
                        requires_grad(model.parameters(), True)
                        loss = loss + loss_kl * args.kl_coef

                        if (args.entropy_coef_mask > 0 or args.entropy_coef_repr > 0 or args.entropy_coef_img > 0) and args.ebm_target in ["mask", "repr", "mask+repr", "image+mask"] and len(buffer) > 1000:
                            neg_img_compare, neg_mask_compare, neg_repr_compare, _ = sample_buffer_conditional(
                                buffer,
                                pos_data=pos_data,
                                ebm_target=args.ebm_target,
                                in_channels=args.in_channels,
                                image_size=args.image_size if args.rescaled_size == "None" else eval(args.rescaled_size),
                                batch_size=100,
                                is_two_branch=args.is_two_branch,
                                w_type=args.w_type,
                                p=1.,
                                device=device,
                            )

                        if args.entropy_coef_img > 0 and args.ebm_target in ["image+mask"] and len(buffer) > 1000:
                            neg_img_flat = torch.clamp(neg_img_kl.view(args.batch_size, -1), 0, 1) if not args.is_image_tuple else torch.clamp(neg_img_kl[1].view(args.batch_size, -1), 0, 1)
                            neg_img_compare_flat = neg_img_compare.view(100, -1) if not args.is_image_tuple else neg_img_compare[1].view(100, -1)
                            dist_matrix_img = torch.norm(neg_img_flat[:,None,:] - neg_img_compare_flat[None,:,:], p=2, dim=-1)
                            loss_entropy_img = -torch.log(dist_matrix_img.min(dim=1, keepdims=True)[0] + args.epsilon_ent)
                            loss = loss + loss_entropy_img * args.entropy_coef_img
                        else:
                            loss_entropy_img = torch.zeros(args.batch_size, 1)

                        if args.entropy_coef_mask > 0 and args.ebm_target in ["mask", "mask+repr", "image+mask"] and len(buffer) > 1000:
                            neg_mask_flat = tuple(torch.clamp(neg_mask_kl[k].view(args.batch_size, -1), 0, 1) for k in range(model.mask_arity))
                            neg_mask_compare_flat = tuple(neg_mask_compare[k].view(100, -1) for k in range(model.mask_arity))
                            dist_matrix_mask = tuple(torch.norm(neg_mask_flat[k][:,None,:] - neg_mask_compare_flat[k][None,:,:], p=2, dim=-1) for k in range(model.mask_arity))
                            loss_entropy_mask = torch.cat([-torch.log(dist_matrix_mask[k].min(dim=1, keepdims=True)[0] + args.epsilon_ent) for k in range(model.mask_arity)], -1).mean(-1, keepdims=True)
                            loss = loss + loss_entropy_mask * args.entropy_coef_mask
                        else:
                            loss_entropy_mask = torch.zeros(args.batch_size, 1)

                        if args.entropy_coef_repr > 0 and args.ebm_target in ["repr", "mask+repr"] and len(buffer) > 1000:
                            if "softmax" not in args.c_repr_mode:
                                neg_repr_flat = torch.clamp(neg_repr_kl, 0, 1)
                            else:
                                neg_repr_flat = neg_repr_kl
                            neg_repr_compare_flat = neg_repr_compare
                            dist_matrix_repr = torch.norm(neg_repr_flat[:,None,:] - neg_repr_compare_flat[None,:,:], p=2, dim=-1)
                            loss_entropy_repr = -torch.log(dist_matrix_repr.min(dim=1, keepdims=True)[0] + args.epsilon_ent)
                            loss = loss + loss_entropy_repr * args.entropy_coef_repr
                        else:
                            loss_entropy_repr = torch.zeros(args.batch_size, 1)

                    loss = loss.mean()
                    if args.pos_consistency_coef > 0:
                        loss_pos_consistency = pos_out.std() * args.pos_consistency_coef
                        loss = loss + loss_pos_consistency
                        loss_pos_consistency_list.append(loss_pos_consistency.item())
                    if args.neg_consistency_coef > 0:
                        loss_neg_consistency = neg_out.std() * args.neg_consistency_coef
                        loss = loss + loss_neg_consistency
                        loss_neg_consistency_list.append(loss_neg_consistency.item())
                    if args.emp_consistency_coef > 0:
                        loss_emp_consistency = emp_out.std() * args.emp_consistency_coef
                        loss = loss + loss_emp_consistency
                        loss_emp_consistency_list.append(loss_emp_consistency.item())

                    # Record:
                    pos_out_list.append(to_np_array(pos_out.mean()))
                    neg_out_list.append(to_np_array(neg_out.mean()))
                    if "mid" in args.energy_mode or "center" in args.energy_mode:
                        emp_out_list.append(to_np_array(emp_out.mean()))
                    for key in loss_core_info:
                        loss_core_info_dict[key].append(loss_core_info[key])
                    loss_list.append(to_np_array(loss.mean()))
                    if args.neg_mode_coef > 0 and args.neg_mode != "None" and args.ebm_target not in ["image+mask"]:
                        neg_out_gen_valid_list.append(to_np_array(neg_gen_valid).mean())
                        neg_out_gen_list.append(to_np_array(neg_gen_valid * neg_out_gen).mean())

                elif args.train_mode == "sl":
                    _, (neg_img_kl, neg_mask_kl, neg_repr_kl, _, _, _), info = neg_mask_sgd_with_kl(
                        model, neg_img, neg_mask, c_repr=neg_repr, args=args)

                    # After obtaining negative examples, now perform gradient descent on fixed pos and neg examples w.r.t model parameter of EBM:
                    requires_grad(model.parameters(), True)
                    model.train()
                    optimizer.zero_grad()

                    loss = 0
                    if args.ebm_target in ["image+mask"]:
                        if args.is_image_tuple:
                            loss_img = loss_op_core(neg_img_kl[1], pos_img[1], loss_type=args.supervised_loss_type)
                        else:
                            loss_img = loss_op_core(neg_img_kl, pos_img, loss_type=args.supervised_loss_type)
                        loss = loss + loss_img
                        loss_img_list.append(to_np_array(loss_img))
                    if args.ebm_target in ["mask", "mask+repr", "image+mask"]:
                        loss_mask = torch.stack([loss_op_core(neg_mask_kl[kk], pos_mask[kk], loss_type=args.supervised_loss_type) for kk in range(len(pos_mask))]).mean()
                        loss = loss + loss_mask
                        loss_mask_list.append(to_np_array(loss_mask))
                    if args.ebm_target in ["repr", "mask+repr"]:
                        loss_repr = loss_op_core(neg_repr_kl, pos_repr, loss_type=args.supervised_loss_type)
                        loss = loss + loss_repr
                        loss_repr_list.append(to_np_array(loss_repr))

                    # Record:
                    loss_list.append(to_np_array(loss.mean()))
                else:
                    raise Exception("train_mode {} is not valid!".format(args.train_mode))

                # Optimization:
                loss.backward()
                clip_grad(optimizer)
                optimizer.step()

                # Put the negative example (pos_img, neg_mask, pos_id) into the buffer:
                buffer.push(imgs=neg_img, masks=neg_mask, c_reprs=neg_repr, class_ids=pos_id, infos=pos_info, ebm_target=args.ebm_target)
            else:
                pos_img, pos_id = pos_data_ori
                pos_img, pos_id = pos_img.to(device), pos_id.to(device)
                neg_img, neg_id = neg_data
                neg_img.requires_grad = True
                requires_grad(model.parameters(), False)
                model.eval()

                if args.lambd_start == -1:
                    args.lambd_start = args.lambd
                lambd_list = args.lambd + 1/2 * (args.lambd_start - args.lambd) * (1 + torch.cos(torch.arange(args.sample_step)/args.sample_step * np.pi))
                if args.step_size_start == -1:
                    args.step_size_start = args.step_size
                step_size_list = args.step_size + 1/2 * (args.step_size_start - args.step_size) * (1 + torch.cos(torch.arange(args.sample_step)/args.sample_step * np.pi))

                for k in range(args.sample_step):
                    if noise.shape[0] != neg_img.shape[0]:
                        noise = torch.randn(neg_img.shape[0], args.in_channels, *(args.image_size if args.rescaled_size == "None" else eval(args.rescaled_size)), device=device)

                    noise.normal_(0, lambd_list[k])
                    neg_img.data.add_(noise.data)

                    neg_out = model(neg_img, neg_id)
                    neg_out.sum().backward()
                    neg_img.grad.data.clamp_(-0.01, 0.01)

                    neg_img.data.add_(neg_img.grad.data, alpha=-step_size_list[k])

                    neg_img.grad.detach_()
                    neg_img.grad.zero_()

                    neg_img.data.clamp_(0, 1)

                neg_img = neg_img.detach()

                requires_grad(model.parameters(), True)
                model.train()

                model.zero_grad()

                pos_out = model(pos_img, pos_id)
                neg_out = model(neg_img, neg_id)

                loss = args.alpha * (pos_out ** 2 + neg_out ** 2)
                loss = loss + (pos_out - neg_out)
                loss = loss.mean()
                loss.backward()

                clip_grad(optimizer)

                optimizer.step()

                buffer.push(neg_img, class_ids=neg_id)

                pos_out_list.append(to_np_array(pos_out.mean()))
                neg_out_list.append(to_np_array(neg_out.mean()))
                loss_list.append(to_np_array(loss.mean()))

        # Epoch ends, record:
        if args.train_mode == "cd":
            record_data(data_record, [epoch, np.mean(pos_out_list), np.mean(neg_out_list), np.mean(loss_list)],
                        ["epoch", "E:pos:train", "E:neg|c_repr:train", "E:train"])
            if "mid" in args.energy_mode or "center" in args.energy_mode:
                record_data(data_record, [np.mean(emp_out_list)], ["E:emp:train"])
            loss_core_info_dict = transform_dict(loss_core_info_dict, "mean")
            record_data(data_record, list(loss_core_info_dict.values()), ["{}:train".format(key) for key in loss_core_info_dict])
            p.print("epoch {}: pos: {:.6f}     neg: {:.6f}".format(
                    str(epoch).zfill(3), np.mean(pos_out_list), np.mean(neg_out_list)), end="")
            if "mid" in args.energy_mode or "center" in args.energy_mode:
                p.print("     emp: {:.6f}".format(np.mean(emp_out_list)), is_datetime=False, end="")
            p.print("     loss: {:.6f}".format(np.mean(loss_list)), is_datetime=False, end="")
            for key in loss_core_info_dict:
                p.print("     {}: {:.6f}".format(key, loss_core_info_dict[key]), is_datetime=False, end="")
            if args.neg_mode_coef > 0 and args.neg_mode != "None":
                if args.ebm_target not in ["image+mask"]:
                    record_data(data_record, [np.mean(neg_out_gen_valid_list), np.mean(neg_out_gen_list)], ["E:neg_gen_valid:train", "E:neg_gen:train"])
                    p.print("        neg_gen: {:.6f}     neg_gen_valid: {:.3f}".format(np.mean(neg_out_gen_list), np.mean(neg_out_gen_valid_list)), is_datetime=False, end="")
                else:
                    record_data(data_record, [np.NaN, np.NaN], ["E:neg_gen_valid:train", "E:neg_gen:train"])
            if args.kl_coef > 0:
                loss_kl_mean = loss_kl.mean().item() * args.kl_coef
                loss_entropy_mask_mean = loss_entropy_mask.mean().item() * args.entropy_coef_mask
                loss_entropy_repr_mean = loss_entropy_repr.mean().item() * args.entropy_coef_repr
                record_data(data_record, [loss_kl_mean, loss_entropy_mask_mean, loss_entropy_repr_mean, args.ebm_target], ["loss_kl", "loss_entropy_mask_mean", "loss_entropy_repr_mean", "ebm_target"])
                print("         loss_kl: {:.6f}     loss_ent_mask: {:.6f}     loss_ent_repr: {:.6f}    ebm_target: {}".format(loss_kl_mean, loss_entropy_mask_mean, loss_entropy_repr_mean, args.ebm_target), end="")
            if "neg_mask_exceed_energy" in info:
                record_data(data_record, [info["neg_mask_exceed_energy"] / args.batch_size], ["neg_mask_exceed_energy"])
                print("         neg_mask_exceed: {:.6f}".format(info["neg_mask_exceed_energy"] / args.batch_size), end="")
            if args.pos_consistency_coef > 0:
                record_data(data_record, [np.mean(loss_pos_consistency_list)], ["E:pos_consistency:train"])
                print("         loss_pos_consistency: {:.6f}".format(np.mean(loss_pos_consistency_list)), end="")
            if args.neg_consistency_coef > 0:
                record_data(data_record, [np.mean(loss_neg_consistency_list)], ["E:neg_consistency:train"])
                print("         loss_neg_consistency: {:.6f}".format(np.mean(loss_neg_consistency_list)), end="")
            if args.emp_consistency_coef > 0:
                record_data(data_record, [np.mean(loss_emp_consistency_list)], ["E:emp_consistency:train"])
                print("         loss_emp_consistency: {:.6f}".format(np.mean(loss_emp_consistency_list)), end="")
        elif args.train_mode == "sl":
            record_data(data_record, [epoch, np.mean(loss_mask_list), np.mean(loss_repr_list), np.mean(loss_list)],
                        ["epoch", "loss_mask:train", "loss_repr:train", "loss:train"])
            p.print("epoch {}: loss_mask: {:.6f}     loss_repr: {:.6f}     loss: {:.6f}".format(
                    str(epoch).zfill(3), np.mean(loss_mask_list), np.mean(loss_repr_list), np.mean(loss_list)), end="")
        else:
            raise
        print()

        if epoch % args.inspect_interval == 0 or epoch == args.epochs - 1:
            # Inspection, plotting and recording:
            torch.cuda.empty_cache()
            image_dir = filename[:-2] + "_samples/"
            make_dir(image_dir)
            concept_embeddings = get_concept_embeddings(CONCEPTS, OPERATORS)
            record_data(data_record, [concept_embeddings, epoch], ["concept_embeddings", "repr_epoch"])
            if args.is_mask:
                val_acc_dict = test_acc(model, args, val_loader, device, CONCEPTS=CONCEPTS, OPERATORS=OPERATORS, suffix=":val")
                val_acc_dict["acc+iou:mean:val"] = np.mean([val_acc_dict[key] for key in val_acc_dict if (key.startswith("acc:") or key.startswith("iou:")) and "_0" not in key and "_1" not in key])
                to_stop = early_stopping.monitor(val_acc_dict["acc+iou:mean:val"])
                record_data(data_record["acc"], [epoch], ["epoch:val"])
                record_data(data_record["acc"], list(val_acc_dict.values()), list(val_acc_dict.keys()))
                print()
                print(filename)
                pp.pprint(val_acc_dict)
                print()
                if args.is_two_branch:
                    pos_img_aug = (pos_img, pos_img) if not args.is_image_tuple else pos_img
                    if pos_img_aug[0].shape[1] == 3:
                        pos_img_core = torch.cat([pos_img_aug[0].detach().to('cpu'), pos_img_aug[1].detach().to('cpu')])
                    else:
                        pos_img_core = torch.cat([onehot_to_RGB(pos_img_aug[0]), onehot_to_RGB(pos_img_aug[1])])

                    if "mask" in args.w_type:
                        pos_mask_core = torch.cat([pos_mask[0].detach().to('cpu').round(), pos_mask[1].detach().to('cpu').round()])
                        neg_mask_core = torch.cat([neg_mask[0].detach().to('cpu').round(), neg_mask[1].detach().to('cpu').round()])
                    else:
                        pos_mask_core = torch.cat([onehot_to_RGB(to_one_hot(pos_mask[0].detach().to('cpu').argmax(1))), onehot_to_RGB(to_one_hot(pos_mask[1].detach().to('cpu').argmax(1)))])
                        neg_mask_core = torch.cat([onehot_to_RGB(to_one_hot(neg_mask[0].detach().to('cpu').argmax(1))), onehot_to_RGB(to_one_hot(neg_mask[1].detach().to('cpu').argmax(1)))])
                    if is_jupyter:
                        p.print("Pos images:")
                        if args.is_image_tuple:
                            if pos_img[0].shape[1] == 3:
                                visualize_matrices(pos_img[0][:6], images_per_row=6, use_color_dict=False)
                                visualize_matrices(pos_img[1][:6], images_per_row=6, use_color_dict=False)
                            else:
                                visualize_matrices(pos_img[0][:6].argmax(1), images_per_row=6)
                                visualize_matrices(pos_img[1][:6].argmax(1), images_per_row=6)
                        else:
                            if pos_img.shape[1] == 3:
                                visualize_matrices(pos_img[:6], images_per_row=6, use_color_dict=False)
                            else:
                                visualize_matrices(pos_img[:6].argmax(1), images_per_row=6)
                        if args.ebm_target in ["image+mask"]:
                            p.print("Neg images from SGLD:")
                            if args.is_image_tuple:
                                if neg_img[0].shape[1] == 3:
                                    visualize_matrices(neg_img[0][:6], images_per_row=6, use_color_dict=False)
                                    visualize_matrices(neg_img[1][:6], images_per_row=6, use_color_dict=False)
                                else:
                                    visualize_matrices(neg_img[0][:6].argmax(1), images_per_row=6)
                                    visualize_matrices(neg_img[1][:6].argmax(1), images_per_row=6)
                            else:
                                if neg_img.shape[1] == 3:
                                    visualize_matrices(neg_img[:6], images_per_row=6, use_color_dict=False)
                                else:
                                    visualize_matrices(neg_img[:6].argmax(1), images_per_row=6)
                        p.print("Pos masks:")
                        visualize_matrices(pos_mask[0][:6,0].round() if "mask" in args.w_type else pos_mask[0][:6].argmax(1), images_per_row=6, subtitles=pos_id[:6])
                        visualize_matrices(pos_mask[1][:6,0].round() if "mask" in args.w_type else pos_mask[1][:6].argmax(1), images_per_row=6, subtitles=pos_id[:6])
                        if args.train_mode == "cd" and args.neg_mode_coef > 0 and args.neg_mode != "None" and args.ebm_target not in ["image+mask"]:
                            p.print("Neg masks generated:")
                            visualize_matrices(neg_mask_gen[0][:6,0].round() if "mask" in args.w_type else neg_mask_gen[0][:6].argmax(1), images_per_row=6, subtitles=["valid" if is_valid else "invalid" for is_valid in neg_gen_valid.squeeze()[:6]])
                            visualize_matrices(neg_mask_gen[1][:6,0].round() if "mask" in args.w_type else neg_mask_gen[1][:6].argmax(1), images_per_row=6)
                        p.print("Neg masks from SGLD:")
                        visualize_matrices(neg_mask[0][:6,0].round() if "mask" in args.w_type else neg_mask[0][:6].argmax(1), images_per_row=6)
                        visualize_matrices(neg_mask[1][:6,0].round() if "mask" in args.w_type else neg_mask[1][:6].argmax(1), images_per_row=6)
                        print()
                    # Save to png files:
                    utils.save_image(
                        pos_img_core,
                        f'{image_dir}{str(epoch).zfill(5)}_image.png',
                        nrow=args.batch_size,
                        normalize=False,
                        value_range=(0, 1),
                    )
                    utils.save_image(
                        pos_mask_core,
                        f'{image_dir}{str(epoch).zfill(5)}_pos_mask.png',
                        nrow=args.batch_size,
                        normalize=False,
                        value_range=(0, 1),
                    )
                    utils.save_image(
                        neg_mask_core,
                        f'{image_dir}{str(epoch).zfill(5)}_neg_mask.png',
                        nrow=args.batch_size,
                        normalize=False,
                        value_range=(0, 1),
                    )
                else:
                    if pos_img.shape[1] == 3:
                        pos_img_core = pos_img.detach().to('cpu')
                    else:
                        pos_img_core = onehot_to_RGB(pos_img)
                    if "mask" in args.w_type:
                        pos_mask_core = pos_mask[0].detach().to('cpu').round()
                        neg_mask_core = neg_mask[0].detach().to('cpu').round()
                    else:
                        pos_mask_core = onehot_to_RGB(to_one_hot(pos_mask[0].detach().to('cpu').argmax(1)))
                        neg_mask_core = onehot_to_RGB(to_one_hot(neg_mask[0].detach().to('cpu').argmax(1)))
                    if is_jupyter:
                        p.print("Pos images:")
                        if pos_img.shape[1] == 3:
                            visualize_matrices(pos_img[:36], images_per_row=6, use_color_dict=False)
                        else:
                            visualize_matrices(pos_img[:36].argmax(1), images_per_row=6)
                        if args.ebm_target in ["image+mask"]:
                            p.print("Neg images from SGLD:")
                            if neg_img.shape[1] == 3:
                                visualize_matrices(neg_img[:6], images_per_row=6, use_color_dict=False)
                            else:
                                visualize_matrices(neg_img[:6].argmax(1), images_per_row=6)
                        p.print("Pos masks:")
                        visualize_matrices(pos_mask[0][:36,0].round() if "mask" in args.w_type else pos_mask[0][:6].argmax(1), images_per_row=6, subtitles=pos_id[:36])
                        if args.train_mode == "cd" and args.neg_mode_coef > 0 and args.neg_mode != "None" and args.ebm_target not in ["image+mask"]:
                            p.print("Neg masks generated:")
                            visualize_matrices(neg_mask_gen[0][:6,0].round() if "mask" in args.w_type else neg_mask_gen[0][:6].argmax(1), images_per_row=6, subtitles=["valid" if is_valid else "invalid" for is_valid in neg_gen_valid.squeeze()[:6]])
                        p.print("Neg masks from SGLD:")
                        visualize_matrices(neg_mask[0][:6,0].round() if "mask" in args.w_type else neg_mask[0][:6].argmax(1), images_per_row=6)
                        print()
                    utils.save_image(
                        pos_img_core,
                        f'{image_dir}{str(epoch).zfill(5)}_image.png',
                        nrow=16,
                        normalize=False,
                        value_range=(0, 1),
                    )
                    utils.save_image(
                        pos_mask_core,
                        f'{image_dir}{str(epoch).zfill(5)}_pos_mask.png',
                        nrow=16,
                        normalize=False,
                        value_range=(0, 1),
                    )
                    utils.save_image(
                        neg_mask_core,
                        f'{image_dir}{str(epoch).zfill(5)}_neg_mask.png',
                        nrow=16,
                        normalize=False,
                        value_range=(0, 1),
                    )
            else:
                if neg_img.shape[1] == 3:
                    neg_img_core = neg_img.detach().to('cpu')
                else:
                    neg_img_core = onehot_to_RGB(neg_img)
                if is_jupyter:
                    visualize_matrices(neg_img[:6].argmax(1), images_per_row=6)
                utils.save_image(
                    neg_img_core,
                    f'{image_dir}{str(epoch).zfill(5)}.png',
                    nrow=16,
                    normalize=True,
                    value_range=(0, 1),
                )
        if epoch % args.save_interval == 0 or epoch == args.epochs - 1:
            record_data(data_record, [epoch, model.model_dict], ["save_epoch", "model_dict"])
            data_record["optimizer_dict"] = to_cpu_recur(optimizer.state_dict())
            try_call(pdump, args=[data_record, filename], max_exp_time=600)
        if epoch % args.inspect_interval == 0 and to_stop:
            p.print("Early-stopping at epoch {}, with acc_mean:val={}".format(epoch, val_acc_dict["acc+iou:mean:val"]))
            record_data(data_record, [epoch, model.model_dict], ["save_epoch", "model_dict"])
            data_record["optimizer_dict"] = to_cpu_recur(optimizer.state_dict())
            try_call(pdump, args=[data_record, filename], max_exp_time=600)
            break

