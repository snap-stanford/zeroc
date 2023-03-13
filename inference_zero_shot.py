#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
from collections import OrderedDict, Iterable
from copy import deepcopy
from datetime import datetime
import matplotlib.pylab as plt
import gc
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
import matplotlib.pylab as plt
import os
import pdb
import pickle
import pprint as pp
import random
import pandas as pd
from scipy import stats
from IPython.display import display, HTML
pd.options.display.max_rows = 1500
pd.options.display.max_columns = 200
pd.options.display.width = 1000
pd.set_option('max_colwidth', 400)

from IPython.display import display
import numpy as np
from sklearn import linear_model
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from zeroc.datasets.arc_image import ARCDataset
from zeroc.datasets.BabyARC.code.dataset.dataset import *
from zeroc.concept_library.concepts import Concept, Concept_Pattern, Placeholder, Tensor
from zeroc.argparser import update_default_hyperparam, get_args_EBM, get_SGLD_kwargs
from zeroc.concept_library.models import load_best_model, GraphEBM, neg_mask_sgd, neg_mask_sgd_ensemble, ResBlock, CResBlock, spectral_norm
from zeroc.train import get_c_core, init_concepts_with_repr, ConceptDataset, ConceptFewshotDataset, ConceptCompositionDataset, get_dataset, requires_grad, test_acc, load_model_energy, SampleBuffer, sample_buffer, id_to_tensor
from zeroc.concept_library.settings import REPR_DIM
from zeroc.utils import REA_PATH, EXP_PATH
try:
    from zeroc.concept_library.util import try_call, plot_simple, plot_2_axis, Attr_Dict, MineDataset, Batch, extend_dims, groupby_add_keys, get_unique_keys_df, filter_df, groupby_add_keys, to_cpu_recur, Printer, get_num_params, transform_dict, get_graph_edit_distance, draw_nx_graph, get_nx_graph, get_triu_ids, get_soft_IoU, get_pdict, Batch, pdump, pload, filter_kwargs, gather_broadcast, ddeepcopy as deepcopy, to_Variable, set_seed, Zip, COLOR_LIST, init_args, make_dir, str2bool, get_filename_short, get_machine_name, get_device, record_data, plot_matrices, filter_filename, get_next_available_key, to_np_array, print_banner, get_filename_short, write_to_config, to_cpu
    from zeroc.concept_library.util import find_connected_components_colordiff, onehot_to_RGB, classify_concept, Shared_Param_Dict, get_inputs_targets_EBM, repeat_n, color_dict, to_one_hot, onehot_to_RGB, get_root_dir, get_module_parameters, assign_embedding_value, get_hashing, to_device_recur, visualize_matrices
except Exception as e:
    raise Exception(f"{e}. Please update the concept_library by running 'git submodule init; git submodule update' in the local zeroc repo!")

p = Printer()


# # 1. Helper functions:

# In[ ]:


def plot_loss(data_record, interval=1):
    fontsize = 12
    plt.figure(figsize=(8,6))
    plt.plot(data_record['epoch'][::interval], data_record['pos_out'][::interval], label="pos")
    plt.plot(data_record['epoch'][::interval], data_record['neg_out'][::interval], label="neg")
    plt.plot(data_record['epoch'][::interval], data_record['loss'][::interval], label="loss")
    plt.plot(data_record['epoch'][::interval], np.array(data_record['loss'][::interval]) - np.array(data_record['pos_out'][::interval]), label="loss-pos")
    plt.legend(fontsize=fontsize)
    plt.tick_params(labelsize=fontsize)
    plt.xlabel("epoch", fontsize=fontsize)
    plt.ylabel("loss", fontsize=fontsize)
    plt.show()


def update_CONCEPTS_OPERATORS(CONCEPTS, OPERATORS, concept_embeddings_load, update_keys=None):
    if update_keys is not None and not isinstance(update_keys, list):
        update_keys = [update_keys]
    for key in CONCEPTS:
        if key in concept_embeddings_load:
            if update_keys is not None and key in update_keys and key in concept_embeddings_load:
                c_repr_curr = CONCEPTS[key].get_node_repr()
                c_repr_curr.data = torch.FloatTensor(concept_embeddings_load[key])
    for key in OPERATORS:
        if update_keys is not None and key in update_keys and key in concept_embeddings_load:
            c_repr_curr = OPERATORS[key].get_node_repr()
            c_repr_curr.data = torch.FloatTensor(concept_embeddings_load[key])


def test_concept_embedding(CONCEPTS, OPERATORS, concept_embeddings_load, raise_warnings_only=False, checked_keys=None):
    concept_embeddings = {key: to_np_array(CONCEPTS[key].get_node_repr()) for key in CONCEPTS}
    operator_embeddings = {key: to_np_array(OPERATORS[key].get_node_repr()) for key in OPERATORS}
    concept_embeddings.update(operator_embeddings)
    different_list = []
    same_list = []
    for key, value in concept_embeddings_load.items():
        if key not in concept_embeddings:
            p.warning("key '{}' not in the current concept/relation embedding.".format(key))
            continue
        if (checked_keys is None or key in checked_keys) and np.abs(value - concept_embeddings[key]).max() > 0:
            different_list.append(key)
        else:
            same_list.append(key)
    if len(different_list) > 0:
        if raise_warnings_only:
            print("The c_repr for {} are the same.\nThe c_repr for {} are different.\n".format(same_list, different_list))
        else:
            raise Exception("The c_repr for {} are the same.\nThe c_repr for {} are different.\n".format(same_list, different_list))


def get_model_samples(
    model,
    args,
    dataset=None,
    init="gaussian",
    sample_step=None,
    batch_size=None,
    ensemble_size=1,
    plot_ensemble_mode="min",
    analysis_modes=["mask|c_repr", "c_repr|mask", "mask|c", "c_repr|c"],
    w_type="image+mask",
    plot_grey_scale=False,
    concept_collection=None,
    CONCEPTS=None,
    OPERATORS=None,
    isplot=True,
    device="cpu",
    plot_topk=3,
):
    """This one is with concept_collection argument."""
    def init_neg_mask(pos_img, buffer, args):
        """Initialize negative mask"""
        neg_data = sample_buffer(
            buffer,
            in_channels=args.in_channels,
            n_classes=args.n_classes,
            image_size=args.image_size,
            batch_size=batch_size*ensemble_size,
            is_mask=args.is_mask,
            is_two_branch=args.is_two_branch,
            w_type=args.w_type,
            p=args.p_buffer,
            device=device,
        )
        _, neg_mask, _, _ = neg_data
        pos_img_l = repeat_n(pos_img, n_repeats=ensemble_size)
        if args.is_two_branch:
            if init == "mask-gaussian":
                neg_mask = (neg_mask[0] * (pos_img_l[0].argmax(1)[:, None] != 0), neg_mask[1] * (pos_img_l[1].argmax(1)[:, None] != 0))
        else:
            if init == "mask-gaussian":
                pos_img_l = repeat_n(pos_img, n_repeats=ensemble_size)
                neg_mask = (neg_mask[0] * (pos_img_l.argmax(1)[:, None] != 0),)
        for k in range(len(neg_mask)):
            neg_mask[k].requires_grad = True
        return neg_mask

    model.eval()
    buffer = SampleBuffer()
    args = deepcopy(args)
    batch_size = args.batch_size if batch_size is None else batch_size
    sample_step = args.sample_step if sample_step is None else sample_step
    if concept_collection is not None:
        args.concept_collection = concept_collection
    args.sample_step = sample_step
    args.ebm_target = "mask"

    if args.is_mask:
        assert dataset is not None
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=Batch(is_collate_tuple=True).collate())
        for pos_data in dataloader:
            break
        pos_img, pos_mask, pos_id, _ = pos_data
        pos_repr = id_to_tensor(pos_id, CONCEPTS=CONCEPTS, OPERATORS=OPERATORS).to(device)
        if args.is_two_branch:
            pos_img = (pos_img[0].to(device), pos_img[1].to(device))
        else:
            pos_img = pos_img.to(device)
        pos_mask = to_device_recur(pos_mask, device)

        data = {
            "pos_img": pos_img,
            "pos_mask": pos_mask,
            "pos_id": pos_id,
            "pos_repr": pos_repr,
            "concept_collection": args.concept_collection,
        }

        # Calculate E(ground truth mask for given concept)
        if "mask|c_repr" in analysis_modes:
            neg_mask = init_neg_mask(pos_img, buffer, args)
            (_, neg_mask_ensemble, _, _, _), neg_out_list_ensemble, _ = neg_mask_sgd_ensemble(
                model, pos_img, neg_mask, pos_repr, z=None, zgnn=None, wtarget=None, args=args,
                ensemble_size=ensemble_size,
            )
            data["mask|c_repr"] = {
                "neg_mask_ensemble": neg_mask_ensemble,
                "neg_out_list_ensemble": neg_out_list_ensemble,
            }
        # Calculate E(concept given a ground truth mask)
        if "c_repr|mask" in analysis_modes:
            data["c_repr|mask"] = {"c_repr_pred": model.classify(pos_img, pos_mask, args.concept_collection, CONCEPTS=CONCEPTS, OPERATORS=OPERATORS)}

        if "mask|c" in analysis_modes:
            neg_mask_ensemble_collection = []
            neg_out_list_ensemble_collection = []
            for j in range(len(args.concept_collection)):
                neg_mask = init_neg_mask(pos_img, buffer, args)
                c_repr = id_to_tensor([args.concept_collection[j]] * len(pos_repr), CONCEPTS=CONCEPTS, OPERATORS=OPERATORS).to(device)
                (_, neg_mask_ensemble, _, _, _), neg_out_list_ensemble, _ = neg_mask_sgd_ensemble(
                    model, pos_img, neg_mask, c_repr, z=None, zgnn=None, wtarget=None, args=args,
                    ensemble_size=ensemble_size,
                    out_mode="min",
                )
                neg_mask_ensemble_collection.append(neg_mask_ensemble)
                neg_out_list_ensemble_collection.append(neg_out_list_ensemble)
            neg_mask_ensemble_collection = tuple(torch.stack([neg_mask_ensemble_ele[k] for neg_mask_ensemble_ele in neg_mask_ensemble_collection], -1) for k in range(len(neg_mask_ensemble_collection[0])))
            neg_out_list_ensemble_collection = np.stack(neg_out_list_ensemble_collection, -1)  # [n_steps, ensemble_size, batch_size, n_repr]
            c_repr_energy = neg_out_list_ensemble_collection[-1].min(0)
            c_repr_argsort = c_repr_energy.argsort(1)
            c_repr_pred_list = []
            for i, argsort in enumerate(c_repr_argsort):
                c_repr_pred = {}
                for k in range(len(args.concept_collection)):
                    id_k = c_repr_argsort[i][k]
                    c_repr_pred[args.concept_collection[id_k]] = c_repr_energy[i][id_k].item()
                c_repr_pred_list.append(c_repr_pred)
            data["both|c"] = {
                "neg_mask_ensemble_collection": neg_mask_ensemble_collection,
                "neg_out_list_ensemble_collection": neg_out_list_ensemble_collection,
                "c_repr_pred": c_repr_pred_list,
            }
        if isplot:
            plot_data(model, data, args, plot_ensemble_mode=plot_ensemble_mode, w_type=w_type, plot_grey_scale=plot_grey_scale, topk=plot_topk)
    else:
        neg_out_list = []
        neg_img, neg_id = neg_data
        neg_img.requires_grad = True
        noise = torch.randn(batch_size, args.in_channels, args.image_size, args.image_size, device=device)
        for k in range(sample_step):
            if "noise" not in locals():
                noise = torch.randn(neg_img.shape[0], args.in_channels, args.image_size, args.image_size, device=device)

            noise.normal_(0, args.lambd)
            neg_img.data.add_(noise.data)

            neg_out = model(neg_img, neg_id)
            neg_out.sum().backward()
            neg_out_list.append(to_np_array(neg_out))
            neg_img.grad.data.clamp_(-0.01, 0.01)

            neg_img.data.add_(neg_img.grad.data, alpha=-args.step_size)

            neg_img.grad.detach_()
            neg_img.grad.zero_()

            neg_img.data.clamp_(0, 1)

        data = {
            "neg_img": neg_img,
            "neg_id": neg_id,
        }
        neg_out_list = np.concatenate(neg_out_list, -1).T
        if isplot:
            if neg_img.shape[1] != 3:
                for img, neg_ele in zip(neg_img.argmax(1), neg_out):
                    print("loss={:.6f}".format(to_np_array(neg_ele)))
                    visualize_matrices([img])
    return data


def plot_data(model, data, args, plot_ensemble_mode="min", w_type="image+mask", topk=3, plot_grey_scale=False):
    plt.figure(figsize=(12,6))
    neg_out_list_ensemble = data["mask|c_repr"]["neg_out_list_ensemble"]  # [n_steps, ensemble_size, batch_size]
    for i in range(min(neg_out_list_ensemble.shape[-1], 6)):
        for k in range(neg_out_list_ensemble.shape[1]):
            plt.plot(neg_out_list_ensemble[:,k,i], c=COLOR_LIST[i], label="example_{}".format(i) if k==0 else None, alpha=0.4)
    plt.legend()
    plt.show()

    pos_img, pos_mask, pos_id, pos_repr = data["pos_img"], data["pos_mask"], data["pos_id"], data["pos_repr"]
    concept_collection = data["concept_collection"]
    if topk == -1:
        topk = len(concept_collection)
    topk = min(topk, len(concept_collection))
    length = len(pos_repr)
    pos_out_last = to_np_array(model(pos_img, mask=pos_mask, c_repr=pos_repr)).squeeze()
    if args.is_two_branch:
        if pos_img[0].shape[1] == 3:
            pos_img_core = torch.cat([pos_img[0].detach().to('cpu'), pos_img[1].detach().to('cpu')])
        else:
            pos_img_core = torch.cat([onehot_to_RGB(pos_img[0]), onehot_to_RGB(pos_img[1])])
        pos_mask_core = torch.cat([pos_mask[0].detach().to('cpu').round(), pos_mask[1].detach().to('cpu').round()])
        for i in range(0, length, 6):
            print("\n{} to {}:".format(i, i+5))
            print("positive images: input (up) and target (down)")
            visualize_matrices(pos_img[0][i:i+6].argmax(1), images_per_row=6)
            visualize_matrices(pos_img[1][i:i+6].argmax(1), images_per_row=6)
            print("positive mask: input obj mask (up) and target obj mask (down)")
            visualize_matrices(pos_mask[0][i:i+6,0].round() if "mask" in w_type else pos_mask[0][i:i+6].argmax(1), images_per_row=6, subtitles=["\n".join(["{}: {:.4f}".format("[{}]".format(key) if pos_id[i+j]==key else key, item) for k, (key, item) in enumerate(Dict.items()) if k < topk]) for j, Dict in enumerate(data["c_repr|mask"]["c_repr_pred"][i:i+6])])
            visualize_matrices(pos_mask[1][i:i+6,0].round() if "mask" in w_type else pos_mask[1][i:i+6].argmax(1), images_per_row=6, subtitles=["c:"+"\n".join(["{}: {:.4f}".format("[{}]".format(key) if pos_id[i+j]==key else key, item) for k, (key, item) in enumerate(Dict.items()) if k < topk]) for j, Dict in enumerate(data["both|c"]["c_repr_pred"][i:i+6])])
            if "mask|c_repr" in data:
                if "neg_out_argmin" not in locals():
                    neg_mask_ensemble = data["mask|c_repr"]["neg_mask_ensemble"]
                    neg_out_argmin = neg_out_list_ensemble[-1].argmin(0)  # neg_out_list_ensemble: [time_step, ensemble_size, batch_size]
                if plot_ensemble_mode == "min":
                    print("negative mask | c_repr: input obj mask (up) and target obj mask (down), with lowest-loss element in the ensemble:")
                    visualize_matrices([neg_mask_ensemble[0][neg_out_argmin[k],k,0].round() if "mask" in w_type else neg_mask_ensemble[0][neg_out_argmin[k],k].argmax(0) for k in range(i, i+6)], images_per_row=6, subtitles=["{}: {:.4f}".format(pos_id[k], neg_out_list_ensemble[-1,neg_out_argmin[k],k]) for k in range(i,i+6)])
                    visualize_matrices([neg_mask_ensemble[1][neg_out_argmin[k],k,0].round() if "mask" in w_type else neg_mask_ensemble[1][neg_out_argmin[k],k].argmax(0)  for k in range(i, i+6)], images_per_row=6)
                    if plot_grey_scale:
                        plot_matrices([neg_mask_ensemble[0][neg_out_argmin[k],k,0] for k in range(i, i+6)] if "mask" in w_type else neg_mask_ensemble[0][neg_out_argmin[k],k].argmax(0), scale_limit=(0,1), images_per_row=6)
                        plot_matrices([neg_mask_ensemble[1][neg_out_argmin[k],k,0] for k in range(i, i+6)] if "mask" in w_type else neg_mask_ensemble[1][neg_out_argmin[k],k].argmax(0), scale_limit=(0,1), images_per_row=6)
                elif plot_ensemble_mode == "all":
                    for j in range(ensemble_size):
                        is_best = j == neg_out_argmin
                        if j == 0:
                            print("negative mask | c_repr: {}th element in the ensemble for input obj mask (up) and target obj mask (down)".format(j))
                        else:
                            print("                        {}th element in the ensemble for input obj mask (up) and target obj mask (down)".format(j))
                        visualize_matrices([neg_mask_ensemble[0][j,k,0].round() if "mask" in w_type else neg_mask_ensemble[0][j,k].argmax(0) for k in range(i, i+6)], images_per_row=6, subtitles=["{}: {:.4f}{}".format(pos_id[k], neg_out_list_ensemble[-1,j,k], ", best" if is_best[k] else "") for k in range(i,i+6)])
                        visualize_matrices([neg_mask_ensemble[1][j,k,0].round() if "mask" in w_type else neg_mask_ensemble[1][j,k].argmax(0) for k in range(i, i+6)], images_per_row=6)
                        if plot_grey_scale:
                            plot_matrices([neg_mask_ensemble[0][j,k,0] for k in range(i, i+6)], scale_limit=(0,1), images_per_row=6)
                            plot_matrices([neg_mask_ensemble[1][j,k,0] for k in range(i, i+6)], scale_limit=(0,1), images_per_row=6)
                else:
                    raise
            if "both|c" in data:
                if "c_repr_pred_c" not in locals():
                    neg_mask_ensemble_collection = data["both|c"]['neg_mask_ensemble_collection']
                    neg_out_list_ensemble_collection = data["both|c"]["neg_out_list_ensemble_collection"]
                    c_repr_pred_c = data["both|c"]['c_repr_pred']
                    neg_out_argsort_c = neg_out_list_ensemble_collection[-1].min(0).argsort(-1)  # length batch_size, each indicating the argsort of concept id
                for j in range(topk):
                    if j == 0:
                        print("negative mask | c: top {}th prediction for input obj mask (up) and target obj mask (down)".format(j+1))
                    else:
                        print("                   top {}th prediction in the ensemble for input obj mask (up) and target obj mask (down)".format(j+1))

                    visualize_matrices([neg_mask_ensemble_collection[0][k,...,neg_out_argsort_c[k][j]].squeeze(0).round() 
                                        if "mask" in w_type else neg_mask_ensemble_collection[0][k,...,neg_out_argsort_c[k][j]].argmax(0)
                                        for k in range(i, i+6)], images_per_row=6,
                                       subtitles=["{}: {:.4f}".format("[{}]".format(concept_collection[neg_out_argsort_c[k][j]]) if concept_collection[neg_out_argsort_c[k][j]]==pos_id[k] else concept_collection[neg_out_argsort_c[k][j]], c_repr_pred_c[k][concept_collection[neg_out_argsort_c[k][j]]]) for k in range(i, i+6)]
                                      )
                    visualize_matrices([neg_mask_ensemble_collection[1][k,...,neg_out_argsort_c[k][j]].squeeze(0).round() 
                                        if "mask" in w_type else neg_mask_ensemble_collection[1][k,...,neg_out_argsort_c[k][j]].argmax(0)
                                        for k in range(i, i+6)], images_per_row=6,
                                       )
            print()
    else:
        if pos_img.shape[1] == 3:
            pos_img_core = pos_img.detach().to('cpu')
        else:
            pos_img_core = onehot_to_RGB(pos_img)
        for i in range(0, length, 6):
            print("\n{} to {}:".format(i, i+5))
            print("positive images:")
            visualize_matrices(pos_img[i:i+6].argmax(1), images_per_row=6)
            print("positive masks:")
            visualize_matrices(pos_mask[0][i:i+6,0].round() if "mask" in w_type else pos_mask[0][i:i+6].argmax(1), images_per_row=6, subtitles=["\n".join(["{}: {:.4f}".format("[{}]".format(key) if pos_id[i+j]==key else key, item) for k, (key, item) in enumerate(Dict.items()) if k < topk]) + "\nc:" + "\n".join(["{}: {:.4f}".format("[{}]".format(key) if pos_id[i+j]==key else key, item) for k, (key, item) in enumerate(data["both|c"]["c_repr_pred"][i+j].items()) if k < topk]) for j, Dict in enumerate(data["c_repr|mask"]["c_repr_pred"][i:i+6])])
            if "mask|c_repr" in data:
                if "neg_out_argmin" not in locals():
                    neg_mask_ensemble = data["mask|c_repr"]["neg_mask_ensemble"]
                    neg_out_argmin = neg_out_list_ensemble[-1].argmin(0)
                if plot_ensemble_mode == "min":
                    print("negative mask | c_repr: for lowest-loss element in the ensemble:")
                    visualize_matrices([neg_mask_ensemble[0][neg_out_argmin[k],k,0].round()
                                        if "mask" in w_type else neg_mask_ensemble[0][neg_out_argmin[k],k].argmax(0)
                                        for k in range(i, i+6)], images_per_row=6, subtitles=["{}: {:.4f}".format(pos_id[k], neg_out_list_ensemble[-1,neg_out_argmin[k],k]) for k in range(i,i+6)])
                    if plot_grey_scale:
                        plot_matrices([neg_mask_ensemble[0][neg_out_argmin[k],k,0] for k in range(i, i+6)], scale_limit=(0,1), images_per_row=6)
                elif plot_ensemble_mode == "all":
                    for j in range(ensemble_size):
                        if j == 0:
                            print("negative mask | c_repr: {}th element in the ensemble".format(j))
                        else:
                            print("                        {}th element in the ensemble".format(j))
                        is_best = j == neg_out_argmin
                        visualize_matrices([neg_mask_ensemble[0][j,k,0].round() for k in range(i, i+6)], images_per_row=6, subtitles=["{}: {:.4f}{}".format(pos_id[k], neg_out_list_ensemble[-1,j,k], ", best" if is_best[k] else "") for k in range(i,i+6)])
                        if plot_grey_scale:
                            plot_matrices([neg_mask_ensemble[0][j,k,0] for k in range(i, i+6)], scale_limit=(0,1), images_per_row=6)
                else:
                    raise
            if "both|c" in data:
                if "c_repr_pred_c" not in locals():
                    neg_mask_ensemble_collection = data["both|c"]['neg_mask_ensemble_collection']
                    neg_out_list_ensemble_collection = data["both|c"]["neg_out_list_ensemble_collection"]
                    c_repr_pred_c = data["both|c"]['c_repr_pred']
                    neg_out_argsort_c = neg_out_list_ensemble_collection[-1].min(0).argsort(-1)  # length batch_size, each indicating the argsort of concept id
                for j in range(topk):
                    if j == 0:
                        print("negative mask | c: top {}th prediction for the mask".format(j+1))
                    else:
                        print("                   top {}th prediction for the mask".format(j+1))

                    visualize_matrices([neg_mask_ensemble_collection[0][k,...,neg_out_argsort_c[k][j]].squeeze(0).round() 
                                        if "mask" in w_type else neg_mask_ensemble_collection[0][k,...,neg_out_argsort_c[k][j]].argmax(0)
                                        for k in range(i, i+6)], images_per_row=6,
                                       subtitles=["{}: {:.4f}".format("[{}]".format(concept_collection[neg_out_argsort_c[k][j]]) if concept_collection[neg_out_argsort_c[k][j]]==pos_id[k] else concept_collection[neg_out_argsort_c[k][j]], c_repr_pred_c[k][concept_collection[neg_out_argsort_c[k][j]]]) for k in range(i, i+6)]
                                      )
            print()


def plot_acc(dirname, filenames, acc_modes=None, filter_mode=None, prefix=None, suffix=None, is_plot_loss=True):
    # Plot the accuracy for the a set of model given paths. You can specify which accuracy modes, prefix, and suffix to be used. 
    # If acc_modes is None, we use all accuracy modes
    for filename in filenames: 
        path = dirname + filename
        data_record = pickle.load(open(path, "rb"))
        args = init_args(update_default_hyperparam(data_record["args"]))
        accuracy = data_record["acc"]

        table_acc_modes = {'Filenames': filenames}
        if acc_modes == None:
            acc_modes = accuracy.keys()
        if filter_mode is None:
            acc_modes = [acc_mode for acc_mode in acc_modes if (prefix is None or acc_mode.startswith(prefix)) and (suffix is None or acc_mode.endswith(suffix))]
        elif filter_mode == "standard":
            acc_modes = ['acc:mask|c_repr:val', 'acc:mask|c:val', 'acc:c_repr|mask:val', 'acc:c_repr|c:val']
        else:
            raise

        print(filename)
        plt.figure(figsize=(8,6))
        best_dict = {}
        for key in acc_modes:      
            x = accuracy["epoch:val"]
            y = accuracy[key]
            best_dict[key] = np.max(accuracy[key])
            plt.plot(x, y, label=key)
        acc_mean = np.array([accuracy[key] for key in acc_modes]).mean(0)
        acc_mean_argmax = acc_mean.argmax()
        plt.plot(x, acc_mean, label="mean_plot")
        plt.axvline(x=accuracy["epoch:val"][acc_mean_argmax], color='k', linestyle='--')
        if is_plot_loss:
            if args.train_mode == "cd":
                plt.plot(data_record["epoch"], data_record["E:pos:train"], label="E:pos:train")
                plt.plot(data_record["epoch"], data_record["E:neg|c_repr:train"], label="E:neg:train")
                plt.plot(data_record["epoch"], data_record["E:neg_gen:train"], label="E:neg_gen:train")
                if "loss_kl" in data_record:
                    plt.plot(data_record["epoch"], data_record["loss_kl"], label="loss_kl")
                    plt.plot(data_record["epoch"], data_record["loss_entropy_mask_mean"], label="loss_entropy_mask_mean")
                    plt.plot(data_record["epoch"], data_record["loss_entropy_repr_mean"], label="loss_entropy_repr_mean")
            elif args.train_mode == "sl":
                plt.plot(data_record["epoch"], data_record["loss_mask:train"], label="loss_mask:train")
                plt.plot(data_record["epoch"], data_record["loss_repr:train"], label="loss_repr:train")
                plt.plot(data_record["epoch"], data_record["loss:train"], label="loss:train")
            else:
                raise

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        title = "Best:"
        for key in best_dict:
            title += "  {}={:.4f}".format(key[4:-4], best_dict[key], end="")
        plt.title(title)
        plt.show()


def analyze(
    dirname,
    filename,
    init="gaussian", # "mask-gaussian"
    batch_size=12,
    load_epoch=-1,
    ensemble_size=8,
    plot_ensemble_mode="min",
    plot_grey_scale=False,
    analysis_result=None,
    # New settings:
    n_examples=1000,
    sample_step=120,
    lambd=-1,
    seed=2,
    CONCEPTS=None,
    OPERATORS=None,
    isplot=True,
):
    """Get test_acc, generate samples, and visualize."""
    print_banner(filename)
    try:
        data_record = pickle.load(open(dirname + filename, "rb"))
    except Exception as e:
        print(e)
        return None, None, None
    if isplot:
        plot_loss(data_record)
    args = init_args(update_default_hyperparam(data_record["args"]))
    pp.pprint(args.__dict__)
    if 'concept_embeddings' in data_record:
        test_concept_embedding(CONCEPTS, OPERATORS, data_record['concept_embeddings'])

    if n_examples is not None:
        args.n_examples = n_examples
    args.gpuid = "False"
    if lambd != -1:
        args.lambd = lambd
    device = "cpu"
    if load_epoch < 0:
        load_epoch += data_record["epoch"][-1]
    id = int(np.round(load_epoch / args.save_interval))
    if id > len(data_record["model_dict"]):
        print("{} has only {} epochs. Continue".format(filename, data_record["epoch"][-1]))
        return None, None, None

    if analysis_result is None:
        set_seed(seed=seed)
        dataset, args = get_dataset(args)
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=True, collate_fn=Batch(is_collate_tuple=True).collate())
        model = load_model_energy(data_record["model_dict"][id], device=device)
        val_acc_dict = test_acc(model, args, dataloader, device, CONCEPTS=CONCEPTS, OPERATORS=OPERATORS, suffix=":val")
        print("\nacc and energy:")
        pp.pprint(val_acc_dict)
        print("\nmodel at epoch {}:".format(id * args.save_interval))
        # Perform gradient descent to find a low energy mask:
        data, neg_out_list_ensemble = get_model_samples(model, args, dataset=dataset, init=init, sample_step=sample_step, batch_size=batch_size, ensemble_size=ensemble_size, plot_ensemble_mode=plot_ensemble_mode, plot_grey_scale=plot_grey_scale, CONCEPTS=CONCEPTS, OPERATORS=OPERATORS, isplot=isplot)
    else:
        print("Loading saved analysis:")
        model = load_model_energy(data_record["model_dict"][id], device=device)
        data = analysis_result["data"]
        neg_out_list_ensemble = analysis_result["neg_out_list_ensemble"]
        val_acc_dict = analysis_result["val_acc_dict"]
        print("\nacc and energy:")
        pp.pprint(val_acc_dict)
        print("\nmodel at epoch {}:".format(id * args.save_interval))
        if isplot:
            plot_data(model, data, neg_out_list_ensemble, args, plot_ensemble_mode=plot_ensemble_mode, plot_grey_scale=plot_grey_scale)
        dataset = None
    print()
    data_record.pop("model_dict")
    info = {
        "args": args,
        "data_record": data_record,
        "data": data,
        "neg_out_list_ensemble": neg_out_list_ensemble,
        "val_acc_dict": val_acc_dict,
    }
    return model, dataset, info


def get_useful_keys(df, args_keys):
    useful_keys = []
    for key in sorted(args_keys):
        try:
            df_mean = df.groupby(by=key).mean()
        except:
            continue
        if len(df_mean) == 1:
            continue
        useful_keys.append(key)
    if "gpuid" in useful_keys:
        useful_keys.remove("gpuid")
    if "is_two_branch" in useful_keys:
        useful_keys.remove("is_two_branch")
    return useful_keys


def print_info(args, keys=None):
    if keys is None:
        keys = ["dataset", "canvas_size", "neg_mode_coef", "ebm_target_mode", "step_size_repr", "c_repr_mode", "c_repr_first", "channel_base", "n_examples", "aggr_mode", "kl_coef", "entropy_coef_mask", "entropy_coef_repr", "is_spec_norm", "color_avail","id"]
    for key in keys:
        print("{}: {}".format(key, getattr(args, key)))


def get_update_acc(dirname, suffix=""):
    """Get Pandas dataframe analysing acc."""
    filenames = filter_filename(dirname, include=".p")
    df_dict_list = []
    for filename in filenames:
        data_record = pickle.load(open(dirname + filename, "rb"))
        args = init_args(update_default_hyperparam(data_record["args"]))
        df_dict = {}
        acc_dict = data_record["acc"]
        for key in acc_dict:
            best_id_acc = np.argmax(acc_dict['acc:mean:val'])
            acc_repr = np.array([acc_dict[key] for key in acc_dict if key.startswith("acc:c_repr")]).mean(0)
            acc_mask = np.array([acc_dict[key] for key in acc_dict if key.startswith("acc:mask") and "_0" not in key and "_1" not in key]).mean(0)
            best_id_acc_repr = np.argmax(acc_repr)
            best_id_acc_mask = np.argmax(acc_mask)
            df_dict["epoch"] = acc_dict["epoch:val"][-1]
            df_dict["best_epoch:acc"] = acc_dict["epoch:val"][best_id_acc]
            df_dict["best_epoch:acc_repr"] = acc_dict["epoch:val"][best_id_acc_repr]
            df_dict["best_epoch:acc_mask"] = acc_dict["epoch:val"][best_id_acc_mask]
            df_dict["acc:c_repr:mean:val"] = np.max(acc_repr)
            df_dict["acc:mask:mean:val"] = np.max(acc_mask)

            #print(df_dict["best_epoch:acc_mask"])
        best_id_model = int(df_dict["best_epoch:acc_mask"] / 20)
        decice= "cuda:0"
        model = load_model_energy(data_record["model_dict"][best_id_model], device=device)
        set_seed(seed=2)
        args.n_examples = 500
        dataset, _ = get_dataset(args)
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, collate_fn=Batch(is_collate_tuple=True).collate())
        decice= "cuda:0"
        update_val_acc_dict = test_acc(model, args, dataloader, device, CONCEPTS=CONCEPTS, OPERATORS=OPERATORS, suffix=suffix)
        print(filename)
        print("epoch: ", best_id_model)
        print(update_val_acc_dict)
        print("\n")


def get_df(dirname):
    """Get Pandas dataframe analysising acc."""
    filenames = filter_filename(dirname, include=".p")
    df_dict_list = []
    for filename in filenames:
        try:
            data_record = pickle.load(open(dirname + filename, "rb"))
        except Exception as e:
            print("Error {} from file {}".format(e, filename))
        args = init_args(update_default_hyperparam(data_record["args"]))

        df_dict = {}
        acc_dict = data_record["acc"]
        for key in acc_dict:
            best_id_acc = np.argmax(acc_dict['acc:mean:val'])
            acc_repr = np.array([acc_dict[key] for key in acc_dict if key.startswith("acc:c_repr")]).mean(0)
            acc_mask = np.array([acc_dict[key] for key in acc_dict if key.startswith("acc:mask") and "_0" not in key and "_1" not in key]).mean(0)
            best_id_acc_repr = np.argmax(acc_repr)
            best_id_acc_mask = np.argmax(acc_mask)
            df_dict["epoch"] = acc_dict["epoch:val"][-1]
            df_dict["best_epoch:acc"] = acc_dict["epoch:val"][best_id_acc]
            df_dict["best_epoch:acc_repr"] = acc_dict["epoch:val"][best_id_acc_repr]
            df_dict["best_epoch:acc_mask"] = acc_dict["epoch:val"][best_id_acc_mask]
            df_dict["acc:c_repr:mean:val"] = np.max(acc_repr)
            df_dict["acc:mask:mean:val"] = np.max(acc_mask)

            if key == "epoch:val":
                pass
            elif key.startswith("acc:"):
                df_dict[key] = np.max(acc_dict[key])
            elif key.startswith("E:"):
                df_dict[key] = acc_dict[key][best_id_acc]
            else:
                raise
        df_dict.update(args.__dict__)
        df_dict["filename"] = filename
        df_dict_list.append(df_dict)

    df = pd.DataFrame(df_dict_list)
    info = {"args_keys": list(args.__dict__.keys()),
            "acc_keys": [key for key in acc_dict.keys() if "_0" not in key and "_1" not in key],
           }
    return df, info


def get_bidirectional_graph(graph):
    Dict = {"IsInside": "IsEnclosed",
            "IsEnclosed": "IsInside",
           }
    graph_new = []
    # Standardize:
    def standardize_graph(graph):
        new_graph = []
        for ele in graph:
            if isinstance(ele[0], list):
                ele = (tuple(ele[0]), ele[1])
            new_graph.append(ele)
        return new_graph
    new_graph = standardize_graph(graph)
    graph_add = []

    for ele in new_graph:
        if isinstance(ele[0], tuple):
            reverse_ele = ((ele[0][1], ele[0][0]), ele[1])
            if reverse_ele not in new_graph:
                graph_add.append(reverse_ele)
    return new_graph + graph_add


# ### 1.1 Helper functions:

# In[ ]:


def load_model_hash(
    hash_str,
    is_dataset=False,
    is_update_repr=False,
    isplot=1,
    mutual_exclusive_coef=None,
    return_args=False,
    update_keys=None,
    load_epoch="best",
):
    try:
        load_epoch = eval(load_epoch)
    except:
        pass
    subfolders = filter_filename(EXP_PATH, exclude=".")
    is_found = False
    for subfolder in subfolders:
        if "evaluation" in subfolder:
            continue
        filenames = filter_filename(f"{EXP_PATH}/{subfolder}/", include=[hash_str, ".p"])
        if len(filenames) == 1:
            is_found = True
            break
    assert is_found, f"Did not find the experiment with hash_str '{hash_str}' under the subfolders of './{EXP_PATH}/'. Please check if the hash_str is correct."
    filename = filenames[0]
    data_record = pickle.load(open(f"{EXP_PATH}/{subfolder}/{filename}", "rb"))
    if is_update_repr:
        update_CONCEPTS_OPERATORS(CONCEPTS, OPERATORS, data_record["concept_embeddings"][-1], update_keys=update_keys)
        test_concept_embedding(CONCEPTS, OPERATORS, data_record["concept_embeddings"][-1], raise_warnings_only=True, checked_keys=update_keys)
    else:
        test_concept_embedding(CONCEPTS, OPERATORS, data_record["concept_embeddings"][-1], checked_keys=update_keys)
    args = init_args(update_default_hyperparam(data_record["args"]))
    if isplot >= 1:
        plot_acc(dirname, [filename], filter_mode="standard")
    model, best_model_id = load_best_model(data_record,
                            keys=["mask|c_repr", "mask|c", "c_repr|mask", "c_repr|c"],
                            load_epoch=load_epoch,
                            return_id=True,
                           )
    model.to(device)

    info = {"load_id": best_model_id}
    if return_args:
        info["args"] = args
    if is_dataset:
        args.n_examples = 500
        args.seed = 2
        print("canvas_size: {}, color_avail: {}".format(args.canvas_size, args.color_avail))
        dataset, args = get_dataset(args, is_load=True)
        info["dataset"] = dataset

        if isplot >= 2:
            init = "mask-input"
            sample_step = 150
            batch_size = 36
            ensemble_size = 16
            plot_grey_scale = False
            plot_ensemble_mode = "min"
            args.lambd_start = 0.1
            if mutual_exclusive_coef is None:
                mutual_exclusive_coef = args.mutual_exclusive_coef
            args.mutual_exclusive_coef = mutual_exclusive_coef
            data = get_model_samples(model, args, dataset=dataset, init=init, sample_step=sample_step, batch_size=batch_size, ensemble_size=ensemble_size, plot_ensemble_mode=plot_ensemble_mode, plot_grey_scale=plot_grey_scale, CONCEPTS=CONCEPTS, OPERATORS=OPERATORS, device=device, isplot=True)
    return model, info


# Build an empty selector:
def get_empty_selector(ebm_dict, CONCEPTS, OPERATORS):
    selector = Concept_Pattern(
        name=None,
        value=Placeholder(Tensor(dtype="cat", range=range(10))),
        attr={},
        is_all_obj=True,
        is_ebm=True,
        is_selector_gnn=False,
        is_default_ebm=False,
        ebm_dict=ebm_dict,
        CONCEPTS=CONCEPTS,
        OPERATORS=OPERATORS,
        device=device,
        cache_forward=False,
        in_channels=10,
        z_mode="None",
        w_type="image+mask",
        mask_mode="concat",
        aggr_mode="max",
        pos_embed_mode="None",
        is_ebm_share_param=True,
        is_relation_z=False,
        img_dims=2,
        is_spec_norm=True,
        act_name="leakyrelu0.2",
        normalization_type="None",
    )
    return selector


def get_c_repr(c_str, num, device, mode):
    if mode == "concept":
        c_repr = CONCEPTS[c_str].get_node_repr().detach()[None].repeat_interleave(num, 0).to(device)
    elif mode == "relation":
        c_repr = OPERATORS[c_str].get_node_repr().detach()[None].repeat_interleave(num, 0).to(device)
    else:
        raise
    return c_repr


def get_graph_info(info):
    # Get all objects and their concepts:
    graph_info = {"obj_type": {}, "relations": {}}
    obj_masks = info["obj_masks"]
    graph_info["obj_masks"] = obj_masks
    for key, mask in obj_masks.items():
        obj_type = classify_concept(mask)
        graph_info["obj_type"][key] = obj_type
    # Get relations:
    for key, relation in info["obj_full_relations"].items():
        relations_valid = {"SameShape", "SameColor", "IsInside"}
        relation_ele_valid = set(relation).intersection(relations_valid)
        if len(relation_ele_valid) == 0:
            continue
        if len(relation_ele_valid) == 2:
            pass
        graph_info["relations"][key] = list(relation_ele_valid)
    graph_info["structure"] = info["structure"]
    return graph_info


def get_concept_from_graph_info(graph_info):
    attr_dict = {}
    for key, obj_type in graph_info["obj_type"].items():
        attr_dict["{}".format(key, obj_type)] = Placeholder(obj_type, value=graph_info["obj_masks"][key])

    concept = Concept(
        name="concept",
        value=Placeholder(Tensor(dtype="cat", range=range(10))),
        attr=attr_dict,
    )

    for key, relations in graph_info["relations"].items():
        for relation in relations:
            concept.add_relation_manual(relation, key[0], key[1])
    return concept


def get_query_concept_pattern(query_type, graph_info):
    def filter_relations(relations):
        relations = deepcopy(relations)
        keys_to_pop = []
        for key, relation in relations.items():
            if (key[1], key[0]) in relations and relations[(key[1], key[0])] == relation and key not in keys_to_pop:
                keys_to_pop.append((key[1], key[0]))
        for key in keys_to_pop:
            relations.pop(key)
        return relations

    query_type = query_type.split("-")[-1]
    obj_names = list(graph_info["obj_type"])
#     graph_info = deepcopy(graph_info)
#     graph_info["relations"] = filter_relations(graph_info["relations"])
    concept = get_concept_from_graph_info(graph_info)

    if query_type == "1c":
        obj_type_avail = list(graph_info["obj_type"].values())
        obj_type_query = np.random.choice(obj_type_avail)
        query_key_candidates = [key for key, obj_type in graph_info["obj_type"].items() if obj_type == obj_type_query]
        refer_masks = torch.stack([graph_info["obj_masks"][key] for key in query_key_candidates])
        query = {
            "graph": [(0, obj_type_query)],
            "refer": 0,
        }
    elif query_type == "2a":
        # [concept, (0, 1)/(1, 0), (refer)]
        assert len(graph_info["relations"]) > 0
        relation_id_selected = np.random.choice(len(graph_info["relations"]))
        relation_key_selected = list(graph_info["relations"])[relation_id_selected]
        key0, key1 = relation_key_selected
        pivot_node_id = np.random.choice(2)
        refer_node_id = 1 - pivot_node_id
        pivot_node_key = relation_key_selected[pivot_node_id]
        c_pattern = Concept_Pattern(
            name=None,
            value=Placeholder(Tensor(dtype="cat", range=range(10))),
            attr={key0: Placeholder(graph_info["obj_type"][key0]),
                  key1: Placeholder(graph_info["obj_type"][key1]),
                 },
            re={relation_key_selected: graph_info["relations"][relation_key_selected][0]},
            refer_node_names=[relation_key_selected[refer_node_id]],
        )
        refer_masks = concept.get_refer_nodes(c_pattern, is_match_node=True)
        refer_masks = {key: value.get_node_value() for key, value in refer_masks.items()}
        query = {"graph": [((0,1),graph_info["relations"][relation_key_selected][0]), (pivot_node_id, graph_info["obj_type"][pivot_node_key])],
                 "refer": refer_node_id}
        refer_masks = torch.stack(list(refer_masks.values()))
    elif query_type == "3a":
        # [concept, (0, 1)/(1, 0), (1,2)/(2,1), (refer)]
        if graph_info["structure"] == ['pivot:Rect', (1, 0, 'IsInside'), '(concept)', (1, 2), '(refer)']:
            query = {
                "graph": [(0, "Rect"), ((1,0), "IsInside"), ((1,2), graph_info["relations"][(obj_names[1], obj_names[2])][0])],
                "refer": 2,
            }
            refer_masks = graph_info["obj_masks"][obj_names[2]][None]
        elif graph_info["structure"] == ["pivot", (0,1), "(concept)", (1,2), "(refer)"]:
            query = {
                "graph": [(0, graph_info["obj_type"][obj_names[0]]), ((0,1), graph_info["relations"][(obj_names[0], obj_names[1])][0]), ((1,2), graph_info["relations"][(obj_names[1], obj_names[2])][0])],
                "refer": 2,
            }
            refer_masks = graph_info["obj_masks"][obj_names[2]][None]
        else:
            return None, None
    elif query_type == "3b":
        if graph_info["structure"] == ["pivot", "pivot", (0,2), (1,2), "(refer)"]:
            query = {
                "graph": [(0, graph_info["obj_type"][obj_names[0]]),
                          (1, graph_info["obj_type"][obj_names[1]]),
                          ((0,2), graph_info["relations"][(obj_names[0], obj_names[2])][0]),
                          ((1,2), graph_info["relations"][(obj_names[1], obj_names[2])][0]),
                         ],
                "refer": 2,
            }
            refer_masks = graph_info["obj_masks"][obj_names[2]][None]
        else:
            return None, None
    return query, refer_masks


def get_all_queries(dataset, query_type, seed=2, isplot=False):
    set_seed(seed)
    query_all_dict = {}
    query_type = query_type.split("-")[-1]
    for i in range(len(dataset)):
        input, target, infos = get_inputs_targets_EBM(dataset[i])
        input = torch.stack(list(input[0].values())).to(device)
        for j in range(len(input)):
            graph_info = get_graph_info(infos[j])
            query, refer_masks = get_query_concept_pattern(query_type, graph_info)
            if query is not None:
                record_data(query_all_dict, [input[j:j+1], query, refer_masks], ["input", "query", "refer_masks"])
                if isplot:
                    visualize_matrices(input[j:j+1].argmax(1))
                    print(query)
                    plot_matrices(refer_masks)
                    print("\n\n")
    return query_all_dict


def get_selector_from_graph(graph, ebm_dict, CONCEPTS, OPERATORS):
    def get_is_exist(selector, node_name):
        return node_name in [ele.split(":")[0] for ele in list(selector.nodes)]
    selector = get_empty_selector(ebm_dict, CONCEPTS, OPERATORS)
    for i, item in enumerate(graph):
        if isinstance(item[0], tuple):
            obj0 = "obj_{}".format(item[0][0]) if get_is_exist(selector, "obj_{}".format(item[0][0])) else None
            obj1 = "obj_{}".format(item[0][1]) if get_is_exist(selector, "obj_{}".format(item[0][1])) else None
            if obj0 is not None and obj1 is None:
                selector.add_relation_manual(item[1], obj0)
            elif obj0 is None and obj1 is not None:
                selector.add_relation_manual(item[1], None, obj1)
            elif obj0 is None and obj1 is None:
                selector.add_relation_manual(item[1])
            else:
                assert obj0 is not None and obj1 is not None
                selector.add_relation_manual(item[1], obj0, obj1)
        else:
            selector.add_obj(CONCEPTS[item[1]], add_obj_name="obj_{}".format(item[0]))
    if "refer" in graph:
        selector.set_refer_nodes("obj_{}".format(graph["refer"]))
    return selector


def get_selector_for_parsing(keys_dict, ebm_dict, CONCEPTS, OPERATORS):
    """E.g. keys_dict = {"Line": 3, "SameShape": 4}"""
    selector = get_empty_selector(ebm_dict, CONCEPTS, OPERATORS)
    for key, count in keys_dict.items():
        if key in CONCEPTS:
            for i in range(count):
                selector.add_obj(CONCEPTS[key])
        elif key in OPERATORS:
            for i in range(count):
                selector.add_relation_manual(key)
        else:
            raise
    return selector


def get_concept_graph(c_type, is_new_vertical=True, is_bidirectional_re=False, is_concept=True, is_relation=True):
    if is_new_vertical:
        if not is_bidirectional_re:
            graph_gt_dict = {
                "Line": [
                    (0, "Line"),
                ],
                "2-Line": [
                    (0, "Line"),
                    (1, "Line"),
                ],
                "3-Line": [
                    (0, "Line"),
                    (1, "Line"),
                    (2, "Line"),
                ],
                "Parallel": [
                    ((0, 1), "Parallel"),
                ],
                "VerticalMid": [
                    ((0, 1), "VerticalMid"),
                ],
                "VerticalEdge": [
                    ((0, 1), "VerticalEdge"),
                ],
                "Eshape": [
                    (0, "Line"),
                    (1, "Line"),
                    (2, "Line"),
                    (3, "Line"),
                    ((0, 1), 'VerticalEdge'),
                    ((0, 2), 'VerticalMid'),
                    ((0, 3), 'VerticalEdge'),
                    ((1, 2), 'Parallel'),
                    ((1, 3), 'Parallel'),
                    ((2, 3), 'Parallel'),
                ],
                "Fshape": [
                    (0, "Line"),
                    (1, "Line"),
                    (2, "Line"),
                    ((0, 1), 'VerticalEdge'),
                    ((0, 2), 'VerticalMid'),
                    ((1, 2), 'Parallel'),  
                ],
                "Lshape": [
                    (0, "Line"),
                    (1, "Line"),
                    ((0, 1), 'VerticalEdge'),
                ],
                "Tshape": [
                    (0, "Line"),
                    (1, "Line"),
                    ((0, 1), 'VerticalMid'),
                ],
                "Cshape": [
                    (0, "Line"),
                    (1, "Line"),
                    (2, "Line"),
                    ((0, 1), 'VerticalEdge'),
                    ((0, 2), 'VerticalEdge'),
                    ((1, 2), 'Parallel'),  
                ],
                "Ashape": [
                    (0, "Line"),
                    (1, "Line"),
                    (2, "Line"),
                    (3, "Line"),
                    ((0, 1), 'VerticalEdge'),
                    ((0, 2), 'VerticalMid'),
                    ((0, 3), 'Parallel'),
                    ((1, 2), 'Parallel'),
                    ((1, 3), 'VerticalEdge'),
                    ((2, 3), 'VerticalMid'),
                ],
                "Hshape": [
                    (0, "Line"),
                    (1, "Line"),
                    (2, "Line"),
                    ((0, 1), 'VerticalMid'),
                    ((1, 2), 'VerticalMid'),
                    ((0, 2), 'Parallel'),
                ],
                "Rect": [
                    (0, "Line"),
                    (1, "Line"),
                    (2, "Line"),
                    (3, "Line"),
                    ((0, 1), 'VerticalEdge'),
                    ((1, 2), 'VerticalEdge'),
                    ((2, 3), 'VerticalEdge'),
                    ((0, 3), 'VerticalEdge'),
                    ((0, 2), 'Parallel'),
                    ((1, 3), 'Parallel'),
                ],
                "RectE1a": [
                    (0, "Rect"),
                    (1, "Rect"),
                    (2, "Eshape"),
                    ((0, 1), "IsInside"),
                    ((1, 0), "IsEnclosed"),
                    ((0, 2), "IsNonOverlapXY"),
                    ((2, 0), "IsNonOverlapXY"),
                    ((1, 2), "IsNonOverlapXY"),
                    ((2, 1), "IsNonOverlapXY"),
                ],
                "RectE1b": [
                    (0, "Rect"),
                    (1, "Rect"),
                    (2, "Eshape"),
                    ((0, 1), "IsInside"),
                    ((1, 0), "IsEnclosed"),
                    ((0, 2), "IsNonOverlapXY"),
                    ((2, 0), "IsNonOverlapXY"),
                    ((1, 2), "IsNonOverlapXY"),
                    ((2, 1), "IsNonOverlapXY"),
                ],
                "RectE2a": [
                    (0, "Rect"),
                    (1, "Rect"),
                    (2, "Eshape"),
                    ((0, 1), "IsInside"),
                    ((1, 0), "IsEnclosed"),
                    ((2, 1), "IsInside"),
                    ((1, 2), "IsEnclosed"),
                    ((0, 2), "IsNonOverlapXY"),
                    ((2, 0), "IsNonOverlapXY"),
                ],
                "RectE2b": [
                    (0, "Rect"),
                    (1, "Rect"),
                    (2, "Eshape"),
                    ((0, 1), "IsInside"),
                    ((1, 0), "IsEnclosed"),
                    ((2, 1), "IsInside"),
                    ((1, 2), "IsEnclosed"),
                    ((0, 2), "IsNonOverlapXY"),
                    ((2, 0), "IsNonOverlapXY"),
                ],
                "RectE3a": [
                    (0, "Rect"),
                    (1, "Rect"),
                    (2, "Eshape"),
                    ((0, 1), "IsInside"),
                    ((1, 0), "IsEnclosed"),
                    ((2, 1), "IsInside"),
                    ((1, 2), "IsEnclosed"),
                    ((2, 0), "IsInside"),
                    ((0, 2), "IsEnclosed"),
                ],
                "RectE3b": [
                    (0, "Rect"),
                    (1, "Rect"),
                    (2, "Eshape"),
                    ((0, 1), "IsInside"),
                    ((1, 0), "IsEnclosed"),
                    ((2, 1), "IsInside"),
                    ((1, 2), "IsEnclosed"),
                    ((2, 0), "IsInside"),
                    ((0, 2), "IsEnclosed"),
                ],
                "RectEconcept": [
                    (0, "Rect"),
                    (1, "Rect"),
                    (2, "Eshape"),
                ],
                "Graph1": [
                    (0, "Red"),
                    ((0,1), "SameColor"),
                    ((1,2), "SameShape"),
                ],
                "Graph2": [
                    (0, "Large"),
                    ((0,1), "SameSize"),
                    ((0,2), "SameColor"),   
                ],
                "Graph3": [
                    (0, "Cube"),
                    ((0,1), "SameShape"),
                    ((1,2), "SameSize"),
                ],
            }
        else:
            graph_gt_dict = {
                "Line": [
                    (0, "Line"),
                ],
                "2-Line": [
                    (0, "Line"),
                    (1, "Line"),
                ],
                "3-Line": [
                    (0, "Line"),
                    (1, "Line"),
                    (2, "Line"),
                ],
                "Parallel": [
                    ((0, 1), "Parallel"),
                    ((1, 0), "Parallel"),
                ],
                "VerticalMid": [
                    ((0, 1), "VerticalMid"),
                    ((1, 0), "VerticalMid"),
                ],
                "VerticalEdge": [
                    ((0, 1), "VerticalEdge"),
                    ((1, 0), "VerticalEdge"),
                ],
                "Eshape": [
                    (0, "Line"),
                    (1, "Line"),
                    (2, "Line"),
                    (3, "Line"),
                    ((0, 1), 'VerticalEdge'),
                    ((1, 0), 'VerticalEdge'),
                    ((0, 2), 'VerticalMid'),
                    ((2, 0), 'VerticalMid'),
                    ((0, 3), 'VerticalEdge'),
                    ((3, 0), 'VerticalEdge'),
                    ((1, 2), 'Parallel'),
                    ((2, 1), 'Parallel'),
                    ((1, 3), 'Parallel'),
                    ((3, 1), 'Parallel'),
                    ((2, 3), 'Parallel'),
                    ((3, 2), 'Parallel'),
                ],
                "Fshape": [
                    (0, "Line"),
                    (1, "Line"),
                    (2, "Line"),
                    ((0, 1), 'VerticalEdge'),
                    ((1, 0), 'VerticalEdge'),
                    ((0, 2), 'VerticalMid'),
                    ((2, 0), 'VerticalMid'),
                    ((1, 2), 'Parallel'),
                    ((2, 1), 'Parallel'),
                ],
                "Lshape": [
                    (0, "Line"),
                    (1, "Line"),
                    ((0, 1), 'VerticalEdge'),
                    ((1, 0), 'VerticalEdge'),
                ],
                "Tshape": [
                    (0, "Line"),
                    (1, "Line"),
                    ((0, 1), 'VerticalMid'),
                    ((1, 0), 'VerticalMid'),
                ],
                "Cshape": [
                    (0, "Line"),
                    (1, "Line"),
                    (2, "Line"),
                    ((0, 1), 'VerticalEdge'),
                    ((1, 0), 'VerticalEdge'),
                    ((0, 2), 'VerticalEdge'),
                    ((2, 0), 'VerticalEdge'),
                    ((1, 2), 'Parallel'),
                    ((2, 1), 'Parallel'),
                ],
                "Ashape": [
                    (0, "Line"),
                    (1, "Line"),
                    (2, "Line"),
                    (3, "Line"),
                    ((0, 1), 'VerticalEdge'),
                    ((1, 0), 'VerticalEdge'),
                    ((0, 2), 'VerticalMid'),
                    ((2, 0), 'VerticalMid'),
                    ((0, 3), 'Parallel'),
                    ((3, 0), 'Parallel'),
                    ((1, 2), 'Parallel'),
                    ((2, 1), 'Parallel'),
                    ((1, 3), 'VerticalEdge'),
                    ((3, 1), 'VerticalEdge'),
                    ((2, 3), 'VerticalMid'),
                    ((3, 2), 'VerticalMid'),
                ],
                "Hshape": [
                    (0, "Line"),
                    (1, "Line"),
                    (2, "Line"),
                    ((0, 1), 'VerticalMid'),
                    ((1, 0), 'VerticalMid'),
                    ((1, 2), 'VerticalMid'),
                    ((2, 1), 'VerticalMid'),
                    ((0, 2), 'Parallel'),
                    ((2, 0), 'Parallel'),
                ],
                "Rect": [
                    (0, "Line"),
                    (1, "Line"),
                    (2, "Line"),
                    (3, "Line"),
                    ((0, 1), 'VerticalEdge'),
                    ((1, 0), 'VerticalEdge'),
                    ((1, 2), 'VerticalEdge'),
                    ((2, 1), 'VerticalEdge'),
                    ((2, 3), 'VerticalEdge'),
                    ((3, 2), 'VerticalEdge'),
                    ((0, 3), 'VerticalEdge'),
                    ((3, 0), 'VerticalEdge'),
                    ((0, 2), 'Parallel'),
                    ((2, 0), 'Parallel'),
                    ((1, 3), 'Parallel'),
                    ((3, 1), 'Parallel'),
                ],
                "RectE1a": [
                    (0, "Rect"),
                    (1, "Rect"),
                    (2, "Eshape"),
                    ((0, 1), "IsInside"),
                    ((1, 0), "IsEnclosed"),
                    ((0, 2), "IsNonOverlapXY"),
                    ((2, 0), "IsNonOverlapXY"),
                    ((1, 2), "IsNonOverlapXY"),
                    ((2, 1), "IsNonOverlapXY"),
                ],
                "RectE1b": [
                    (0, "Rect"),
                    (1, "Rect"),
                    (2, "Eshape"),
                    ((0, 1), "IsInside"),
                    ((1, 0), "IsEnclosed"),
                    ((0, 2), "IsNonOverlapXY"),
                    ((2, 0), "IsNonOverlapXY"),
                    ((1, 2), "IsNonOverlapXY"),
                    ((2, 1), "IsNonOverlapXY"),
                ],
                "RectE2a": [
                    (0, "Rect"),
                    (1, "Rect"),
                    (2, "Eshape"),
                    ((0, 1), "IsInside"),
                    ((1, 0), "IsEnclosed"),
                    ((2, 1), "IsInside"),
                    ((1, 2), "IsEnclosed"),
                    ((0, 2), "IsNonOverlapXY"),
                    ((2, 0), "IsNonOverlapXY"),
                ],
                "RectE2b": [
                    (0, "Rect"),
                    (1, "Rect"),
                    (2, "Eshape"),
                    ((0, 1), "IsInside"),
                    ((1, 0), "IsEnclosed"),
                    ((2, 1), "IsInside"),
                    ((1, 2), "IsEnclosed"),
                    ((0, 2), "IsNonOverlapXY"),
                    ((2, 0), "IsNonOverlapXY"),
                ],
                "RectE3a": [
                    (0, "Rect"),
                    (1, "Rect"),
                    (2, "Eshape"),
                    ((0, 1), "IsInside"),
                    ((1, 0), "IsEnclosed"),
                    ((2, 1), "IsInside"),
                    ((1, 2), "IsEnclosed"),
                    ((2, 0), "IsInside"),
                    ((0, 2), "IsEnclosed"),
                ],
                "RectE3b": [
                    (0, "Rect"),
                    (1, "Rect"),
                    (2, "Eshape"),
                    ((0, 1), "IsInside"),
                    ((1, 0), "IsEnclosed"),
                    ((2, 1), "IsInside"),
                    ((1, 2), "IsEnclosed"),
                    ((2, 0), "IsInside"),
                    ((0, 2), "IsEnclosed"),
                ],
                "RectEconcept": [
                    (0, "Rect"),
                    (1, "Rect"),
                    (2, "Eshape"),
                ],
                "Graph1": [
                    (0, "Red"),
                    ((0,1), "SameColor"),
                    ((1,2), "SameShape"),
                    ((1,0), "SameColor"),
                    ((2,1), "SameShape"),
                ],
                "Graph2": [
                    (0, "Large"),
                    ((0,1), "SameSize"),
                    ((0,2), "SameColor"),
                    ((1,0), "SameSize"),
                    ((2,0), "SameColor"),   
                ],
                "Graph3": [
                    (0, "Cube"),
                    ((0,1), "SameShape"),
                    ((1,2), "SameSize"),
                    ((1,0), "SameShape"),
                    ((2,1), "SameSize"),
                ],
            }
    else:
        if not is_bidirectional_re:
            graph_gt_dict = {
                "Eshape": [
                    (0, "Line"),
                    (1, "Line"),
                    (2, "Line"),
                    (3, "Line"),
                    ((0, 1), 'Vertical'),
                    ((0, 2), 'Vertical'),
                    ((0, 3), 'Vertical'),
                    ((1, 2), 'Parallel'),
                    ((1, 3), 'Parallel'),
                    ((2, 3), 'Parallel'),
                ],
                "Fshape": [
                    (0, "Line"),
                    (1, "Line"),
                    (2, "Line"),
                    ((0, 1), 'Vertical'),
                    ((0, 2), 'Vertical'),
                    ((1, 2), 'Parallel'),  
                ],
                "Lshape": [
                    (0, "Line"),
                    (1, "Line"),
                    ((0, 1), 'Vertical'),
                ],
                "Tshape": [
                    (0, "Line"),
                    (1, "Line"),
                    ((0, 1), 'Vertical'),
                ],
                "Cshape": [
                    (0, "Line"),
                    (1, "Line"),
                    (2, "Line"),
                    ((0, 1), 'Vertical'),
                    ((0, 2), 'Vertical'),
                    ((1, 2), 'Parallel'),  
                ],
                "Ashape": [
                    (0, "Line"),
                    (1, "Line"),
                    (2, "Line"),
                    (3, "Line"),
                    ((0, 1), 'Vertical'),
                    ((0, 2), 'Vertical'),
                    ((0, 3), 'Parallel'),
                    ((1, 2), 'Parallel'),
                    ((1, 3), 'Vertical'),
                    ((2, 3), 'Vertical'),
                ],
                "Hshape": [
                    (0, "Line"),
                    (1, "Line"),
                    (2, "Line"),
                    ((0, 1), 'Vertical'),
                    ((1, 2), 'Vertical'),
                    ((0, 2), 'Parallel'),
                ],
                "Rect": [
                    (0, "Line"),
                    (1, "Line"),
                    (2, "Line"),
                    (3, "Line"),
                    ((0, 1), 'Vertical'),
                    ((1, 2), 'Vertical'),
                    ((2, 3), 'Vertical'),
                    ((0, 3), 'Vertical'),
                    ((0, 2), 'Parallel'),
                    ((1, 3), 'Parallel'),
                ],
                "RectE1a": [
                    (0, "Rect"),
                    (1, "Rect"),
                    (2, "Eshape"),
                    ((0,1), "IsInside"),
                ],
                "RectE2a": [
                    (0, "Rect"),
                    (1, "Rect"),
                    (2, "Eshape"),
                    ((0,1), "IsInside"),
                    ((2,1), "IsInside"),
                ],
                "RectE3a": [
                    (0, "Rect"),
                    (1, "Rect"),
                    (2, "Eshape"),
                    ((0,1), "IsInside"),
                    ((2,1), "IsInside"),
                    ((2,0), "IsInside"),
                ],
            }
        else:
            raise
    graph = graph_gt_dict[c_type]
    if not is_concept:
        graph = [ele for ele in graph if not isinstance(ele[0], Number)]
    if not is_relation:
        graph = [ele for ele in graph if not isinstance(ele[0], tuple)]
    return graph


def get_reverse_re(r_type):
    """Get the relation in the reverse direction."""
    Dict = {
        "Parallel": "Parallel",
        "Vertical": "Vertical",
        "VerticalMid": "VerticalMid",
        "VerticalEdge": "VerticalEdge",
        "SameColor": "SameColor",
        "SameShape": "SameShape",
        "SameAll": "SameAll",
        "IsInside": "IsOutside",
        "SameRow": "SameRow",
        "SameCol": "SameCol",
        "IsTouch": "IsTouch",
    }
    return Dict[r_type]


def get_penalty_dict(graph, threshold=0.35):
    penalty_dict = {}
    penalty_value_dict = {}
    invalid_relations = []
    for item in graph:
        if isinstance(item[0], tuple):
            if item[2] > threshold:
                invalid_relations.append(item)
                record_data(penalty_dict, [1, 1], [item[0][0], item[0][1]])
                record_data(penalty_value_dict, [item[2], item[2]], [item[0][0], item[0][1]])
    penalty_dict = transform_dict(penalty_dict, "sum")
    penalty_value_dict = transform_dict(penalty_value_dict, "sum")
    return penalty_dict, penalty_value_dict


def remove_node(graph, key):
    id_to_remove = []
    removed_items = []
    for id, item in enumerate(graph):
        if isinstance(item[0], Number):
            if item[0] == key:
                id_to_remove.append(id)
        else:
            assert isinstance(item, tuple)
            if item[0][0] == key or item[0][1] == key:
                id_to_remove.append(id)
    for id in reversed(id_to_remove):
        removed_items.insert(0, graph.pop(id))
    return graph, removed_items


def filter_graph_with_threshold(graph, mode="greedy", threshold=0.35):
    """Filter graph based on threshold."""
    def get_argmax_key(Dict):
        id = np.argmax(list(Dict.values()))
        key = list(Dict)[id]
        return key
    graph = deepcopy(graph)
    if threshold is None:
        return graph, []
    removed_items = []
    if mode == "greedy":
        for i in range(len(graph)):
            penalty_dict, penalty_value_dict = get_penalty_dict(graph, threshold=threshold)
            if len(penalty_value_dict) == 0:
                break
            key_to_remove = get_argmax_key(penalty_value_dict)
            graph, removed_item = remove_node(graph, key_to_remove)
            removed_items += removed_item
    else:
        raise
    return graph, removed_items


def filter_graph_with_masks(graph, masks_is_invalid):
    """Filter the graph with given masks_is_invalid.

    Args:
        masks_is_invalid: has the format of e.g. [True, False, True, False, False].
    """
    invalid_ids = np.where(masks_is_invalid)[0]
    removed_indices = []
    for i in range(len(graph)):
        node_or_edge = graph[i][0]
        if isinstance(node_or_edge, tuple):
            # An edge:
            to_remove = False
            for k in invalid_ids:
                if k in node_or_edge:
                    to_remove = True
                    break
        else:
            # A node:
            to_remove = node_or_edge in invalid_ids
        if to_remove:
            removed_indices.append(i)
    new_graph = [graph[i] for i in range(len(graph)) if i not in removed_indices]
    return new_graph


def random_mask_obj(input, prob):
    p = 1 / 2 / (1 - prob)
    mask = input.argmax(1) != 0
    out = mask.float() * (torch.rand(mask.shape) * p).round().to(mask.device)
    return out.round()[None]


def random_obj(input, concept_prob_dict):
    """Choose random object based on the concept type."""
    list_of_objs = find_connected_components_colordiff(onehot_to_RGB(input).squeeze(), is_mask=True)
    Dict = {}
    for i, obj in enumerate(list_of_objs):
        mask = (obj[0].max(0)[0] != 0).float()
        c_type = classify_concept(mask)
        Dict[i] = {"c_type": c_type, "prob": concept_prob_dict[c_type]}
    prob_list = np.array([ele["prob"] for ele in Dict.values()])
    prob_list = prob_list / prob_list.sum()
    id = np.random.choice(len(prob_list), p=prob_list)
    mask = torch.FloatTensor(list_of_objs[id][2]).to(device)
    return mask[None,None]


def get_rand_graph(freq_dict, is_new_vertical=False):
    line_num, counts = np.unique(freq_dict["Line"], return_counts=True)
    line_freq = counts / counts.sum()
    lines = np.random.choice(line_num, p=line_freq)
    graph_list = []
    for i in range(lines):
        graph_list.append((i, "Line"))
    id_pairs = list(zip(*get_triu_ids(lines)))
    for id_pair in id_pairs:
        # re_chosen = np.random.choice(["VerticalMid", "VerticalEdge", "Parallel"], p=[1/3, 1/3, 1/3])
        if is_new_vertical:
            re_chosen = np.random.choice(["VerticalMid", "VerticalEdge", "Parallel"], p=freq_ver_para)
        else:
            re_chosen = np.random.choice(["Vertical", "Parallel"], p=freq_ver_para)
        graph_list.append((id_pair, re_chosen))
    return graph_list


def get_c_type_from_data(data):
    if isinstance(data[2], tuple):
        assert len(np.unique(data[2])) == 1
        # Requiring that all data has the same positive labels:
        assert data[2][0] == data[3][0]["obj_spec"][0][0][1].split("_")[0]
        return data[2][0]
    else:
        assert data[2] == data[3]["obj_spec"][0][0][1].split("_")[0]
        return data[2]


def random_obj(input, concept_prob_dict=None):
    """Choose random object based on the concept type."""
    list_of_objs = find_connected_components_colordiff(onehot_to_RGB(input).squeeze(), is_mask=True)
    id = np.random.choice(len(list_of_objs))
    mask = torch.FloatTensor(list_of_objs[id][2]).to(device)
    return mask[None,None]


def shorten_graph(graph):
    return [ele[:3] for ele in graph]


def is_mask_invalid(mask, threshold_pixels=0):
    n_pixels = (mask.round() > 0).long().sum().item()
    return n_pixels <= threshold_pixels


def get_fewshot_gt_idx(dataloader):
    gt_idx_list = []
    for data in dataloader:
        concept_id = data[2]["concept_id"][0]
        example_concept_ids = [ele[0] for ele in data[1][2]]
        gt_idx = example_concept_ids.index(concept_id)
        gt_idx_list.append(gt_idx)
    gt_idx_list = np.array(gt_idx_list)
    return gt_idx_list


def update_default_hyperparam(Dict):
    """Default hyperparameters for previous experiments, after adding these new options."""
    default_param = {
        "is_round_mask": True,
        "SGLD_is_penalize_lower": "False",
        "val_batch_size": 1,
    }
    for key, item in default_param.items():
        if key not in Dict:
            Dict[key] = item
    return Dict


def update_default_hyperparam_generalization(Dict):
    """Default hyperparameters for previous experiments, after adding these new options."""
    default_param = {
        "init": "random",
        "SGLD_is_penalize_lower_seq": "None",
        "lambd": 0.005,
        "infer_order": "simul",
        "is_bidirectional_re": False,
        "min_n_distractors": 0,
        "max_n_distractors": 3,
        "allow_connect": True,
        "is_concept": True,
        "is_relation": True,
        "is_proper_size": False,
        "is_harder_distractor": False,
        "id": "None",
    }
    for key, item in default_param.items():
        if key not in Dict:
            Dict[key] = item
    return Dict


def get_selector_SGLD(
    selector,
    input,
    args,
    neg_mask=None,
    mask_exclude=None,
    ebm_target="mask",
    ensemble_size=16,
    sample_step=150,
    topk=1,
    isplot=1,
    init="mask-random",
    **kwargs
):
    """
    Performs SGLD on the mask or img, given the selector.

    Returns:
        imgs_top: each element has shape of [topk, B, 10, H, W]
        masks_top: each element has shape of [topk, B, 1, H, W]
    """
    def init_neg_mask(pos_img, init, args):
        """Initialize negative mask"""
        assert not isinstance(pos_img, tuple) and len(pos_img.shape) == 4
        if init == "mask-random":
            assert pos_img.shape[1] == 10
            pos_img_l = repeat_n(pos_img, n_repeats=ensemble_size)
            neg_mask = tuple(torch.rand(pos_img_l.shape[0], 1, *pos_img_l.shape[2:]).to(pos_img.device) * (pos_img_l.argmax(1)[:, None] != 0) for k in range(selector.mask_arity))
            for k in range(len(neg_mask)):
                neg_mask[k].requires_grad = True
        elif init == "random":
            neg_mask = None
        else:
            raise
        return neg_mask

    if neg_mask is None:
        neg_mask = init_neg_mask(input, init, args)
    is_grad = False
    args = deepcopy(args)
    
    for key, value in kwargs.items():
        if value is not None:
            setattr(args, key, value)
    args.ebm_target = ebm_target

    (img_ensemble, neg_mask_ensemble, _, _, _), neg_out_list_ensemble, info_ensemble = neg_mask_sgd_ensemble(
        selector, input, neg_mask, c_repr=None, z=None, zgnn=None, wtarget=None, args=args,
        mask_info={"mask_exclude": mask_exclude} if mask_exclude is not None else None,
        ensemble_size=ensemble_size, is_grad=is_grad,
        out_mode="all-sorted",
        return_history=True,
        record_interval=2,
    )
    masks_top = tuple(neg_mask_ensemble[k][:topk] for k in range(len(neg_mask_ensemble)))
    if ebm_target in ["image+mask"]:
        imgs_top = img_ensemble[:topk]
    else:
        imgs_top = None
    energy_mask = {}
    info = {}
    energy = neg_out_list_ensemble[-1][:topk]
    for key in selector.info:
        energy_mask[key] = selector.info[key][:topk]  # each with [topk, B]
    # energy_mask = np.stack(list(energy_mask.values()), -1)
    info["mutual_exclusive_list"] = info_ensemble["mutual_exclusive_list"][-1][:topk]
    info["energy"] = energy
    info["energy_mask"] = energy_mask
    info["mask_list"] = tuple(mask_list[:,:topk] for mask_list in info_ensemble["neg_mask_list"])
    refer_node_names = selector.refer_node_names if selector.refer_node_names is not None else list(selector.nodes)
    if isplot >= 1:
        # energy_str = ["{:.4f}".format(ele) for ele in energy]
        if ebm_target == "mask":
            for j in range(len(input)):
                if input[j].shape[-3] == 10:
                    visualize_matrices([input[j].argmax(0)], use_color_dict=True)
                elif input[j].shape[-3] == 3:
                    visualize_matrices([input[j]], use_color_dict=False)
                else:
                    raise
                energy_mask_j = {key: value[:,j] for key, value in energy_mask.items()}
                pp.pprint(energy_mask_j)
                # print(["[E={:.4f}]".format(ele) if list(selector.topological_sort)[k] in refer_node_names else "E={:.4f}".format(ele) for k, ele in enumerate(energy_mask[j])])
                if "mutual_exclusive_list" in info:
                    print("mutual_exclusive: {}".format(info["mutual_exclusive_list"][:,j]))
                for k in range(topk):
                    print("top {}, energy: {:.3f}   mutual_exclusive: {:.3f}:".format(k, energy[k,j], info["mutual_exclusive_list"][k,j]))
                    subtitles = ["E={:.3f}".format(value[k]) for key, value in energy_mask_j.items() if "(" not in key]
                    if len(subtitles) < len(masks_top):
                        subtitles += [None for _ in range(len(masks_top) - len(subtitles))]
                    plot_matrices(torch.stack(masks_top)[:,k,j].squeeze(-3), scale_limit=(0,1), images_per_row=6,
                                  subtitles=subtitles)
        elif ebm_target == "image+mask":
            for j in range(len(input)):
                energy_mask_j = {key: value[:,j] for key, value in energy_mask.items()}
                pp.pprint(energy_mask_j)
                if "mutual_exclusive_list" in info:
                    print("mutual_exclusive: {}".format(info["mutual_exclusive_list"][:,j]))
                for k in range(topk):
                    visualize_matrices([imgs_top[k,j].argmax(0)], is_color_dict=True, subtitles=["E={:.4f}".format(energy[k,j])])
                    plot_matrices(torch.stack(masks_top)[:,k,j].squeeze(-3), scale_limit=(0,1), images_per_row=6,
                                  subtitles=["E={:.3f}".format(value[k]) for key, value in energy_mask_j.items()])
        else:
            raise
    return imgs_top, masks_top, info


def parse_selector_from_image(
    input,
    args,
    keys_dict,
    topk=3,
    isplot=False,
    infer_order="simul",
    init="mask-random",
    threshold_pixels=0,
):
    """Obtain selector from parsing the image."""
    if infer_order == "simul":
        selector = get_selector_for_parsing(keys_dict, ebm_dict, CONCEPTS, OPERATORS)
        # Parse the graph
        _, masks_top, info = get_selector_SGLD(
            selector, input, args,
            ebm_target="mask",
            init=init,
            SGLD_object_exceed_coef=0,
            SGLD_mutual_exclusive_coef=args.SGLD_mutual_exclusive_coef,
            SGLD_pixel_entropy_coef=args.SGLD_pixel_entropy_coef,
            ensemble_size=args.ensemble_size,
            sample_step=args.sample_step,
            topk=topk,
            isplot=isplot,
        )
    elif infer_order == "seq":
        # Parse concepts sequentially:
        masks_top_sum = None
        if input.shape[1] == 10:
            assert len(input.shape) == 4
            masks_all_gt = input[:,:1] != 1
        masks_collect = []
        key_list = []
        for key in keys_dict:
            key_list += [key] * keys_dict[key]
        key_list = np.random.permutation(key_list)
        for key in key_list:
            selector = get_selector_for_parsing({key: 1}, ebm_dict, CONCEPTS, OPERATORS)
            _, masks_top, info = get_selector_SGLD(
                selector, input, args,
                ebm_target="mask",
                mask_exclude=masks_top_sum,
                init=init,
                SGLD_object_exceed_coef=0,
                SGLD_mutual_exclusive_coef=args.SGLD_mutual_exclusive_coef,
                SGLD_pixel_entropy_coef=args.SGLD_pixel_entropy_coef,
                SGLD_is_penalize_lower=args.SGLD_is_penalize_lower_seq,
                ensemble_size=args.ensemble_size,
                sample_step=args.sample_step,
                topk=1,
                isplot=isplot,
            )
            for mask in masks_top:
                if mask.round().sum() > 0:
                    masks_collect.append(repeat_n(mask.squeeze(1), n_repeats=args.ensemble_size))
            if masks_top_sum is None:
                masks_top_sum = torch.stack([mask.detach() for mask in masks_top]).sum(0).clamp(0, 1).round().squeeze(1)
            else:
                masks_top_sum = (masks_top_sum + torch.stack([mask.detach() for mask in masks_top]).sum(0).squeeze(1).round()).clamp(0, 1)
            if isplot:
                print("masks_sum:")
                plot_matrices(masks_top_sum.squeeze(1))
            if input.shape[1] == 10:
                if np.clip(to_np_array(masks_all_gt) - to_np_array(masks_top_sum), a_min=0, a_max=None).sum() == 0:
                    if isplot:
                        print("All objects accounted for. Break.")
                        break
        # Fine-tune with all concepts:
        selector = get_selector_for_parsing(keys_dict, ebm_dict, CONCEPTS, OPERATORS)
        # Parse the graph:
        _, masks_top, info = get_selector_SGLD(
            selector, input, args,
            neg_mask=tuple(masks_collect),
            ebm_target="mask",
            init=init,
            lambd_start=args.lambd,
            lambd=args.lambd,
            SGLD_object_exceed_coef=0,
            SGLD_mutual_exclusive_coef=args.SGLD_mutual_exclusive_coef,
            SGLD_pixel_entropy_coef=args.SGLD_pixel_entropy_coef,
            ensemble_size=args.ensemble_size,
            sample_step=min(60, args.sample_step),
            topk=topk,
            isplot=isplot,
        )

    else:
        raise Exception("infer_order '{}' is not valid!".format(infer_order))
    graph_dict = {k: [] for k in range(topk)}
    graph_trimmed_dict = {}
    input_expand = input.expand(topk, *input.shape[1:])
    for i, node_name in enumerate(selector.topological_sort):
        c_type = node_name.split(":")[-1]
        E_c = ebm_dict[c_type](input_expand, (masks_top[i].squeeze(1),))
        for k in range(topk):
            graph_dict[k].append((i, c_type, E_c[k].item()))

    id_pairs = list(zip(*get_triu_ids(len(masks_top))))
    for id0, id1 in id_pairs:
        E_re = {}
        for re_key in re_keys:
            if args.is_round_mask:
                # Each value has shape of [topk, 1]
                if args.is_bidirectional_re:
                    E_re[re_key] = to_np_array(ebm_dict[re_key]((input_expand, input_expand), (masks_top[id0].round().squeeze(1), masks_top[id1].round().squeeze(1))) +
                                               ebm_dict[get_reverse_re(re_key)]((input_expand, input_expand), (masks_top[id1].round().squeeze(1), masks_top[id0].round().squeeze(1))))
                else:
                    E_re[re_key] = to_np_array(ebm_dict[re_key]((input_expand, input_expand), (masks_top[id0].round().squeeze(1), masks_top[id1].round().squeeze(1))))
            else:
                if args.is_bidirectional_re:
                    E_re[re_key] = to_np_array(ebm_dict[re_key]((input_expand, input_expand), (masks_top[id0].squeeze(1), masks_top[id1].squeeze(1))) +
                                               ebm_dict[get_reverse_re(re_key)]((input_expand, input_expand), (masks_top[id1].squeeze(1), masks_top[id0].squeeze(1))))
                else:
                    E_re[re_key] = to_np_array(ebm_dict[re_key]((input_expand, input_expand), (masks_top[id0].squeeze(1), masks_top[id1].squeeze(1))))
        E_re_all = np.concatenate(list(E_re.values()), -1)
        idx_argmin = E_re_all.argmin(-1)
        re_argmin = np.array(list(E_re))[idx_argmin].tolist()
        E_min = np.take_along_axis(E_re_all, indices=idx_argmin[:,None], axis=-1).squeeze()
        for k in range(topk):
            graph_dict[k].append(((id0, id1), re_argmin[k], E_min[k]))

    for k in range(topk):
        masks_is_invalid = [is_mask_invalid(mask[k], threshold_pixels=threshold_pixels) for mask in masks_top]
        graph_trimmed_dict[k] = filter_graph_with_masks(graph_dict[k], masks_is_invalid=masks_is_invalid)
    return graph_trimmed_dict, graph_dict, tuple(mask.squeeze(1) for mask in masks_top), info


# In[ ]:


# Make sure that the initialization of CONCEPTS and OPERATORS (including their embedding) is after setting the seed
set_seed(1)
from zeroc.concepts_shapes import OPERATORS, CONCEPTS, load_task, seperate_concept
try:
    get_ipython().run_line_magic('matplotlib', 'inline')
    is_jupyter = True
except:
    is_jupyter = False


# # 2. Inference:

# ### 2.1 Loading atomic EBMs:

# In[ ]:


is_load = False
date_time = f"{datetime.now().month}-{datetime.now().day}"
if is_load:
    all_dict = pload(EXP_PATH + "/generalization/all_dict_April11.p")
    models = all_dict["models"]
    CONCEPTS = all_dict["CONCEPTS"]
    OPERATORS = all_dict["OPERATORS"]
    SGLD_args = all_dict["args"]
else:
    if "models" not in locals():
        models = {}
    device = "cpu"

    all_dict = {
        "models": models,
        "CONCEPTS": CONCEPTS,
        "OPERATORS": OPERATORS,
        # "args": info["args"],
    }
    for key, model in models.items():
        model.to("cpu")
    make_dir(EXP_PATH + f"/generalization/all_dict_{date_time}.p")
    pdump(all_dict, EXP_PATH + f"/generalization/all_dict_{date_time}.p")


# ### 2.2 Settings:

# In[ ]:


parser = argparse.ArgumentParser(description='PDE argparse.')
parser.add_argument('--evaluation_type', type=str, help='Evaluation type')
parser.add_argument('--dataset', type=str, default="h-r^2ai+2a+3ai+3a+3b:SameShape+SameColor(Line+Rect+RectSolid+Lshape)", help='dataset')
parser.add_argument('--canvas_size', type=int, default=16, help='Canvas_size')
parser.add_argument('--canvas_size_3D', type=int, default=32, help='Canvas_size')
parser.add_argument('--color_avail', type=str, default="1,2", help='color_avail.')
parser.add_argument('--max_n_distractors', type=int, default=3, help='Maximum number of distractors for grounding.')
parser.add_argument('--min_n_distractors', type=int, default=0, help='Minimum number of distractors for grounding.')
parser.add_argument('--allow_connect', type=str2bool, nargs='?', const=True, default=True,
                        help='Whether or not to allow objects to connect in the image.')
parser.add_argument('--is_proper_size', type=str2bool, nargs='?', const=True, default=False,
                        help='If True, the main character for grounding will have a proper size.')
parser.add_argument('--is_harder_distractor', type=str2bool, nargs='?', const=True, default=False,
                        help='If True, will use harder distractors.')
parser.add_argument('--model_type', type=str, default="hc-ebm", help='Model type.')
parser.add_argument('--SGLD_mutual_exclusive_coef', type=float, default=0, help='SGLD_mutual_exclusive_coef.')
parser.add_argument('--SGLD_pixel_entropy_coef', type=float, default=0, help='SGLD_pixel_entropy_coef.')
parser.add_argument('--SGLD_is_anneal', type=str2bool, nargs='?', const=True, default=True,
                        help='If True, will anneal the SGLD coefficients.')
parser.add_argument('--SGLD_is_penalize_lower', type=str, default="False", help='if True or "True", will penalize that the sum is less than 1. If "False" or False, will not. If "obj", will only penalize on the object locations (if n_channels==10)..')
parser.add_argument('--SGLD_is_penalize_lower_seq', type=str, default="None", help='if True or "True", will penalize that the sum is less than 1. If "False" or False, will not. If "obj", will only penalize on the object locations (if n_channels==10)..')
parser.add_argument('--concept_model_hash', type=str, default="None", help='hash key for the concept model.')
parser.add_argument('--relation_model_hash', type=str, default="None", help='hash key for the relation model.')
parser.add_argument('--concept_model_hash_3D', type=str, default="None", help='hash key for the concept model.')
parser.add_argument('--relation_model_hash_3D', type=str, default="None", help='hash key for the relation model.')
parser.add_argument('--concept_load_id', type=str, default="best", help='"best" or a number string.')
parser.add_argument('--relation_load_id', type=str, default="best", help='"best" or a number string.')
parser.add_argument('--concept_load_id_3D', type=str, default="best", help='"best" or a number string.')
parser.add_argument('--relation_load_id_3D', type=str, default="best", help='"best" or a number string.')
parser.add_argument('--is_new_vertical', type=str2bool, nargs='?', const=True, default=True,
                        help='If True, will use VertMid, VertEdge.')
parser.add_argument('--is_concept', type=str2bool, nargs='?', const=True, default=True,
                        help='If False, will remove the concept EBMs in the selector.')
parser.add_argument('--is_relation', type=str2bool, nargs='?', const=True, default=True,
                        help='If False, will remove the relation EBMs in the selector.')
parser.add_argument('--is_bidirectional_re', type=str2bool, nargs='?', const=True, default=False,
                        help='If True, each relation will be evaluated on both directions.')
parser.add_argument('--is_round_mask', type=str2bool, nargs='?', const=True, default=True,
                        help='If True, will use VertMid, VertEdge.')
parser.add_argument('--val_batch_size', type=int, default=1, 
                        help='batch_size for validation.')
parser.add_argument('--val_n_examples', type=int, default=400, 
                        help='batch_size for validation.')
parser.add_argument('--is_analysis', type=str2bool, nargs='?', const=True, default=False,
                        help='If True, will perform analysis.')
parser.add_argument('--ensemble_size', type=int, default=16, help='ensemble_size')
parser.add_argument('--sample_step', type=int, default=150, help='ensemble_size')
parser.add_argument('--lambd', type=float, default=0.005, help='noise scale at the end.')
parser.add_argument('--infer_order', type=str, default="simul", help='Choose from "parallel", "sequential".')
parser.add_argument('--init', type=str, default="random", help='Choose from "random", "mask-random"')
parser.add_argument('--load_parse_src', type=str, default="gt", help='Choose from "gt", or a path to a parse file.')
parser.add_argument('--inspect_interval', type=int, default=20, help='inspect interval')
parser.add_argument('--seed', type=int, default=2, help='color_avail.')
parser.add_argument('--topk', type=int, default=16, help='color_avail.')
parser.add_argument('--id', type=str, default="0", help='Id')
parser.add_argument('--gpuid', type=str, help='Id')
parser.add_argument('--date_time', type=str, help='Month and date')
parser.add_argument('--data_range', type=str, default="None", help='"None" or "100:200"')
parser.add_argument('--verbose', type=int, default=1, help='Verbose.')

try:
    get_ipython().run_line_magic('matplotlib', 'inline')
    args = parser.parse_args([])
    args.min_n_distractors = 1
    args.max_n_distractors = 2
    args.is_proper_size = True
    # args.dataset = "pc-Cshape+Eshape+Fshape+Ashape+Hshape+Rect"
    # args.dataset = "pc-RectE1a+RectE2a+RectE3a"
    # args.dataset = "pg-Lshape+Tshape+Cshape+Eshape+Fshape+Ashape+Rect^Eshape+Fshape+Ashape+Rect"
    # args.dataset = "yc-Eshape[5,9]+Fshape[5,9]+Ashape[5,9]"
    args.dataset = "c-Lshape+Tshape+Cshape+Eshape+Fshape+Ashape"
    # args.dataset = "c-Eshape+Fshape+Ashape"
    # args.dataset = "c-RectE1a+RectE2a+RectE3a"
    # args.dataset = "h-r^2ai+2a+3ai+3a+3b:SameShape+SameColor(Line+Rect+RectSolid+Lshape)"
    # args.evaluation_type = "pc-parse+classify^parse" # "grounding-Lshape"   # "parse", "grounding", "classify", "query-1c", "unify", 
    # args.evaluation_type = "classify"
    # args.evaluation_type = "yc-parse+classify"
    # args.dataset = "c-Lshape+Eshape+Fshape+Ashape"
    args.SGLD_mutual_exclusive_coef = 500
    args.SGLD_mask_entropy_coef = 0
    args.model_type = "hc-ebm"
    # args.model_type = "rand-obj"  # "rand-mask-obj" or "rand-obj"
    # args.model_type = "rand-graph"
    args.ensemble_size = 64
    args.allow_connect = False
    args.inspect_interval = 1
    args.sample_step = 150
    args.is_new_vertical = True
    args.is_concept = True
    args.is_bidirectional_re = True
    args.canvas_size = 16
    args.SGLD_is_anneal = True
    args.val_n_examples = 200
    args.SGLD_is_penalize_lower = "False"
    args.SGLD_is_penalize_lower_seq = "False"

    # args.relation_model_hash = "1IB9rf++"  # new3
    # args.relation_model_hash = "BiP7IcmF" # center^stop:0.1 with r-rmbx
    # args.relation_model_hash= "vcLANlzs" # center^stop:0.1 with r-mbx
    ## max_n_distractors=0:
    # args.concept_model_hash = "VRaV6VTb" # randpatch rmb, or "1NHyHlRM" (rmbx)
    # args.relation_model_hash = "cFyJdFYC" # randpatch rmb, or "i2um9RI2" (rmbx)
    args.concept_load_id = "best"
    args.relation_load_id = "best"

    ## For classification of Eshape:
    args.evaluation_type = "classify"
    args.dataset = "c-Eshape+Fshape+Ashape"
    args.canvas_size = 16
    args.concept_model_hash = "fRZtzn33"
    args.relation_model_hash = "Wfxw19nM"

    # ## For grounding Eshape:
    # # args.evaluation_type = "grounding-RectE1a"
    # args.evaluation_type = "grounding-Eshape"
    # args.dataset = "c-Lshape+Tshape+Cshape+Eshape+Fshape+Ashape"
    # args.canvas_size = 16
    # args.concept_model_hash = "fRZtzn33"
    # args.relation_model_hash = "1IB9rf++"
    # args.relation_load_id = 75
    # args.val_n_examples = 400

    # # ## For 3D dataset:
    # args.evaluation_type = "yc-parse+classify^classify"
    # args.concept_model_hash = "fRZtzn33"
    # args.relation_model_hash = "Wfxw19nM"
    # args.concept_model_hash_3D = "jk4HQfir"
    # args.relation_model_hash_3D = "x72bDIyX"
    # args.dataset = "yc-Eshape[5,9]+Fshape[5,9]+Ashape[5,9]"
    # args.canvas_size = 16
    # args.canvas_size_3D = 32
    # args.load_parse_src = "gt"
    # args.load_parse_src = "evaluation_yc-parse+classify^parse_1-21/evaluation_yc-parse+classify^parse_canvas_16_color_1,2_ex_400_min_0_model_hc-ebm_mutu_500.0_ens_64_sas_150_newv_True_batch_1_con_fRZtzn33_re_Wfxw19nM_bi_True_seed_2_id_None_Hash_mU7ILNWm_turing3.p"

    # ## For RectE:
    # args.evaluation_type = "grounding-RectE1a"
    # args.concept_model_hash = "AaalzcSD"
    # args.relation_model_hash = "NAzKCenZ"
    # args.dataset = "c-Lshape+Tshape+Cshape+Eshape+Fshape+Ashape"
    # args.canvas_size = 20
    # args.max_n_distractors = 1
    # args.min_n_distractors = 1
    # args.is_harder_distractor = False
    # # args.concept_load_id = 30
    # # args.relation_load_id = 10
    
    # ## For clevr:
    # args.evaluation_type = "classify"
    # args.dataset = "u-graph-Graph1+Graph2+Graph3"
    # args.canvas_size = 32
    # args.concept_model_hash = "fuXnlqYJ"
    # args.relation_model_hash = "LMBhNjDn"

    args.is_round_mask = True
    # args.lambd = 0.001
    args.infer_order = "simul"

    # args.val_n_examples = 40
    args.val_batch_size = 1
    args.is_analysis = True
    args.init = "random"
    args.seed = 2
    args.verbose = 2
    args.gpuid = "0"
    args.id = "test"
    args.date_time = "{}-{}".format(datetime.now().month, datetime.now().day)
except:
    args = parser.parse_args()
is_input = False
assert args.val_n_examples % args.val_batch_size == 0
device = get_device(args)
dirname = EXP_PATH + "/evaluation_{}_{}/".format(args.evaluation_type, args.date_time)

# Model:
if args.evaluation_type.startswith("query") or args.evaluation_type.startswith("unify"):
    assert args.color_avail == "1,2"
    assert args.dataset.startswith("h-")
    if args.canvas_size == 8:
        concept_model = models["JjloOXSA"]
        relation_model = models["j50bI0im"]
    elif args.canvas_size == 16:
        concept_model = models["l6Gaygnh"]
        relation_model = models["1WKDD+X6"]
    else:
        raise
    if args.evaluation_type.startswith("unify"):
        re_keys = ["SameShape", "SameColor", "IsInside"]
        task_size = 6
elif args.evaluation_type.startswith("parse") or    args.evaluation_type.startswith("grounding") or    args.evaluation_type.startswith("classify") or    args.evaluation_type.startswith("generation") or    args.evaluation_type.startswith("pc-parse+classify") or    args.evaluation_type.startswith("yc-parse+classify"):
    assert args.color_avail == "1,2"
    assert args.dataset.startswith("c-") or args.dataset.startswith("pc-") or args.dataset.startswith("yc-") or args.dataset.startswith("u-")
    if args.evaluation_type.startswith("parse"):
        assert args.dataset == "c-Lshape+Tshape+Cshape+Eshape+Fshape+Ashape"
    elif args.evaluation_type.startswith("pc-parse+classify"):
        assert args.dataset in ["pc-Lshape+Tshape+Cshape+Eshape+Fshape+Ashape+Rect",
                                "pc-Cshape+Eshape+Fshape+Ashape+Hshape+Rect",
                                "pc-RectE1a+RectE2a+RectE3a",
                               ]
        if args.dataset == "pc-RectE1a+RectE2a+RectE3a":
            keys_dict = {"Eshape": 1, "Rect": 2}
        else:
            keys_dict = {"Line": 4}
    elif args.evaluation_type.startswith("yc-parse+classify"):
        assert args.dataset in ["yc-Eshape[5,9]+Fshape[5,9]+Ashape[5,9]"]
        if args.dataset == "yc-Eshape[5,9]+Fshape[5,9]+Ashape[5,9]":
            keys_dict = {"Line": 4}
        else:
            raise
    elif args.evaluation_type.startswith("classify") or args.evaluation_type.startswith("generation"):
        assert args.dataset in ["c-Eshape+Fshape+Ashape", "c-RectE1a+RectE2a+RectE3a", "u-graph-Graph1+Graph2+Graph3"]
        if args.dataset == "c-RectE1a+RectE2a+RectE3a":
            assert args.canvas_size == 20
        if args.dataset.startswith("u-"):
            assert args.canvas_size == 32
    elif args.evaluation_type.startswith("grounding-RectE"):
        assert args.canvas_size == 20
    if args.is_new_vertical:
        re_keys = ["Parallel", "VerticalMid", "VerticalEdge"]
    else:
        re_keys = ["Parallel", "Vertical"]
    task_size = 1
else:
    raise
if args.concept_model_hash != "None":
    models[args.concept_model_hash], info = load_model_hash(
        args.concept_model_hash, return_args=True, isplot=0,
        is_update_repr=True,
        update_keys=[
            "Line",
            "Lshape",
            "Rect",
            "RectSolid",
            "Eshape",
            "Fshape",
            "Red",
            "Blue",
            "Green",
            "Cube",
            "Cylinder",
            "Large",
            "Small",
        ],
        load_epoch=args.concept_load_id,
    )
    concept_model = models[args.concept_model_hash]
    SGLD_args = info["args"]
    args.concept_load_id = info["load_id"]
else:
    args.concept_load_id = "None"
if args.relation_model_hash != "None":
    models[args.relation_model_hash], info = load_model_hash(
        args.relation_model_hash, return_args=True, isplot=0,
        is_update_repr=True,
        update_keys=[
            "SameShape",
            "SameColor",
            "IsInside",
            "VerticalMid",
            "VerticalEdge",
            "Parallel",
            "Vertical",
            "IsEnclosed",
            "IsNonOverlapXY",
            "SameSize",
        ],
        load_epoch=args.relation_load_id,
    )
    relation_model = models[args.relation_model_hash]
    args.relation_load_id = info["load_id"]
else:
    args.relation_load_id = "None"
ebm_dict = Shared_Param_Dict(
    concept_model=concept_model,
    relation_model=relation_model,
).to(device)
for c_str in ["Line", "Lshape", "Rect", "RectSolid", "Eshape", "Fshape", "Red", "Blue", "Green", "Cube", "Cylinder", "Large", "Small"]:
    ebm_dict.add_c_repr(CONCEPTS[c_str].get_node_repr()[None].to(device), c_str, ebm_mode="concept")
for c_str in ["SameShape", "SameColor", "IsInside", "VerticalMid", "VerticalEdge", "Parallel", "Vertical", "SameSize"]:
    ebm_dict.add_c_repr(OPERATORS[c_str].get_node_repr()[None].to(device), c_str, ebm_mode="operator")
if args.evaluation_type.startswith("yc-"):
    if "^" in args.evaluation_type:
        tasks = args.evaluation_type.split("^")[-1].split("+")
    else:
        tasks = ["parse", "classify"]

    if "classify" in tasks:
        models_3D = {}
        print("Loading concept_3D:")
        models_3D[args.concept_model_hash_3D], info = load_model_hash(
            args.concept_model_hash_3D, return_args=True, isplot=0,
            is_update_repr=True, update_keys=["Line", "Lshape", "Rect", "RectSolid", "Eshape", "Fshape"],
            load_epoch=args.concept_load_id_3D,
        )
        concept_model_3D = models_3D[args.concept_model_hash_3D]
        args.concept_load_id_3D = info["load_id"]
        print("Loading relation_3D:")
        models_3D[args.relation_model_hash_3D], info = load_model_hash(
            args.relation_model_hash_3D, return_args=True, isplot=0,
            is_update_repr=True, update_keys=["SameShape", "SameColor", "IsInside", "VerticalMid", "VerticalEdge", "Parallel", "Vertical", "IsEnclosed", "IsNonOverlapXY"],
            load_epoch=args.relation_load_id_3D,
        )
        relation_model_3D = models_3D[args.relation_model_hash_3D]
        args.relation_load_id_3D = info["load_id"]
        ebm_dict_3D = Shared_Param_Dict(
            concept_model=concept_model_3D,
            relation_model=relation_model_3D,
        ).to(device)
        for c_str in ["Line", "Lshape", "Rect", "RectSolid", "Eshape", "Fshape"]:
            ebm_dict_3D.add_c_repr(CONCEPTS[c_str].get_node_repr()[None].to(device), c_str, ebm_mode="concept")
        for c_str in ["SameShape", "SameColor", "IsInside", "VerticalMid", "VerticalEdge", "Parallel", "Vertical"]:
            ebm_dict_3D.add_c_repr(OPERATORS[c_str].get_node_repr()[None].to(device), c_str, ebm_mode="operator")

# Composite dataset, canvas_size=8, up to 3b:
if args.evaluation_type.startswith("yc-parse+classify"):
    composite_args = init_args({
        "dataset": "yc-Eshape[5,9]+Fshape[5,9]+Ashape[5,9]",
        "seed_3d": 42,
        "n_examples": 200,
        "num_processes_3d": 5,
        "n_queries_per_class": 1,
        "color_map_3d": "same",
        "add_thick_surf": (0, 0.5),
        "add_thick_depth": (0, 0.5),
        "image_size_3d": (256, 256),
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
else:
    composite_args = init_args({
        "dataset": args.dataset,
        "seed": 2,
        "n_examples": args.val_n_examples,
        "canvas_size": args.canvas_size,
        "rainbow_prob": 0.,
        "w_type": "image+mask",
        "color_avail": args.color_avail,
        "min_n_distractors": 0,
        "max_n_distractors": 2 if args.evaluation_type.startswith("grounding") else 0,
        "allow_connect": True,
        "parsing_check": True if args.evaluation_type.startswith("grounding") else False,
    })
    
dataset, composite_args = get_dataset(composite_args, is_load=True)
for key, value in get_SGLD_kwargs(SGLD_args).items():
    if key not in ["SGLD_mutual_exclusive_coef",
                   "SGLD_pixel_entropy_coef",
                   "SGLD_is_anneal",
                   "SGLD_is_penalize_lower",
                   "sample_step",
                   "lambd",
                  ]:
        setattr(args, key, value)
args.is_two_branch = SGLD_args.is_two_branch
args.is_image_tuple = False
args.step_size_start = SGLD_args.step_size_start
args.image_size = (args.canvas_size, args.canvas_size)
args.SGLD_anneal_power = 2
args.SGLD_fine_mutual_exclusive_coef = 0
args.SGLD_mask_entropy_coef = 0
args.rescaled_size = "None"
if args.SGLD_is_penalize_lower_seq == "None":
    args.SGLD_is_penalize_lower_seq = args.SGLD_is_penalize_lower
filename = "evaluation_{}_canvas_{}_color_{}_ex_{}_min_{}_model_{}_mutu_{}_ens_{}_sas_{}_newv_{}_batch_{}_con_{}_re_{}_bi_{}_range_{}_seed_{}_id_{}_Hash_{}_{}.p".format(
    args.evaluation_type, args.canvas_size, args.color_avail, args.val_n_examples, args.min_n_distractors, args.model_type, args.SGLD_mutual_exclusive_coef, args.ensemble_size, args.sample_step, args.is_new_vertical, args.val_batch_size, args.concept_model_hash, args.relation_model_hash, args.is_bidirectional_re, args.data_range, args.seed, args.id, get_hashing(str(args.__dict__), length=8), get_machine_name())
print_banner(filename)
pp.pprint(args.__dict__)
data_record = {"args": args.__dict__}


# # 3. Zero-shot tasks:

# ### 3.1 Parse:

# In[ ]:


if args.evaluation_type.startswith("parse"):
    data_record["parsing_all_dict"] = {}
    set_seed(args.seed)
    if args.model_type == "rand-graph":
        freq_dict = {}
        for data in dataset:
            c_type_gt = get_c_type_from_data(data)
            graph_gt = get_concept_graph(c_type_gt, is_new_vertical=args.is_new_vertical)
            n_line = np.sum([1 for ele in graph_gt if ele[-1] == "Line"])
            if args.is_new_vertical:
                n_vertical_mid = np.sum([1 for ele in graph_gt if ele[-1] == 'VerticalMid'])
                n_vertical_edge = np.sum([1 for ele in graph_gt if ele[-1] == 'VerticalEdge'])
            else:
                n_vertical = np.sum([1 for ele in graph_gt if ele[-1] == 'Vertical'])
            n_parallel = np.sum([1 for ele in graph_gt if ele[-1] == 'Parallel'])
            if args.is_new_vertical:
                record_data(freq_dict, [n_line, n_vertical_mid, n_vertical_edge, n_parallel], ["Line", "VerticalMid", "VerticalEdge", "Parallel"])
            else:
                record_data(freq_dict, [n_line, n_vertical, n_parallel], ["Line", "Vertical", "Parallel"])
        freq_mean = transform_dict(freq_dict, "mean")
        if args.is_new_vertical:
            freq_vertical_edge = freq_mean["VerticalEdge"]
            freq_vertical_mid = freq_mean["VerticalMid"]
        else:
            freq_vertical = freq_mean["Vertical"]
        freq_parallel = freq_mean['Parallel']
        if args.is_new_vertical:
            freq_ver_para = np.array([freq_vertical_mid, freq_vertical_edge, freq_parallel])
        else:
            freq_ver_para = np.array([freq_vertical, freq_parallel])
        freq_ver_para = freq_ver_para / freq_ver_para.sum()
    else:
        assert args.model_type == "hc-ebm"

    make_dir(dirname + filename)
    for i in range(len(dataset)):
        data = dataset[i]
        if i % args.inspect_interval == 0 and i > 0:
            print("{}:".format(i))
            isplot = is_jupyter
        else:
            isplot = 0

        if args.model_type == "hc-ebm":
            input = data[0][None].to(device)
            keys_dict = {"Line": 4}
            selector = get_selector_for_parsing(keys_dict, ebm_dict, CONCEPTS, OPERATORS)
            _, masks_top, info = get_selector_SGLD(
                selector, input, args,
                ebm_target="mask",
                init=args.init,
                SGLD_object_exceed_coef=0,
                SGLD_mutual_exclusive_coef=args.SGLD_mutual_exclusive_coef,
                SGLD_pixel_entropy_coef=args.SGLD_pixel_entropy_coef,
                ensemble_size=args.ensemble_size,
                sample_step=args.sample_step,
                isplot=isplot,
            )
            graph = []
            for k, node_name in enumerate(selector.topological_sort):
                c_type = node_name.split(":")[-1]
                E_c = ebm_dict[c_type](input, (masks_top[k],)).item()
                graph.append((k, c_type, E_c))

            id_pairs = list(zip(*get_triu_ids(len(masks_top))))
            for id0, id1 in id_pairs:
                E_re = {}
                for re_key in re_keys:
                    if args.is_round_mask:
                        E_re[re_key] = ebm_dict[re_key]((input, input), (masks_top[id0].round(), masks_top[id1].round())).item()
                    else:
                        E_re[re_key] = ebm_dict[re_key]((input, input), (masks_top[id0], masks_top[id1])).item()

                idx_argmin = np.argmin(list(E_re.values()))
                re_argmin = list(E_re)[idx_argmin]
                E_min = list(E_re.values())[idx_argmin]
                graph.append(((id0, id1), re_argmin, E_min))
        elif args.model_type == "rand-graph":
            graph = get_rand_graph(freq_dict, is_new_vertical=args.is_new_vertical)
            input = None
            masks_top = None
        else:
            raise
        if i % args.inspect_interval == 0:
            p.print("graph:")
            shortened_graph = [ele[:3] for ele in graph]
            p.print(shortened_graph)
            print("\n\n")
        record_data(data_record["parsing_all_dict"],
                    [i, graph, to_cpu_recur(masks_top), to_cpu_recur(input), data[3] if args.evaluation_type.startswith("parse") else data[j][3]],
                    ["batch", "graph", "masks_top", "input", "info"])
        if i % args.inspect_interval == 0 and i > 0 or i == len(dataset) - 1:
            pdump(data_record, dirname + filename)
            p.print("Saved at {}".format(dirname + filename))

    if args.model_type == "rand-graph":
        # Test if the rand-graph obeys the statistics:
        graphs = data_record["parsing_all_dict"]["graph"]
        freq_dict2 = {}
        for graph in graphs:
            n_line = np.sum([1 for ele in graph if ele[-1] == "Line"])
            n_vertical_edge = np.sum([1 for ele in graph if ele[-1] == 'VerticalEdge'])
            n_vertical_mid = np.sum([1 for ele in graph if ele[-1] == 'VerticalMid'])
            n_parallel = np.sum([1 for ele in graph if ele[-1] == 'Parallel'])
            record_data(freq_dict2, [n_line, n_vertical_edge, n_vertical_mid, n_parallel], ["Line", "VerticalEdge", "VerticalMid", "Parallel"])
        freq2_mean = transform_dict(freq_dict2, "mean")
        freq2_vertical_edge = freq2_mean["VerticalEdge"]
        freq2_vertical_mid = freq2_mean["VerticalMid"]
        freq2_parallel = freq2_mean['Parallel']
        freq2_ver_para = np.array([freq2_vertical_mid, freq2_vertical_edge, freq2_parallel])
        freq2_ver_para = freq2_ver_para / freq2_ver_para.sum()


# ### 3.2 Grounding：

# In[ ]:


if args.evaluation_type.startswith("grounding"):
    isplot = is_jupyter
    is_input = False
    c_type = args.evaluation_type.split("-")[-1]

    graph = get_concept_graph(c_type,
                              is_new_vertical=args.is_new_vertical,
                              is_bidirectional_re=args.is_bidirectional_re,
                              is_concept=args.is_concept,
                              is_relation=args.is_relation,
                             )
    query = {"graph": graph}
    set_seed(args.seed)
    allow_connect = False
    if c_type == "Eshape":
        if args.is_proper_size:
            dataset_type = "c-Eshape[6,8]+Cshape+Lshape+Tshape+Rect+RectSolid^Eshape[6,8]"
        else:
            dataset_type = "c-Eshape+Cshape+Lshape+Tshape+Rect+RectSolid^Eshape"
    elif c_type == "Fshape":
        if args.is_proper_size:
            dataset_type = "c-Fshape[6,8]+Cshape+Lshape+Tshape+Rect+RectSolid^Fshape[6,8]"
        else:
            dataset_type = "c-Fshape+Cshape+Lshape+Tshape+Rect+RectSolid^Fshape"
    elif c_type == "Ashape":
        if args.is_proper_size:
            dataset_type = "c-Ashape[6,8]+Cshape+Lshape+Tshape+Rect+RectSolid^Ashape[6,8]"
        else:
            dataset_type = "c-Ashape+Cshape+Lshape+Tshape+Rect+RectSolid^Ashape"
    elif c_type == "RectE1a":
        if args.is_harder_distractor:
            dataset_type = "c-RectE1a+Eshape+Rect+Tshape+Fshape+Ashape^RectE1a"
            allow_connect = True
        else:
            dataset_type = "c-RectE1a+Cshape+Lshape+Tshape+Rect+RectSolid^RectE1a"
    elif c_type == "RectE2a":
        if args.is_harder_distractor:
            dataset_type = "c-RectE2a+Eshape+Rect+Tshape+Fshape+Ashape^RectE2a"
            allow_connect = True
        else:
            dataset_type = "c-RectE2a+Cshape+Lshape+Tshape+Rect+RectSolid^RectE2a"
    elif c_type == "RectE3a":
        if args.is_harder_distractor:
            dataset_type = "c-RectE3a+Eshape+Rect+Tshape+Fshape+Ashape^RectE3a"
            allow_connect = True
        else:
            dataset_type = "c-RectE3a+Cshape+Lshape+Tshape+Rect+RectSolid^RectE3a"
    else:
        raise
    composite_args = init_args({
        "dataset": dataset_type,
        "seed": 2,
        "n_examples": args.val_n_examples,
        "canvas_size": args.canvas_size,
        "rainbow_prob": 0.,
        "w_type": "image+mask",
        "color_avail": args.color_avail,
        "min_n_distractors": args.min_n_distractors,
        "max_n_distractors": args.max_n_distractors,
        "allow_connect": allow_connect,
        "parsing_check": True,
    })
    dataset, composite_args = get_dataset(composite_args, is_load=True)
    selector = get_selector_from_graph(query["graph"], ebm_dict, CONCEPTS, OPERATORS)
    Dict = {"args": args.__dict__}
    dataloader = DataLoader(dataset[:200], batch_size=args.val_batch_size, collate_fn=Batch(is_collate_tuple=True).collate(), shuffle=False)
    make_dir(dirname + filename)
    for i, data in enumerate(dataloader):
        if isplot:
            dataset.draw(i)
        c_type = get_c_type_from_data(data)
        img = data[0].to(device)
        graph = get_concept_graph(c_type,
                                  is_new_vertical=args.is_new_vertical,
                                  is_bidirectional_re=args.is_bidirectional_re,
                                  is_concept=args.is_concept,
                                  is_relation=args.is_relation,
                                 )
        query = {"graph": graph}
        if args.model_type == "hc-ebm":
            selector = get_selector_from_graph(query["graph"], ebm_dict, CONCEPTS, OPERATORS)
            _, masks_top, info = get_selector_SGLD(
                selector, img, args,
                ebm_target="mask",
                init=args.init,
                SGLD_object_exceed_coef=0,
                SGLD_mutual_exclusive_coef=args.SGLD_mutual_exclusive_coef,
                SGLD_pixel_entropy_coef=args.SGLD_pixel_entropy_coef,
                ensemble_size=args.ensemble_size,
                sample_step=args.sample_step,
                isplot=isplot,
            )  # Each element of masks_top: [topk:1, B, 1, H, W]
            mask_composite = torch.stack(masks_top).max(0)[0][0]
            if isplot:
                for i in range(len(info["mask_list"])):
                    print("{} th mask:".format(i))
                    plot_matrices(info["mask_list"][i].squeeze()[::2], images_per_row=15, scale_limit=(0,1))
                print("max mask:")
                plot_matrices(np.stack(info["mask_list"]).max(0).squeeze()[::2], images_per_row=15, scale_limit=(0,1))
            if is_input:
                user_input = input("continue [y/n]:")
                if user_input in ["y", "Y"]:
                    pass
                else:
                    raise
        elif args.model_type == "rand-obj":
            mask_composite = random_obj(img)
        iou = to_np_array(get_soft_IoU(mask_composite.round(), data[1][0].round().to(device), dim=(-3,-2,-1)), full_reduce=False)
        record_data(Dict, [iou], ["iou"])
        record_data(Dict, [img, np.stack(to_np_array(*masks_top, keep_list=True), 2)[0]], ["input", "masks_top"])
        if isplot:
            plot_matrices([mask_composite.squeeze(0)[0], data[1][0].round()[0,0]], images_per_row=6,
                          subtitles=["iou: {:.3f}".format(to_np_array(iou)), None])
        if i % args.inspect_interval == 0 or i == len(dataset) - 1:
            p.print("{}: iou_mean {:.4f}".format(i, np.mean(np.concatenate(Dict["iou"]))))
            try_call(pdump, args=[Dict, dirname + filename], max_exp_time=300)
    Dict["iou"] = np.concatenate(Dict["iou"])
    pdump(Dict, dirname + filename)


# In[ ]:


# input = data[0].to(device)[None]
# graph = get_concept_graph(c_type,
#                           is_new_vertical=args.is_new_vertical,
#                           is_bidirectional_re=args.is_bidirectional_re,
#                           is_concept=args.is_concept,
#                          )
# query = {"graph": graph}
# if args.model_type == "hc-ebm":
#     selector = get_selector_from_graph(query["graph"], ebm_dict, CONCEPTS, OPERATORS)
#     _, masks_top, info = get_selector_SGLD(
#         selector, input, args,
#         ebm_target="mask",
#         init=args.init,
#         SGLD_object_exceed_coef=0,
#         SGLD_mutual_exclusive_coef=args.SGLD_mutual_exclusive_coef,
#         SGLD_pixel_entropy_coef=args.SGLD_pixel_entropy_coef,
#         ensemble_size=args.ensemble_size,
#         sample_step=args.sample_step,
#         isplot=isplot,
#     )  # Each element of masks_top: [topk:1, B, 1, H, W]
#     mask_composite = torch.stack(masks_top).max(0)[0][0]
#     plot_matrices([mask_composite.squeeze(0)[0], data[1][0].round()[0]], images_per_row=6,
#                    subtitles=["iou: {:.3f}".format(to_np_array(iou)), None])


# In[ ]:


# mask2 = torch.zeros(16,16).to(device)
# mask2[5:13, -1:] = 1


# In[ ]:


# mask4 = torch.zeros(16,16).to(device)
# mask4[-4, -6:] = 1


# In[ ]:


# mask1 = torch.zeros(16,16).to(device)
# mask1[5, -6:] = 1


# In[ ]:


# masks_top2= (mask2[None,None],mask1[None,None], masks_top[2][0], mask4[None,None])


# In[ ]:


# plot_matrices(tuple(mask.squeeze() for mask in masks_top2))


# In[ ]:


# if args.is_analysis:
#     c_type = "Parallel"
#     graph = get_concept_graph(c_type)
#     selector = get_selector_from_graph(graph, ebm_dict, CONCEPTS, OPERATORS)

#     col = 4
#     col2 = 10
#     image = torch.zeros(16,16).to(device)
#     image[4:12, col:col+1] = 1
#     image[10:12, col2:col2+1] = 1
#     mask1 = torch.zeros(16, 16).to(device)
#     mask1[4:12, col:col+1] = 1
#     mask2 = torch.zeros(16, 16).to(device)
#     mask2[10:12, col2:col2+1] = 1

#     img = to_one_hot(image).to(device)
#     E = selector(img[None], (mask1[None,None], mask2[None,None]))
#     visualize_matrices([image], subtitles=["E={:.4f}".format(to_np_array(E.item()))])
#     plot_matrices([mask1, mask2])


# In[ ]:


# c_type


# In[ ]:


# if args.is_analysis:
#     c_type = "VerticalEdge"
#     graph = get_concept_graph(c_type)
#     selector = get_selector_from_graph(graph, ebm_dict, CONCEPTS, OPERATORS)
#     row = 15
#     col = 15
#     image = torch.zeros(16,16).to(device)
#     image[4:row, col:col+1] = 1
#     image[row:row+1, 7:15] = 1
#     mask1 = torch.zeros(16, 16).to(device)
#     mask1[4:row, col:col+1] = 1
#     mask2 = torch.zeros(16, 16).to(device)
#     mask2[row:row+1, 7:15] = 1

#     img = to_one_hot(image).to(device)
#     E = selector(img[None], (mask1[None,None], mask2[None,None]))
#     visualize_matrices([image], subtitles=["E={:.4f}".format(to_np_array(E.item()))])
#     plot_matrices([mask1, mask2])


# ### 3.3 Classify:

# In[ ]:


if args.evaluation_type.startswith("classify"):
    set_seed(args.seed)
    if args.data_range == "None":
        include_ids = None
    else:
        include_ids = list(np.arange(*([int(ele) for ele in args.data_range.split(":")])))
    isplot = is_jupyter
    if args.dataset.startswith("u-"):
        c_types = args.dataset.split("-")[-1].split("+")
    else:
        c_types = args.dataset.split("-")[1].split("+")
    c_type_graphs = {c_type: get_concept_graph(c_type,
                                               is_new_vertical=args.is_new_vertical,
                                               is_bidirectional_re=args.is_bidirectional_re,
                                               is_concept=args.is_concept,
                                               is_relation=args.is_relation,
                                              ) for c_type in c_types}
    selectors = {c_type: get_selector_from_graph(graph, ebm_dict, CONCEPTS, OPERATORS) for c_type, graph in c_type_graphs.items()}
    Dict = {"args": args.__dict__, "results": {}}
    dataloader = DataLoader(dataset[:200], batch_size=args.val_batch_size, collate_fn=Batch(is_collate_tuple=True).collate(), shuffle=False)
    alpha_list = np.linspace(0,1,21).round(3)
    make_dir(dirname + filename)
    for i, data in enumerate(dataloader):
        if include_ids is not None:
            if i not in include_ids:
                print(f"Skip: {i}")
                continue
        if i % args.inspect_interval == 0:
            p.print("iter: {}".format(i))
        energy_dict = {}
        for key, selector in selectors.items():
            img = data[0].to(device)
            if args.dataset.startswith("u-"):
                img = F.interpolate(img, size=(32,32), mode="nearest")
            if isplot:
                if data[2][0] == key:
                    p.print("{}: [{}]:".format(i, key), banner_size=50)
                else:
                    p.print("{}: {}:".format(i, key), banner_size=50)
            _, masks_top, info = get_selector_SGLD(
                selector, img, args,
                ebm_target="mask",
                init=args.init,
                SGLD_object_exceed_coef=0,
                SGLD_mutual_exclusive_coef=args.SGLD_mutual_exclusive_coef,
                SGLD_pixel_entropy_coef=args.SGLD_pixel_entropy_coef,
                ensemble_size=args.ensemble_size,
                sample_step=args.sample_step,
                isplot=isplot,
            )  # mask_top: each element [topk, B, 1, H, W]
            energy_dict[key] = {}
            energy_dict[key]["masks_top"] = np.stack(to_np_array(*masks_top, keep_list=True), 2)[0]  # [B, n_masks, 1, H, W]
            energy_dict[key]["energy_mask"] = np.stack(list(info['energy_mask'].values()), 2)[0]  # [B, n_masks_concept_re]
            energy_dict[key]["energy_mean"] = np.mean(energy_dict[key]["energy_mask"], -1)  # [B]
            energy_dict[key]["energy_sum"] = np.sum(energy_dict[key]["energy_mask"], -1)  # [B]
            energy_dict[key]["mutual_exclusive"] = info['mutual_exclusive_list'][0]
            for alpha in alpha_list:
                energy_dict[key]["energy_total^m:{:.2f}".format(alpha)] = energy_dict[key]["energy_mean"] + energy_dict[key]["mutual_exclusive"] * alpha
                energy_dict[key]["energy_total^s:{:.2f}".format(alpha)] = energy_dict[key]["energy_sum"] + energy_dict[key]["mutual_exclusive"] * alpha
            
            gt_mask = to_np_array(img[:,:1] != 1)
            masks_top_sum = np.stack(to_np_array(*masks_top, keep_list=True)).sum(0)[0]
            energy_dict[key]["pixels_under"] = np.clip(gt_mask - masks_top_sum, a_min=0, a_max=None).sum((-3,-2,-1))
            energy_dict[key]["pixels_exceed_out"] = np.clip((masks_top_sum - gt_mask) * (1-gt_mask), a_min=0, a_max=None).sum((-3,-2,-1))
            batch_size = masks_top_sum.shape[0]
            energy_dict[key]["pixels_exceed_out_mean"] = np.clip((masks_top_sum - gt_mask).reshape(batch_size, -1)[:, (1-gt_mask).reshape(-1).astype(bool)], a_min=0, a_max=None).mean((-1))
            energy_dict[key]["pixels_exceed_in"] = np.clip((masks_top_sum - gt_mask) * gt_mask, a_min=0, a_max=None).sum((-3,-2,-1))

            for subkey in energy_dict[key]:
                record_data(Dict["results"], energy_dict[key][subkey], "{}:{}".format(key, subkey))
            if isplot:
                for ii in range(len(info["mask_list"])):
                    print("{} th mask:".format(ii))
                    plot_matrices(info["mask_list"][ii].squeeze()[::2], images_per_row=15, scale_limit=(0,1))
                print("max mask:")
                plot_matrices(np.stack(info["mask_list"]).max(0).squeeze()[::2], images_per_row=15, scale_limit=(0,1))
            if is_input:
                user_input = input("continue [y/n]:")
                if user_input in ["y", "Y"]:
                    pass
                else:
                    raise
        record_data(Dict["results"], np.array(data[2]), "ground_truth")
        for alpha in alpha_list:
            energy_total_list_m = []
            energy_total_list_s = []
            for c_type in c_types:
                energy_total_list_m.append(energy_dict[c_type]["energy_total^m:{:.2f}".format(alpha)])
                energy_total_list_s.append(energy_dict[c_type]["energy_total^s:{:.2f}".format(alpha)])
            energy_total_list_m = np.stack(energy_total_list_m, -1)  # [B, n_c_types]
            energy_total_list_s = np.stack(energy_total_list_s, -1)  # [B, n_c_types]
            pred_argmin_m = np.argmin(energy_total_list_m, -1)  # [B]
            pred_argmin_s = np.argmin(energy_total_list_s, -1)  # [B]
            pred_m = np.array([c_types[k] for k in pred_argmin_m])
            pred_s = np.array([c_types[k] for k in pred_argmin_s])
            record_data(Dict["results"], pred_m, "pred^m:{:.2f}".format(alpha))
            record_data(Dict["results"], pred_s, "pred^s:{:.2f}".format(alpha))
            record_data(Dict["results"], Dict["results"]["pred^m:{:.2f}".format(alpha)][-1] == Dict["results"]["ground_truth"][-1], "acc^m:{:.2f}".format(alpha))
            record_data(Dict["results"], Dict["results"]["pred^s:{:.2f}".format(alpha)][-1] == Dict["results"]["ground_truth"][-1], "acc^s:{:.2f}".format(alpha))
        if i % args.inspect_interval == 0 or i == len(dataloader) - 1:
            p.print("gt: {}".format(data[2][0]))
            for alpha in alpha_list:
                print("{}: \tpred^m: {} \tacc^m: {:.4f}\t acc_mean^m: {:.4f}".format(
                    i,
                    Dict["results"]["pred^m:{:.2f}".format(alpha)][-1][0],
                    Dict["results"]["acc^m:{:.2f}".format(alpha)][-1][0],
                    np.concatenate(Dict["results"]["acc^m:{:.2f}".format(alpha)]).mean(),
                ))
            print()
            for alpha in alpha_list:
                print("{}: \tpred^s: {} \tacc^s: {:.4f}\t acc_mean^s: {:.4f}".format(
                    i,
                    Dict["results"]["pred^s:{:.2f}".format(alpha)][-1][0],
                    Dict["results"]["acc^s:{:.2f}".format(alpha)][-1][0],
                    np.concatenate(Dict["results"]["acc^s:{:.2f}".format(alpha)]).mean(),
                ))
            try_call(pdump, args=[Dict, dirname + filename], max_exp_time=300)
    Dict["results"] = transform_dict(Dict["results"], "concatenate")
    pdump(Dict, dirname + filename)


# ### 3.4 Parse2D+classify3D:

# In[ ]:


if args.evaluation_type.startswith("yc"):
    set_seed(args.seed)
    alpha_list = np.linspace(0, 1, 21).round(2)
    isplot = is_jupyter
    data_record = {"args": args.__dict__}
    data_record["results"] = {}
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=Batch(is_collate_tuple=True).collate(), shuffle=False)

    if "parse" in tasks:
        for i, data in enumerate(dataloader):
            inputs = data[0][0]
            concept_labels = [ele[0] for ele in data[0][2]]
            example_labels = [ele[0] for ele in data[1][2]]
            example_idx = np.array([concept_labels.index(example_label) for example_label in example_labels])
            record_data(data_record, [concept_labels, example_labels, example_idx], ["concept_labels", "example_labels", "example_idx"])
            if i % args.inspect_interval == 0 or i == len(dataloader) - 1:
                p.print("Task {}:".format(i), banner_size=100)
                print("concept_labels: {}".format(concept_labels))
                if "^parse" not in args.evaluation_type:
                    print("example_labels: {}".format(example_labels))
                    print("example_idx: {}\n".format(example_idx))

            distance_edit_dict_all = {}
            is_isomorphic_dict_all = {}
            for j, input in enumerate(inputs):
                input = input.to(device)
                if (i % args.inspect_interval == 0 or i == len(dataloader) - 1) and args.verbose >= 1:
                    print("Task {}, parsing concept {}, type {}:".format(i, j, data[0][2][j][0]))
                graph_trimmed_dict, graph_dict, masks_top, info = parse_selector_from_image(
                    input=input,
                    args=args,
                    keys_dict=keys_dict,
                    isplot=isplot,
                    infer_order=args.infer_order,
                    topk=args.topk,
                    init=args.init,
                )
                graph_gt = get_concept_graph(concept_labels[j], is_new_vertical=args.is_new_vertical)
                distance_edit_dict = {k: get_graph_edit_distance(graph_trimmed_dict[k], graph_gt, to_undirected=True) for k in range(args.topk)}
                is_isomorphic_dict = {k: distance_edit_dict[k] == 0 for k in range(args.topk)}
                distance_edit_dict_all[concept_labels[j]] = distance_edit_dict
                is_isomorphic_dict_all[concept_labels[j]] = is_isomorphic_dict
                masks_top_sum = np.stack(to_np_array(*masks_top, keep_list=True)).sum(0)
                gt_mask = to_np_array(input[:,:1] != 1)
                # if is_isomorphic_dict_all[concept_labels[j]][0] != 1:
                #     print("error:")
                #     pp.pprint(graph_gt)
                #     pp.pprint(graph_trimmed_dict[0])
                #     isplot = True
                #     print()
                # else:
                #     isplot = False

                pixels_under = np.clip(gt_mask - masks_top_sum, a_min=0, a_max=None).sum((-3,-2,-1))
                pixels_exceed_out = np.clip((masks_top_sum - gt_mask) * (1-gt_mask), a_min=0, a_max=None).sum((-3,-2,-1))
                pixels_exceed_out_mean = np.clip((masks_top_sum - gt_mask).reshape(masks_top_sum.shape[-1], -1)[:, (1-gt_mask).flatten().astype(bool)], a_min=0, a_max=None).mean((-1))
                pixels_exceed_in = np.clip((masks_top_sum - gt_mask) * gt_mask, a_min=0, a_max=None).sum((-3,-2,-1))
                record_data(data_record,
                            [graph_gt, np.stack(to_np_array(*masks_top, keep_list=True)), masks_top_sum, gt_mask, pixels_under, pixels_exceed_out, pixels_exceed_out_mean, pixels_exceed_in, info],
                            [f"graph_gt_{concept_labels[j]}", f"masks_top_{concept_labels[j]}", f"masks_top_sum_{concept_labels[j]}", f"gt_mask_{concept_labels[j]}", f"pixels_under_{concept_labels[j]}", f"pixels_exceed_out_{concept_labels[j]}", f"pixels_exceed_out_mean_{concept_labels[j]}", f"pixels_exceed_in_{concept_labels[j]}", f"info_{concept_labels[j]}"])
                record_data(data_record,
                            [graph_gt, np.stack(to_np_array(*masks_top, keep_list=True)), masks_top_sum, gt_mask, pixels_under, pixels_exceed_out, pixels_exceed_out_mean, pixels_exceed_in, info],
                            [f"graph_gt_ex_{j}", f"masks_top_ex_{j}", f"masks_top_sum_ex_{j}", f"gt_mask_ex_{j}", f"pixels_under_ex_{j}", f"pixels_exceed_out_ex_{j}", f"pixels_exceed_out_mean_ex_{j}", f"pixels_exceed_in_ex_{j}", f"info_ex_{j}"])
        
                energy_list = []
                concept_energy_list = []
                relation_energy_list = []
                for k in range(args.topk):
                    energy_list.append(np.sum([ele[2] for ele in graph_trimmed_dict[k]]))
                    concept_energy_list.append(np.sum([ele[2] for ele in graph_trimmed_dict[k] if not isinstance(ele[0], tuple)]))
                    relation_energy_list.append(np.sum([ele[2] for ele in graph_trimmed_dict[k] if isinstance(ele[0], tuple)]))
                    record_data(data_record, [graph_trimmed_dict[k], graph_dict[k], distance_edit_dict[k], is_isomorphic_dict[k], energy_list[-1], concept_energy_list[-1], relation_energy_list[-1]],
                                [f"graph_trimmed_{concept_labels[j]}_{k}", f"graph_{concept_labels[j]}_{k}", f"distance_edit_{concept_labels[j]}_{k}", f"is_isomorphic_{concept_labels[j]}_{k}", f"energy_{concept_labels[j]}_{k}", f"concept_energy_{concept_labels[j]}_{k}", f"relation_energy_{concept_labels[j]}_{k}"])
                    record_data(data_record, [graph_trimmed_dict[k], graph_dict[k], distance_edit_dict[k], is_isomorphic_dict[k], energy_list[-1], concept_energy_list[-1], relation_energy_list[-1]],
                                [f"graph_trimmed_ex_{j}_{k}", f"graph_ex_{j}_{k}", f"distance_edit_ex_{j}_{k}", f"is_isomorphic_ex_{j}_{k}", f"energy_ex_{j}_{k}", f"concept_energy_ex_{j}_{k}", f"relation_energy_ex_{j}_{k}"])
                energy_list = np.array(energy_list)
                concept_energy_list = np.array(concept_energy_list)
                relation_energy_list = np.array(relation_energy_list)
                record_data(data_record, [energy_list, concept_energy_list, relation_energy_list],
                            [f"energy_{concept_labels[j]}", f"concept_energy_{concept_labels[j]}", f"relation_energy_{concept_labels[j]}"])
                record_data(data_record, [energy_list, concept_energy_list, relation_energy_list],
                            [f"energy_ex_{j}", f"concept_energy_ex_{j}", f"relation_energy_ex_{j}"])
                record_data(data_record, [np.mean(list(distance_edit_dict.values())), np.mean(list(is_isomorphic_dict.values()))],
                            [f"distance_edit_{concept_labels[j]}", f"is_isomorphic_{concept_labels[j]}"])
                record_data(data_record, [np.mean(list(distance_edit_dict.values())), np.mean(list(is_isomorphic_dict.values()))],
                            [f"distance_edit_ex_{j}", f"is_isomorphic_ex_{j}"])
                pairwise_distance_topk = np.array([[get_graph_edit_distance(graph_trimmed_dict[ll], graph_trimmed_dict[mm], to_undirected=True) for mm in range(args.topk)] for ll in range(args.topk)])
                record_data(data_record, [pairwise_distance_topk], [f"pairwise_distance_topk_{concept_labels[j]}"])
                record_data(data_record, [pairwise_distance_topk], [f"pairwise_distance_topk_ex_{j}"])
                if (i % args.inspect_interval == 0 or i == len(dataloader) - 1) and args.verbose >= 1:
                    print("distance_edit:")
                    print(np.array(list(distance_edit_dict.values())))
                    print("energy:")
                    print(energy_list)
                    print("concept_energy:")
                    print(concept_energy_list)
                    print("relation_energy:")
                    print(relation_energy_list)
                    print("pixels_under:")
                    print(pixels_under)
                    print("pixels_exceed_in:")
                    print(pixels_exceed_in)
                    print("pixels_exceed_out:")
                    print(pixels_exceed_out)
                    print("pixels_exceed_out_mean:")
                    print(pixels_exceed_out_mean)
                    print(f"pairwise_distance_topk_{concept_labels[j]}:")
                    print(pairwise_distance_topk.mean(-1))
                    print()
                del input
                del masks_top
                gc.collect()
                if (i % args.inspect_interval == 0 or i == len(dataloader) - 1) and args.verbose >= 2:
                    print("{}th concept, gt concept: {}".format(j, concept_labels[j]))
                    print("graph_gt:")
                    pp.pprint(graph_gt)
                    print("graph_trimmed:")
                    pp.pprint(graph_trimmed_dict)
                    print("Edit distance: {}   is_isomorphic: {}".format(distance_edit_dict, is_isomorphic_dict))
                    print()
                    if isplot:
                        for i in range(1):
                            print(f"top {i} SGLD:")
                            for ll in range(len(info["mask_list"])):
                                print(f"  mask {ll}:")
                                plot_matrices(info["mask_list"][ll][::2,i].squeeze((1,2)), images_per_row=19, figsize=(20,4))
                            print("Max mask:")
                            plot_matrices(np.stack(info["mask_list"]).max(0)[::2,i].squeeze((1,2)), images_per_row=19, figsize=(20,4))
            for k in range(args.topk):
                record_data(data_record, [np.mean([distance_edit_dict_all[c_type][k] for c_type in concept_labels]), np.mean([is_isomorphic_dict_all[c_type][k] for c_type in concept_labels])],
                            [f"distance_edit_{k}", f"is_isomorphic_{k}"])
            record_data(data_record, [np.mean([distance_edit_dict_all[c_type][k] for k in range(args.topk) for c_type in concept_labels]), np.mean([is_isomorphic_dict_all[c_type][k] for k in range(args.topk) for c_type in concept_labels])],
                            [f"distance_edit", f"is_isomorphic"])
            for c_type in sorted(concept_labels):
                for k in range(args.topk):
                    print("{}_{}:     \tedit_distance_mean: {:.4f}    acc_iso_mean: {:.4f}".format(
                        c_type, k, np.mean(data_record[f"distance_edit_{c_type}_{k}"]), np.mean(data_record[f"is_isomorphic_{c_type}_{k}"])))
                print("{}_m:     \tedit_distance_mean: {:.4f}    acc_iso_mean: {:.4f}".format(
                    c_type, np.mean([np.mean(data_record[f"distance_edit_{c_type}_{k}"]) for k in range(args.topk)]),
                    np.mean([np.mean(data_record[f"is_isomorphic_{c_type}_{k}"]) for k in range(args.topk)])))
            print()
            for c_type in sorted(concept_labels):
                print("{}_m:     \tedit_distance_mean: {:.4f}    acc_iso_mean: {:.4f}".format(
                    c_type, np.mean(data_record[f"distance_edit_{c_type}"]), np.mean(data_record[f"is_isomorphic_{c_type}"])))
            print()
            for k in range(args.topk):
                print("k_{}:    \tedit_distance_mean: {:.4f}    acc_iso_mean: {:.4f}".format(
                    k, np.mean(data_record[f"distance_edit_{k}"]), np.mean(data_record[f"is_isomorphic_{k}"])))
            print()
            print("all:     \tedit_distance_mean: {:.4f}    acc_iso_mean: {:.4f}".format(
                    np.mean([np.mean(data_record[f"distance_edit_{c_type}_{k}"]) for k in range(args.topk) for c_type in concept_labels]),
                    np.mean([np.mean(data_record[f"is_isomorphic_{c_type}_{k}"]) for k in range(args.topk) for c_type in concept_labels])))
            print()
            # Printing:
            if i % args.inspect_interval == 0 or i == len(dataloader) - 1:
                make_dir(dirname + filename)
                try_call(pdump, args=[data_record, dirname + filename])
                p.print("Task {} finished and saved.\n".format(i))
                print()
                sys.stdout.flush()
            del data
            gc.collect()

    if "classify" in tasks:
        # Obtain the graph for each demonstration for each example:
        graphs_gt_all = []
        concept_labels_all = []
        example_labels_all = []
        example_idx_all = []
        concept_idx_all = []
        for i, data in enumerate(dataloader):
            concept_labels = [ele[0] for ele in data[0][2]]
            example_labels = [ele[0] for ele in data[1][2]]
            example_idx = np.array([concept_labels.index(example_label) for example_label in example_labels])
            concept_idx = np.array([example_labels.index(concept_label) for concept_label in concept_labels])

            graphs_gt_all.append([get_concept_graph(concept_label, is_new_vertical=args.is_new_vertical, is_concept=args.is_concept, is_relation=args.is_relation) for concept_label in concept_labels])
            concept_labels_all.append(concept_labels)
            example_labels_all.append(example_labels)
            example_idx_all.append(example_idx)
            concept_idx_all.append(concept_idx)

        # Classify:
        Dict = {"args": args.__dict__, "results": {}}
        alpha_list = np.linspace(0,1,21).round(3)
        make_dir(dirname + filename)
        n_concepts_per_task = len(graphs_gt_all[i])
        n_examples_per_task = len(example_labels)

        if args.load_parse_src != "gt":
            parse_result = pload(EXP_PATH + "/" + args.load_parse_src)
            print("Loading parsing result from {}.".format(EXP_PATH + "/" + args.load_parse_src))

        args_3D = deepcopy(args)
        args_3D.canvas_size = args.canvas_size_3D
        args_3D.image_size = (args.canvas_size_3D, args.canvas_size_3D)
        for i, data in enumerate(dataloader):
            selectors = {}
            for j in range(n_concepts_per_task):
                if args.load_parse_src == "gt":
                    graph_j = graphs_gt_all[i][j]
                else:
                    # Load from file:
                    graph_j = parse_result[f'graph_trimmed_ex_{j}_0'][i]
                if args.is_bidirectional_re:
                    graph_j = get_bidirectional_graph(graph_j)
                selectors[j] = get_selector_from_graph(graph_j, ebm_dict_3D, CONCEPTS, OPERATORS)
            if i % args.inspect_interval == 0:
                p.print("iter: {}".format(i))
            energy_dict = {}
            input = torch.cat(data[1][0]).to(device)
            input_rescaled = F.interpolate(input, size=args.canvas_size_3D, mode="nearest")
            mask = torch.cat(data[1][1])
            mask_rescaled = F.interpolate(mask, size=args.canvas_size_3D, mode="nearest")
            for j in range(n_concepts_per_task):
                selector = selectors[j]
                for ll in range(n_examples_per_task):
                    if isplot:
                        p.print(f"i={i}, j={j}, ll={ll}:")
                    _, masks_top, info = get_selector_SGLD(
                        selector, input_rescaled[ll:ll+1], args_3D,
                        ebm_target="mask",
                        init=args.init,
                        SGLD_object_exceed_coef=0,
                        SGLD_mutual_exclusive_coef=args.SGLD_mutual_exclusive_coef,
                        SGLD_pixel_entropy_coef=args.SGLD_pixel_entropy_coef,
                        ensemble_size=args.ensemble_size,
                        sample_step=args.sample_step,
                        isplot=isplot,
                    )  # mask_top: each element [topk, n_examples, 1, H, W]
                    energy_dict[(j,ll)] = {}
                    energy_dict[(j,ll)]["masks_top"] = np.stack(to_np_array(*masks_top, keep_list=True), 2)[0]  # [n_ex, n_masks, 1, H, W]
                    energy_dict[(j,ll)]["masks_max"] =  energy_dict[(j,ll)]["masks_top"].max(1)  # [n_ex, 1, H, W]
                    energy_dict[(j,ll)]["energy_mask"] = np.stack(list(info['energy_mask'].values()), 2)[0]  # [B, n_masks_concept_re]
                    energy_dict[(j,ll)]["energy"] = info['energy'][0]
                    energy_dict[(j,ll)]["energy_mean"] = np.mean(energy_dict[(j,ll)]["energy_mask"], -1)  # [B]
                    energy_dict[(j,ll)]["energy_sum"] = np.sum(energy_dict[(j,ll)]["energy_mask"], -1)  # [B]
                    energy_dict[(j,ll)]["mutual_exclusive"] = info['mutual_exclusive_list'][0]
                    for alpha in alpha_list:
                        energy_dict[(j,ll)]["energy_total^m:{:.2f}".format(alpha)] = energy_dict[(j,ll)]["energy_mean"] + energy_dict[(j,ll)]["mutual_exclusive"] * alpha
                        energy_dict[(j,ll)]["energy_total^s:{:.2f}".format(alpha)] = energy_dict[(j,ll)]["energy_sum"] + energy_dict[(j,ll)]["mutual_exclusive"] * alpha
            key_list = ["masks_top", "masks_max", "energy", "energy_mask", "energy_mean", "energy_sum", "mutual_exclusive"]
            for alpha in alpha_list:
                key_list.append("energy_total^m:{:.2f}".format(alpha))
                key_list.append("energy_total^s:{:.2f}".format(alpha))
            energy_dict_all = {}
            for key in key_list:
                try:
                    energy_dict_all[key] = np.array([np.array([energy_dict[(j,ll)][key] for ll in range(n_examples_per_task)]) for j in range(n_concepts_per_task)])  # [n_concepts, n_ex]
                except:
                    energy_dict_all[key] = [[energy_dict[(j,ll)][key] for ll in range(n_examples_per_task)] for j in range(n_concepts_per_task)]
            concept_idx_pred = energy_dict_all["energy"].squeeze().argmin(1)
            concept_idx_gt = concept_idx_all[i]
            acc = (concept_idx_pred == concept_idx_gt).mean()
            masks_max = energy_dict_all["masks_max"]  # "masks_max", : [n_concepts, n_ex, 1, 1, H, W]
            mask_chosen = np.concatenate([masks_max[j, concept_idx_pred[j]] for j in range(n_concepts_per_task)])
            mask_gt = np.stack([mask_rescaled[concept_idx_pred[j]] for j in range(n_concepts_per_task)])
            iou_ex = get_soft_IoU(mask_chosen.round(), mask_gt.round(), dim=(-3,-2,-1))
            iou = iou_ex.mean()

            # mask_rescaled: [n_ex, 1, H, W]
            p.print("i={}, acc={:.3f}, iou={:.4f}  pred:{}, gt: {}".format(i, acc, iou, concept_idx_pred, concept_idx_gt))
            record_data(Dict["results"], [acc, concept_idx_pred, concept_idx_gt, iou, iou_ex, mask_rescaled], ["acc", "concept_idx_pred", "concept_idx_gt", "iou", "iou_ex", "mask_rescaled"])
            record_data(Dict["results"], list(energy_dict_all.values()), list(energy_dict_all.keys()))
            if i % args.inspect_interval == 0 or i == len(dataloader) - 1:
                p.print("{}: acc={}, iou={}".format(i, np.mean(Dict["results"]["acc"]), np.mean(Dict["results"]["iou"])))
                try_call(pdump, args=[Dict, dirname + filename], max_exp_time=300)
                print()
        try_call(pdump, args=[Dict, dirname + filename])


# In[ ]:


# Random graph:
if args.evaluation_type.startswith("yc") and args.model_type == "rand-graph":
    # Get frequency:
    freq_dict = {}
    for data in dataset:
        c_types = data[0][2]
        for c_type in c_types:
            graph_gt = get_concept_graph(c_type, is_new_vertical=args.is_new_vertical, is_bidirectional_re=False)
            n_line = np.sum([1 for ele in graph_gt if ele[1] == "Line"])
            n_vertical_mid = np.sum([1 for ele in graph_gt if ele[1] == 'VerticalMid'])
            n_vertical_edge = np.sum([1 for ele in graph_gt if ele[1] == 'VerticalEdge'])
            n_parallel = np.sum([1 for ele in graph_gt if ele[-1] == 'Parallel'])
            record_data(freq_dict, [n_line, n_vertical_mid, n_vertical_edge, n_parallel], ["Line", "VerticalMid", "VerticalEdge", "Parallel"])
    freq_mean = transform_dict(freq_dict, "mean")
    freq_vertical_edge = freq_mean["VerticalEdge"]
    freq_vertical_mid = freq_mean["VerticalMid"]
    freq_parallel = freq_mean['Parallel']
    freq_ver_para = np.array([freq_vertical_mid, freq_vertical_edge, freq_parallel])
    freq_ver_para = freq_ver_para / freq_ver_para.sum()

    # Compute metric:
    distance_edit_all = []
    is_isomorphic_all = []
    for data in dataset:
        c_types = data[0][2]
        for c_type in c_types:
            graph_gt = get_concept_graph(c_type, is_new_vertical=args.is_new_vertical, is_bidirectional_re=False)
            graph_rand = get_rand_graph(freq_dict, is_new_vertical=args.is_new_vertical)
            distance_edit = get_graph_edit_distance(graph_rand, graph_gt, to_undirected=True)
            is_isomorphic = (distance_edit == 0).astype(int)
            distance_edit_all.append(distance_edit)
            is_isomorphic_all.append(is_isomorphic)
    print("Edit_distance: {:.4f}".format(np.mean(distance_edit_all)))
    print("acc_isomorphic: {:.4f}".format(np.mean(is_isomorphic_all)))


# # 4. Analysis:

# ## 4.1 For zero-shot:

# In[ ]:


isplot = False
verbose = 1
args.is_analysis = is_jupyter


# ### 4.1.1 Parse:

# In[ ]:


args.evaluation_type = "parse"
verbose = True
parse_filter_mode = "mask"
isplot = False
show_only_error = True
inspect_interval = 500
is_baseline = False
if args.is_analysis and args.evaluation_type.startswith("parse"):
    # # Mask-RCNN, New Vertical, 400 examples:
    # graph_list = pload("/dfs/user/tailin/.results/BabyARC_baselines/relation_models/mask_rcnn/eval/maskrcnn-conv_d_10-05_e_obj16full-cnn-v3-lr2.5e-5-fix-turing4_m_turing4_Hash_9IeiNpkP.p")
    # is_baseline = False

    # # Rand-graph:
    # filename = "evaluation_parse_canvas_16_color_1,2_model_rand-graph_mutu_500_ens_64_sas_150_newv_True_seed_2_id_0_Hash_+hRuVNuO_turing4.p"
    # dirname = "/dfs/user/tailin/.results/evaluation_parse_12-10/"
    
    # Best for new_vertical, Dec 8:
    filename = "evaluation_parse_canvas_16_color_1,2_model_hc-ebm_mutu_500.0_ens_64_sas_150_newv_True_seed_2_id_None_Hash_kEIST2aZ_turing4.p"
    dirname = "results/evaluation_parse_1-21/"  # new 400 examples

    filenames = filter_filename(dirname, include=["canvas_16"], exclude=["Vht+VpvC"])
    # filenames = [filename]
    analysis_all = {}
    df_dict_list = []
    threshold_pixels = 0
    thresholds = [None, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0]
    print("Processing {} files.".format(len(filenames)))
    for filename in filenames:
        print(filename)
        data_record = pload(dirname + filename)
        for key in ["is_new_vertical", "concept_model_hash", "SGLD_mutual_exclusive_coef", "SGLD_pixel_entropy_coef", "SGLD_is_anneal", "sample_step", "ensemble_size", "SGLD_is_penalize_lower", "is_round_mask"]:
            print("{}: {}".format(key, update_default_hyperparam(data_record["args"])[key]))
        analysis_dict = {}

        df_dict = {}
        df_dict.update(data_record["args"])
        df_dict["hash"] = filename.split("_")[-2]
        for i in range(len(data_record['parsing_all_dict']["graph"])):
            if is_baseline:
                data = dataset[i]
                c_type_gt = data[3]["obj_spec"][0][0][1].split("_")[0]
                graph_gt = get_concept_graph(c_type_gt, is_new_vertical=args.is_new_vertical)
                graph_pred_ori = graph_list[i]
            else:
                c_type_gt = data_record['parsing_all_dict']["info"][i]["obj_spec"][0][0][1].split("_")[0]
                graph_gt = get_concept_graph(c_type_gt, is_new_vertical=data_record["args"]["is_new_vertical"])
                graph_pred_ori = shorten_graph(data_record['parsing_all_dict']["graph"][i])
            mask_top = data_record['parsing_all_dict']["masks_top"][i]
            if args.model_type == "hc-ebm":
                if parse_filter_mode == "threshold":
                    for threshold in thresholds:
                        graph_pred, removed_items = filter_graph_with_threshold(graph_pred_ori, threshold=threshold)
                        distance_edit = get_graph_edit_distance(graph_pred, graph_gt, to_undirected=True)
                        is_isomorphic = distance_edit == 0
                        record_data(analysis_dict, [distance_edit, is_isomorphic], ["distance_edit:{}".format(threshold), "is_isomorphic:{}".format(threshold)])
                elif parse_filter_mode == "mask":
                    masks_top = data_record['parsing_all_dict']["masks_top"][i]
                    masks_is_invalid = [is_mask_invalid(mask, threshold_pixels=threshold_pixels) for mask in masks_top]
                    graph_pred = filter_graph_with_masks(graph_pred_ori, masks_is_invalid=masks_is_invalid)
                    distance_edit = get_graph_edit_distance(graph_pred, graph_gt, to_undirected=True)
                    is_isomorphic = distance_edit == 0
                    record_data(analysis_dict, [distance_edit, is_isomorphic], ["distance_edit", "is_isomorphic"])
                else:
                    raise
            elif args.model_type == "rand-graph":
                distance_edit = get_graph_edit_distance(graph_pred_ori, graph_gt, to_undirected=True)
                is_isomorphic = distance_edit == 0
                record_data(analysis_dict, [distance_edit, is_isomorphic], ["distance_edit", "is_isomorphic"])
            if i % inspect_interval == 0 and i > 0 or i == len(data_record['parsing_all_dict']["graph"]) - 1:
                print("Data {}:".format(i))
                if isplot and (show_only_error is False or is_isomorphic != 1):
                    visualize_matrices(data_record['parsing_all_dict']["input"][i].argmax(1))
                    plot_matrices(torch.stack(data_record['parsing_all_dict']["masks_top"][i]).squeeze(), images_per_row=6, scale_limit=(0,1))
                    print("prediction:")
                    pp.pprint(graph_pred)
                    print("\nground-truth:")
                    pp.pprint(graph_gt)
                if args.model_type == "hc-ebm":
                    if parse_filter_mode == "threshold":
                        for threshold in thresholds:
                            print("Threshold: {}    \tEdit distance: {}  \tmean: {:.3f} \t is_iso: {:.4f}".format(
                                threshold, analysis_dict["distance_edit:{}".format(threshold)][-1],
                                np.mean(analysis_dict["distance_edit:{}".format(threshold)]),
                                np.mean(analysis_dict["is_isomorphic:{}".format(threshold)])))
                    elif parse_filter_mode == "mask":
                        print("Edit distance: {}  \tmean: {:.3f} \t is_iso: {:.4f}".format(
                            analysis_dict["distance_edit"][-1],
                            np.mean(analysis_dict["distance_edit"]),
                            np.mean(analysis_dict["is_isomorphic"])))
                elif args.model_type == "rand-graph":
                    print("Edit distance: {}  \tmean: {:.3f} \t is_iso: {:.4f}".format(
                        analysis_dict["distance_edit"][-1],
                        np.mean(analysis_dict["distance_edit"]),
                        np.mean(analysis_dict["is_isomorphic"])))
                print("\n")
        record_data(df_dict, [np.mean(analysis_dict["distance_edit"]), np.mean(analysis_dict["is_isomorphic"])], ["distance_edit", "is_isomorphic"], nolist=True)
        df_dict_list.append(df_dict)
        analysis_all[filename] = analysis_dict
        print("\n")
    analysis_all["df"] = pd.DataFrame(df_dict_list)
    df_group = groupby_add_keys(analysis_all["df"], by=["is_new_vertical", "concept_model_hash", "SGLD_mutual_exclusive_coef", "SGLD_pixel_entropy_coef", "SGLD_is_anneal", "SGLD_is_penalize_lower", "sample_step", "ensemble_size", "is_round_mask"],
        add_keys="hash",
        other_keys=["is_isomorphic", "distance_edit"],
        mode={"max": ["is_isomorphic"], "min":["distance_edit"], "count": None}
    )
    display(df_group.style.background_gradient(cmap=plt.cm.get_cmap("PiYG")))
    pdump(analysis_all, dirname + "analysis_all.p")


# ### 4.1.2 Grounding:

# In[ ]:


args.evaluation_type = "grounding"
total_tasks = 200
if args.is_analysis and args.evaluation_type.startswith("grounding"):
    df_dict_list = []
    for c_type in ["Eshape", "Fshape", "Ashape", "RectE1a", "RectE2a", "RectE3a"]:
        dirname = EXP_PATH + "/evaluation_grounding-{}_1-21/".format(c_type)
        try:
            filenames = filter_filename(dirname, include=".p")
        except:
            continue
        for filename in filenames:
            try:
                data_record = pload(dirname + filename)
            except Exception as e:
                print(f"Error {e} happens for file {filename}")
            iou = np.mean(data_record["iou"][:total_tasks])
            df_dict = update_default_hyperparam_generalization(data_record["args"])
            df_dict["hash"] = filename.split("_")[-2]
            df_dict["machine"] = filename.split("_")[-1][:-2]
            df_dict["iou"] = iou
            df_dict["epoch"] = len(data_record["iou"][:total_tasks])
            df_dict["load_epoch"] = data_record["args"]["relation_load_id"] * 5
            df_dict_list.append(df_dict)

    df = pd.DataFrame(df_dict_list)
    df_group = groupby_add_keys(
        df,
        by=["evaluation_type",
            # 'SGLD_mutual_exclusive_coef',
            # 'SGLD_pixel_entropy_coef',
            # 'ensemble_size',
            # 'sample_step',
            # "SGLD_is_penalize_lower",
            "concept_model_hash",
            "relation_model_hash",
            # "is_bidirectional_re",
            # "max_n_distractors",
            # "min_n_distractors",
            "allow_connect",
            "is_proper_size",
            "is_concept",
            "is_relation",
            # "id",
            "gpuid",
           ],
        add_keys=["hash", "machine"], other_keys=["iou", "epoch", "load_epoch"],
        mode={
            "max": ["iou", "epoch", "load_epoch"],
            "count": None,
        })
    display(df_group.style.background_gradient(cmap=plt.cm.get_cmap("PiYG")))


# In[ ]:


if args.is_analysis and args.evaluation_type.startswith("grounding"):
    dirname = EXP_PATH + "/evaluation_grounding-{}_1-21/".format("Eshape")
    hash_str = "1AMKX9yn"
    filename = filter_filename(dirname, include=hash_str)[0]
    data_record = pload(dirname + "/" + filename)
    for i, iou in enumerate(data_record["iou"]):
        visualize_matrices(data_record["input"][i].argmax(1), subtitles=["{}, iou: {}".format(i, iou)], images_per_row=6)
        plot_matrices(np.concatenate([data_record["masks_top"][i].sum(1).squeeze(0),
                    data_record["masks_top"][i].squeeze((0,2))]), images_per_row=6)


# ### 4.1.3 Classify:

# In[ ]:


args.evaluation_type = "classify"
verbose = 0
total_tasks = 200
if args.is_analysis and args.evaluation_type.startswith("classify"):
    dirname = EXP_PATH + "/evaluation_classify_1-21/"
    filenames = filter_filename(dirname, include=".p", exclude="df_")
    df_dict_list = []
    for filename in filenames:
        df_dict = {}
        if verbose > 0:
            p.print("{}:".format(filename), banner_size=100)
        data_record = pload(dirname + filename)
        if "acc^s:0.00" not in data_record["results"]:
            if verbose > 0:
                print("{} not fully processed.".format(filename))
            continue
        df_dict.update(update_default_hyperparam_generalization(data_record["args"]))
        df_dict["load_epoch"] = data_record["args"]["relation_load_id"] * 5
        df_dict["filename"] = filename
        df_dict["hash"] = filename.split("_")[-2]
        df_dict["machine"] = filename.split("_")[-1][:-2]
        if verbose > 0:
            for key in ["SGLD_mutual_exclusive_coef", "SGLD_is_penalize_lower", "concept_model_hash", "relation_model_hash"]:
                print("{}: {}".format(key, data_record["args"][key]))
        acc_s_list = []
        alpha_s_list = []
        df_dict["n_examples"] = len(data_record["results"]["acc^s:0.00"][:total_tasks])
        for alpha in np.linspace(0,1,11):
            if verbose > 0:
                print("acc^s:{:.2f}: {:.3f}".format(alpha, np.mean(data_record["results"]["acc^s:{:.2f}".format(alpha)][:total_tasks])))
            acc_s_list.append(np.mean(data_record["results"]["acc^s:{:.2f}".format(alpha)][:total_tasks]))
            alpha_s_list.append(alpha)
        df_dict["acc_s_max"] = np.max(acc_s_list)
        df_dict["acc_s_argmax"] = alpha_s_list[np.argmax(acc_s_list)]
        df_dict["acc_s_0"] = acc_s_list[0]
        df_dict["acc_s_0.1"] = acc_s_list[1]
        if verbose > 0:
            print()
        acc_m_list = []
        alpha_m_list = []
        for alpha in np.linspace(0,1,11):
            if verbose > 0:
                print("acc^m:{:.2f}: {:.3f}".format(alpha, np.mean(data_record["results"]["acc^m:{:.2f}".format(alpha)][:total_tasks])))
            acc_m_list.append(np.mean(data_record["results"]["acc^m:{:.2f}".format(alpha)][:total_tasks]))
            alpha_m_list.append(alpha)
        df_dict["acc_m_max"] = np.max(acc_m_list)
        df_dict["acc_m_argmax"] = alpha_m_list[np.argmax(acc_m_list)]
        df_dict["acc_m_0"] = acc_m_list[0]
        df_dict["acc_m_0.1"] = acc_m_list[1]
        df_dict_list.append(df_dict)
    df = pd.DataFrame(df_dict_list)
    df_group = groupby_add_keys(
        df,
        by=["dataset",
            # "SGLD_mutual_exclusive_coef",
            # "SGLD_is_penalize_lower",
            "concept_model_hash",
            "relation_model_hash",
            # "ensemble_size",
            # "is_bidirectional_re",
            # "max_n_distractors",
            # "id",
            "gpuid",
           ],
        add_keys=["hash", "machine"],
        other_keys=[ "acc_s_0", "acc_s_0.1", "acc_s_argmax", "acc_m_max", "acc_m_argmax", "n_examples", "load_epoch"],
        mode={
            "mean": [ "acc_s_0", "acc_s_0.1", "acc_s_argmax", "acc_m_argmax", "n_examples", "load_epoch"],
            "max": ["acc_s_max", "acc_m_max"],
            "count": None,
        }
    )
    pdump({"df": df, "df_group": df_group}, dirname + "df_{}-{}.p".format(datetime.now().month, datetime.now().day))
    display(df_group.style.background_gradient(cmap=plt.cm.get_cmap("PiYG")))


# ### 4.2.1 Parse2D+classify3D:

# #### 4.2.1.1 Parse:

# In[ ]:


## 1-1:
args.evaluation_type = "yc-parse+classify^parse"
verbose = 1
energy_criterion = {
    "energy": 2,
    "pixels_under": 2,
    "pixels_exceed_in": 2,
    "pixels_exceed_out_mean": 2,
    "pairwise_distance_topk_mean": 0.1,
}
isplot = False
def get_instance_energy(df_dict, energy_criterion):
    energy = 0
    for key, value in energy_criterion.items():
        energy += df_dict[key] * value
    return energy
dff_dict_list = []
args.is_analysis = is_jupyter
if args.is_analysis and args.evaluation_type.startswith("yc-parse+classify"):
    dirname = EXP_PATH + "/evaluation_yc-parse+classify^parse_1-21/"
    filenames = filter_filename(dirname, include=".p", exclude="df_")
    # filenames = ["evaluation_yc-parse+classify^parse_canvas_16_color_1,2_ex_400_min_0_model_hc-ebm_mutu_500.0_ens_64_sas_150_newv_True_batch_1_con_fRZtzn33_re_Wfxw19nM_bi_True_seed_2_id_None_Hash_mU7ILNWm_turing3.p"]
    # filenames = ["evaluation_yc-parse+classify^parse_canvas_16_color_1,2_ex_400_min_0_model_hc-ebm_mutu_500.0_ens_64_sas_150_newv_True_batch_1_con_4Qb0Vu0x_re_Wfxw19nM_bi_True_seed_2_id_None_Hash_84sNKLJz_hyperturing1.p"]  # Without L^em
    for filename in filenames:
        dff_dict = {}
        p.print(filename, banner_size=100, is_datetime=False)
        try:
            data_record = pload(dirname + filename)
        except Exception as e:
            print(e)
            continue
        if 'is_isomorphic_0' not in data_record:
            print("'is_isomorphic_0' not in data_record, skip.")
            continue
        args = init_args(update_default_hyperparam_generalization(update_default_hyperparam(data_record["args"])))
        c_types = get_c_core(args.dataset.split("-")[1].split("+"))
        print(f"concept_hash: {args.concept_model_hash}")
        print(f"relation_hash: {args.relation_model_hash}")
        print(f"SGLD_is_penalize_lower: {args.SGLD_is_penalize_lower}")
        print(f"init: {args.init}")
        print(f"infer_order: {args.infer_order}")
        print(f"SGLD_is_penalize_lower_seq: {args.SGLD_is_penalize_lower_seq}")
        print(f"lambd: {args.lambd}")
        print(f"is_bidirectional_re: {args.is_bidirectional_re}")
        print(f"tasks completed: {len(data_record[f'graph_trimmed_{c_types[0]}_0'])}")
        x = np.arange(args.topk)
        acc_rank = [np.mean(data_record[f"is_isomorphic_{i}"]) for i in range(args.topk)]
        distance_rank = [np.mean(data_record[f"distance_edit_{i}"]) for i in range(args.topk)]
        # plot_simple(y=acc_rank, title="acc vs. rank", xlabel="rank", ylabel="acc")
        plot_2_axis(x=x, y1=acc_rank, y2=distance_rank,
                    xlabel="rank", ylabel1="acc", ylabel2="distance",
                    ylim1=(0,0.8), ylim2=(0,2.5),
                    title="acc & distance vs. rank",
                   )

        """
        For each time step, compute the acc, and visualize:
        """
        Dict = {}
        Dict_all = {}
        best_acc_dict = {}
        argmax_dict = {}
        graph_distance_min_acc_dict = {}
        graph_distance_mode_acc_dict = {}
        energy_dict = {}
        concept_energy_dict = {}
        energy_min_acc_dict = {}
        concept_energy_min_acc_dict = {}
        df_dict_list = []
        for j, c_type in enumerate(c_types):
            Dict[c_type] = []
            Dict_all[c_type] = []
            argmax_dict[c_type] = []
            energy_dict[c_type] = []
            concept_energy_dict[c_type] = []
            if isplot:
                print(f"concept: {c_type}")
            for i in range(len(data_record['distance_edit_0'])):
                List_i = []
                energy_i = []
                concept_energy_i = []
                instance_energy_list = []
                for k in range(args.topk):
                    List_i.append(float(data_record[f"is_isomorphic_{c_type}_{k}"][i]))
                    graph_trimmed = data_record[f"graph_trimmed_{c_type}_{k}"][i]
                    energy_i.append(np.sum([ele[2] for ele in graph_trimmed]))
                    concept_energy_i.append(np.sum([ele[2] for ele in graph_trimmed if not isinstance(ele[0], tuple)]))

                    # Conctruct df_dict:
                    df_dict = {}
                    df_dict["task"] = i
                    df_dict["c_type"] = c_type
                    df_dict["energy"] = data_record[f"energy_{c_type}"][i][k]
                    df_dict["rank"] = k
                    df_dict["concept_energy"] = data_record[f"concept_energy_{c_type}"][i][k]
                    df_dict["relation_energy"] = data_record[f"relation_energy_{c_type}"][i][k]
                    df_dict["distance_edit"] = data_record[f"distance_edit_{c_type}_{k}"][i]
                    df_dict["is_isomorphic"] = data_record[f"is_isomorphic_{c_type}_{k}"][i]
                    df_dict["pixels_under"] = data_record[f"pixels_under_{c_type}"][i][k]
                    df_dict["pixels_exceed_in"] = data_record[f"pixels_exceed_in_{c_type}"][i][k]
                    df_dict["pixels_exceed_out"] = data_record[f"pixels_exceed_out_{c_type}"][i][k]
                    df_dict["pixels_exceed_out_mean"] = data_record[f"pixels_exceed_out_mean_{c_type}"][i][k]
                    df_dict["pairwise_distance_topk"] = data_record[f"pairwise_distance_topk_{c_type}"][i][k]
                    df_dict["pairwise_distance_topk_mean"] = data_record[f"pairwise_distance_topk_{c_type}"][i][k].mean(-1)
                    df_dict["graph_trimmed"] = data_record[f"graph_trimmed_{c_type}_{k}"][i]
                    df_dict["graph_gt"] = data_record[f"graph_gt_{c_type}"][i]
                    instance_energy_list.append(get_instance_energy(df_dict, energy_criterion))
                    df_dict_list.append(df_dict)

                    # Plot the top instance:
                    if isplot and k == 0 and df_dict["is_isomorphic"] != True:
                        p.print(f"Task {i}: graph_trimmed, distance_edit = {df_dict['distance_edit']}", banner_size=60, is_datetime=False)
                        p.print(df_dict["graph_trimmed"], is_datetime=False)
                        plot_matrices(data_record[f"gt_mask_{c_type}"][i].squeeze(1), images_per_row=8)
                        plot_matrices(data_record[f"masks_top_{c_type}"][i][:, k].squeeze(1), images_per_row=8)
                instance_energy_list = np.array(instance_energy_list)
                instance_energy_argsort = instance_energy_list.argsort()
                for kk, idx in enumerate(instance_energy_argsort):
                    df_dict_list[idx - args.topk]["instance_energy_rank"] = kk

                Dict[c_type].append(max(List_i))
                Dict_all[c_type].append(List_i)
                argmax_dict[c_type].append(np.argmax(List_i))
                energy_dict[c_type].append(energy_i)
                concept_energy_dict[c_type].append(concept_energy_i)
            best_acc_dict[c_type] = np.mean(Dict[c_type])
            Dict_all[c_type] = np.array(Dict_all[c_type])
            energy_dict[c_type] = np.array(energy_dict[c_type])
            concept_energy_dict[c_type] = np.array(concept_energy_dict[c_type])
            graph_distance_min_id = np.array(data_record[f"pairwise_distance_topk_{c_types[0]}"]).mean(-1).argmin(-1)
            array = np.array(data_record[f"pairwise_distance_topk_{c_types[0]}"]).mean(-1)
            mode = stats.mode(array, axis=1)[0]
            graph_distance_mode_id = np.argmax(mode==array, axis=1)
            graph_distance_min_acc_dict[c_type] = np.mean([Dict_all[c_type][i, idx] for i, idx in enumerate(graph_distance_min_id)])
            graph_distance_mode_acc_dict[c_type] = np.mean([Dict_all[c_type][i, idx] for i, idx in enumerate(graph_distance_mode_id)])
            energy_min_acc_dict[c_type] = np.mean([Dict_all[c_type][i, idx] for i, idx in enumerate(energy_dict[c_type].argmin(1))])
            concept_energy_min_acc_dict[c_type] = np.mean([Dict_all[c_type][i, idx] for i, idx in enumerate(concept_energy_dict[c_type].argmin(1))])

        df = pd.DataFrame(df_dict_list)
        rank1_acc_dict = {c_type: np.mean(data_record[f"is_isomorphic_{c_type}_{0}"]) for c_type in c_types}
        rank1_distance_edit_dict = {c_type: np.mean(data_record[f"distance_edit_{c_type}_{0}"]) for c_type in c_types}
        for key in ["concept_model_hash", "relation_model_hash", "SGLD_is_penalize_lower", "SGLD_is_penalize_lower_seq",
                   "init", "infer_order", "lambd", "is_bidirectional_re"]:
            dff_dict[key] = getattr(args, key)
        dff_dict["tasks_completed"] = len(data_record[f'graph_trimmed_{c_types[0]}_0'])
        dff_dict["hash"] = filename.split("_")[-2]

        dff_dict["acc_rank1"] = np.mean(list(rank1_acc_dict.values()))
        dff_dict["distance_edit_rank1"] = np.mean(list(rank1_distance_edit_dict.values()))
        dff_dict["acc_weighted"] = df.groupby(by=["instance_energy_rank"]).mean().iloc[0]["is_isomorphic"]
        dff_dict["acc_best"] = np.mean(list(best_acc_dict.values()))
        dff_dict["acc_graph_min"] = np.mean(list(graph_distance_min_acc_dict.values()))
        dff_dict["acc_graph_mode"] = np.mean(list(graph_distance_mode_acc_dict.values()))
        dff_dict["acc_energy_min"] = np.mean(list(energy_min_acc_dict.values()))
        dff_dict["acc_concept_energy_min"] = np.mean(list(concept_energy_min_acc_dict.values()))
        dff_dict_list.append(dff_dict)
        display(df.groupby(by="c_type").mean())
        if verbose >= 2:
            print("Acc for rank-1 model:")
            pp.pprint(rank1_acc_dict)
            print("Avg rank-1 acc: {:.4f}".format(np.mean(list(rank1_acc_dict.values()))))
            print()
            print("Acc for graph_distance_min:")
            pp.pprint(graph_distance_min_acc_dict)
            print("Avg graph_distance_min acc: {:.4f}".format(np.mean(list(graph_distance_min_acc_dict.values()))))
            print()
            print("Acc for graph_distance_mode:")
            pp.pprint(graph_distance_mode_acc_dict)
            print("Avg graph_distance_mode acc: {:.4f}".format(np.mean(list(graph_distance_mode_acc_dict.values()))))
            print()
            print("Acc for energy_min:")
            pp.pprint(energy_min_acc_dict)
            print("Avg energy_min acc: {:.4f}".format(np.mean(list(energy_min_acc_dict.values()))))
            print()
            print("Acc for concept_energy_min:")
            pp.pprint(concept_energy_min_acc_dict)
            print("Avg concept_energy_min acc: {:.4f}".format(np.mean(list(concept_energy_min_acc_dict.values()))))
            print()
            print("Best acc:")
            pp.pprint(best_acc_dict)
            print("Avg best acc: {:.4f}".format(np.mean(list(best_acc_dict.values()))))
            print()

            display(df.groupby(by=["instance_energy_rank"]).mean().style.background_gradient(cmap=plt.cm.get_cmap("PiYG_r")))
    dff = pd.DataFrame(dff_dict_list)
    mean_edit_distance = np.mean([
        np.mean(data_record["distance_edit_ex_0_0"]),
        np.mean(data_record["distance_edit_ex_1_0"]),
        np.mean(data_record["distance_edit_ex_2_0"])])
    acc_top1 = np.mean(data_record[f"is_isomorphic_0"])
    print("iso_acc: {:.4f}    edit_distance: {:.4f}".format(acc_top1, mean_edit_distance))


# In[ ]:


if args.is_analysis and args.evaluation_type.startswith("yc-parse+classify"):
    keys = ["concept_model_hash", "relation_model_hash"]
    display(dff.groupby(by=keys).mean().style.background_gradient(cmap=plt.cm.get_cmap("PiYG")))


# #### 4.2.1.2 Classify:

# In[ ]:


if args.is_analysis and args.evaluation_type.startswith("yc-parse+classify"):
    dirname = EXP_PATH + "/evaluation_yc-parse+classify^classify_1-21/"
    filenames = filter_filename(dirname, include=".p", exclude="df_")
    df_dict_list = []
    for filename in filenames:
        df_dict = {}
        p.print(filename, is_datetime=False)
        try:
            data_record = pload(dirname + filename)
        except Exception as e:
            print(e)
            continue
        df_dict = deepcopy(update_default_hyperparam_generalization(data_record["args"]))
        df_dict["acc"] = np.mean(data_record["results"]["acc"])
        df_dict["iou"] = np.mean(data_record["results"]["iou"])
        df_dict["tasks"] = len(data_record["results"]["iou"])
        # df_dict["hash"] = 
        df_dict_list.append(df_dict)
    df = pd.DataFrame(df_dict_list)


# In[ ]:


if args.is_analysis and args.evaluation_type.startswith("yc-parse+classify"):
    df_group = groupby_add_keys(
            df,
            by=["load_parse_src","ensemble_size", "gpuid",
               ],
            add_keys=[],
            other_keys=["acc", "iou","tasks",],
            mode={
                "mean": ["acc", "iou","tasks"],
                "count": None,
            }
        )
    pdump({"df": df, "df_group": df_group}, dirname + "df_{}-{}.p".format(datetime.now().month, datetime.now().day))
    display(df_group.style.background_gradient(cmap=plt.cm.get_cmap("PiYG")))

