#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from collections import OrderedDict, Counter
from copy import deepcopy
import itertools
import json
import matplotlib.pylab as plt
import networkx as nx
from networkx.algorithms import isomorphism
from networkx.readwrite import json_graph
import numpy as np
import pdb
import pickle
from scipy import optimize
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..', '..'))
from zeroc.concept_library.settings import REPR_DIM
from zeroc.concept_library.concepts import BaseGraph, Graph, Concept, Concept_Ensemble, Placeholder, Tensor, IS_VIEW, CONCEPTS, NEW_CONCEPTS, OPERATORS, to_Concept, parse_obj
from zeroc.concept_library.util import find_connected_components, find_connected_components_colordiff, visualize_dataset, visualize_matrices, combine_pos, to_Graph, get_last_output, broadcast_inputs, get_Dict_with_first_key
from zeroc.concept_library.util import seperate_concept, get_input_output_mode_dict, find_valid_operators, get_inputs_targets, get_patch, set_patch, score_fun_IoU, shrink, plot_matrices, to_tuple, get_indices, get_hashing
from zeroc.concept_library.util import to_np_array, record_data, to_Variable, split_string, get_next_available_key, make_dir, sort_two_lists, try_remove, print_banner
from zeroc.concept_library.util import get_key_of_largest_value, broadcast_keys, remove_duplicates, split_bucket, get_next_available_key, switch_dict_keys
from zeroc.utils import load_dataset
IS_VIEW = True
IS_CUDA = False

# CONCEPTS = OrderedDict()     # Predefined concepts
# NEW_CONCEPTS = OrderedDict() # Newly learned concepts
# OPERATORS = OrderedDict()    # Predefined operators


# ## Parsing:

# In[ ]:


# Helper functions:
def get_trans(patches, cost_dict, OPERATORS):
    """Obtain all transformations and the corresponding cost for the patches."""
    patch_trans_dict = OrderedDict()
    for name, patch in patches.items():
        patch_dict = OrderedDict()
        for op_name in cost_dict:
            patch_dict[op_name] = OPERATORS[op_name].copy()(patch)
        patch_trans_dict[name] = patch_dict
    return patch_trans_dict


def mask_target_patch(target_patch, mask, pos_rel):
    """Update the target_patch according to the mask and the mask's relative position in the target."""
    assert isinstance(target_patch, Concept)
    value = target_patch.get_node_value()
    value_new = get_patch(value, pos_rel) * mask
    pos = target_patch.get_node_value("pos")
    pos_new = (int(pos[0] + pos_rel[0]), int(pos[1] + pos_rel[1]), int(pos_rel[2]), int(pos_rel[3]))
    target_patch_new = deepcopy(target_patch)
    target_patch_new.set_node_value(value_new).set_node_value(pos_new, "pos")
    return target_patch_new


def visualize_objs(objs, image_sizes):
    if not isinstance(objs, list):
        objs = [objs]
    assert len(objs) <= 2
    matrices = []
    for i, (obj, image_size) in enumerate(zip(objs, image_sizes)):
        value = torch.zeros(image_size)
        pos = obj.get_node_value("pos")
        value[..., int(pos[0]): int(pos[0] + pos[2]), int(pos[1]): int(pos[1] + pos[3])] = obj.get_node_value()
        matrices.append(value)
    plot_matrices(matrices)


def visualize_pair_list(pair_list_all, image_sizes):
    for pair_list in pair_list_all:
        print("{}, target relative position {}:".format(pair_list["op_names"], pair_list["relpos"]))
        visualize_objs([pair_list["input"], pair_list["target"]], image_sizes)


def visualize_pair(pair):
    """Visualize a full parsed pair_list."""
    for example_key, pair_example in pair.items():
        print_banner("Example {}:\n".format(example_key), banner_size=42)
        for pair_info in pair_example["pair_list"]:
            print("{}, input '{}', target '{}', relpos {}:".format(pair_info["op_names"], pair_info["input_key"], pair_info["target_key"], pair_info["relpos"]))
            plot_matrices([pair_info["input"].get_node_value().long(), pair_info["target"].get_node_value().long()])
        print("Unexplained:")
        plot_matrices([pair_example["info"]["input_unexplained"].long(), pair_example["info"]["target_unexplained"].long()])


def get_relpos(pos_input, pos_target):
    """Get relative position from the pos of the input and pos of the target."""
    pos_input, pos_target = to_tuple(pos_input), to_tuple(pos_target)
    assert pos_input[2:] == pos_target[2:]
    return (pos_target[0] - pos_input[0], pos_target[1] - pos_input[1])


def trans_pos(pos, relpos):
    """Obtain the translated position from starting pos and relpos."""
    if isinstance(pos, Concept):
        pos = pos.get_root_value().to(pos.device)
    if isinstance(relpos, Concept):
        relpos = relpos.get_root_value().to(relpos.device)
    pos = torch.cat([pos[:1] + relpos[0], pos[1:2] + relpos[1], pos[2:]])
    return pos


def get_pos(image):
    """Obtain the default pos of the patch."""
    if isinstance(image, Concept):
        image = image.get_root_value()
    return to_Variable([0, 0, image.shape[0], image.shape[1]]).to(image.device)


def get_color(image):
    """Obtain the color of the patch. If more than one color, return -1."""
    if isinstance(image, Concept):
        image = image.get_root_value()
    if image is None or np.prod(image.shape) == 0:
        device = image.device if image is not None else torch.device("cpu")
        return to_Variable([-1]).to(device)
    image_color = torch.unique(image)
    image_color = image_color[image_color!=0]
    if len(image_color) == 1:
        return image_color
    else:
        # Having more than one color:
        return to_Variable([-1]).to(image.device)


def get_pair_list_from_mask(pair_list, input, target):
    """Get the pair_list from the masked pair_list."""
    def get_obj(mask_obj, image):
        """Get the image object from the mask object."""
        if isinstance(image, Concept):
            image = image.get_node_value()
        pos = mask_obj.get_node_value("pos")
        image_selected = get_patch(image, pos)
        image_masked = deepcopy(mask_obj)
        image_masked = image_masked.set_node_value(image_selected).set_node_value(pos, "pos")
        return image_masked
    pair_list_new = []
    for pair in pair_list:
        pair_new = {"op_names": pair["op_names"], "relpos": pair["relpos"]}
        pair_new["input"] = get_obj(pair["input"], input)
        pair_new["target"] = get_obj(pair["target"], target)
        pair_new["input_key"] = pair["input_key"]
        pair_new["target_key"] = pair["target_key"]
        pair_list_new.append(pair_new)
    return pair_list_new


def load_task(task_id, directory=None, isplot=True):
    """Load task into inputs and targets with option of plotting."""
    dataset = load_dataset(task_id, directory=directory)
    dataset_concept = to_Graph(dataset, CONCEPTS["Image"])
    if isplot:
        visualize_dataset(dataset)
    train_dataset = dataset_concept["train"]
    inputs, targets = get_inputs_targets(train_dataset)
    test_dataset = dataset_concept["test"]
    inputs_test, targets_test = get_inputs_targets(test_dataset)
    return (inputs, targets), (inputs_test, targets_test)


# In[ ]:


# Core functions:
def get_distance_square(obj1, obj2, norm_dis=30, max_margin=3):
    assert isinstance(obj1, Concept) and isinstance(obj2, Concept)
    patch1 = obj1.get_root_value()
    patch2 = obj2.get_root_value()
    shape1 = patch1.shape
    shape2 = patch2.shape

    best_match_list = []
    best_match_dict = {}
    if shape2[-2] - max_margin <= shape1[-2] <= shape2[-2] and shape2[-1] - max_margin <= shape1[-1] <= shape2[-1]:
        for i in range(shape2[-2] - shape1[-2] + 1):
            for j in range(shape2[-1] - shape1[-1] + 1):
                mask1 = patch1 > 0
                if len(shape1) == 3:
                    mask1 = mask1.any(0, keepdims=True)
                patch2_select = patch2[..., i: i + shape1[-2], j: j + shape1[-1]]
                match_matrix = ((patch2_select == patch1) & mask1) | (~mask1)
                if match_matrix.all():
                    best_match_list.append((i, j))
        pos1 = obj1.get_node_value("pos")
        pos2 = obj2.get_node_value("pos")
        for i, j in best_match_list:
            pos2_select = (int(pos2[0]) + i, int(pos2[1]) + j, int(pos1[2]), int(pos1[3]))
            dis_square = (pos1[0] - pos2_select[0]) ** 2 + (pos1[1] - pos2_select[1]) ** 2
            best_match_dict[pos2_select] = {"cost": dis_square / norm_dis ** 2,
                                            "target_mask": patch1 > 0,
                                            "target_pos_rel": (i, j, shape1[-2], shape1[-1]),
                                           }
    return best_match_dict


def get_pairwise_distance(input_patch_trans, target_patches, cost_dict, norm_dis=30, max_margin=3, is_obj_weight=True):
    """Get pairwise cost between all transformations of each input patch and all target patches"""
    distance_dict = OrderedDict()
    for i, input_trans in enumerate(input_patch_trans.values()):
        for j, target in enumerate(target_patches.values()):
            for op_name, input_trans_ele in input_trans.items():
                distance_dict[(i, j, op_name)] = get_distance_square(input_trans_ele, target, norm_dis=norm_dis, max_margin=max_margin)
                for arg in distance_dict[(i, j, op_name)]:
                    if is_obj_weight:
                        target_mask = distance_dict[(i, j, op_name)][arg]["target_mask"]
                        cost = cost_dict[op_name] * (1 + (target_mask.shape[-1] + target_mask.shape[-2]) / 30)
                    else:
                        cost = cost_dict[op_name]
                    distance_dict[(i, j, op_name)][arg]["explained"] = target_mask.sum()
                    distance_dict[(i, j, op_name)][arg]["cost"] += cost
    return distance_dict


def segment(
    input,
    target,
    input_unexplained=None,
    target_unexplained=None,
    cost_dict=None,
    max_cost=10,
    norm_dis=30,
    is_color=True,
    input_given_objs=None,
    target_given_objs=None,
):
    def get_best_pos(result):
        pos_best = None
        cost_min = np.Inf
        for pos in result:
            cost = to_np_array(result[pos]["cost"])
            if cost < cost_min:
                pos_best = pos
                cost_min = cost
        return pos_best

    def dict_2_tensor(distance_dict, max_cost=10):
        cost_tensor = torch.ones(len(input_patches), len(target_patches), len(cost_dict)) * max_cost
        explained_tensor = torch.zeros(len(input_patches), len(target_patches), len(cost_dict))
        for i in range(len(input_patches)):
            for j in range(len(target_patches)):
                for k, op_name in enumerate(cost_dict):
                    result = distance_dict[(i, j, op_name)]
                    if len(result) > 0:
                        pos_best = get_best_pos(result)
                        cost_tensor[i, j, k] = result[pos_best]["cost"]
                        distance_dict[(i, j, op_name)] = {pos_best: result[pos_best]}
                        explained_tensor[i, j, k] = result[pos_best]["explained"]
        return cost_tensor, explained_tensor

    info = {}
    # Initialize tensors for unexplained:
    if input_unexplained is None:
        input_unexplained = input.get_node_value() > 0
        if len(input_unexplained.shape) == 3:
            input_unexplained = input_unexplained.any(0)
    else:
        input_unexplained = deepcopy(input_unexplained)
    if target_unexplained is None:
        target_unexplained = target.get_node_value() > 0
        if len(target_unexplained.shape) == 3:
            target_unexplained = target_unexplained.any(0)
    else:
        target_unexplained = deepcopy(target_unexplained)

    if input_given_objs is not None and len(input_given_objs) > 0:
        input_patches = {obj_name: input.get_attr(obj_name) for obj_name in input_given_objs}
        info["input_objs_unexplained"] = deepcopy(input_given_objs)
    else:
        info["input_objs_unexplained"] = None
        if is_color:
            input_patches = to_Concept(find_connected_components_colordiff(input.get_node_value() * input_unexplained))
        else:
            input_patches = to_Concept(find_connected_components(input.get_node_value() * input_unexplained))
    if target_given_objs is not None and len(target_given_objs) > 0:
        target_patches = {obj_name: target.get_attr(obj_name) for obj_name in target_given_objs}
        info["target_objs_unexplained"] = deepcopy(target_given_objs)
    else:
        info["target_objs_unexplained"] = None
        if is_color:
            target_patches = to_Concept(find_connected_components_colordiff(target.get_node_value() * target_unexplained))
        else:
            target_patches = to_Concept(find_connected_components(target.get_node_value() * target_unexplained))

    if cost_dict is None:
        cost_dict = {"Identity": 0, "hFlip": 1, "vFlip": 1, "RotateA": 1, "RotateB": 1, "RotateC": 1, "DiagFlipA": 1, "DiagFlipB": 1}

    input_patch_trans = get_trans(input_patches, cost_dict, OPERATORS)
    distance_dict = get_pairwise_distance(
        input_patch_trans,
        target_patches,
        cost_dict,
        norm_dis=norm_dis,
        max_margin=3 if target_given_objs is None else 0,
    )
    cost_tensor, explained_tensor = dict_2_tensor(distance_dict, max_cost=max_cost)
    val_min, arg_min = cost_tensor.min(-1)
    row_ind, col_ind = linear_sum_assignment(val_min)
    ratio_vec = torch.zeros(len(row_ind))
    for mm, (i, j) in enumerate(zip(row_ind, col_ind)):
        ratio_vec[mm] = explained_tensor[i, j, arg_min[i, j]] / (0.1 + val_min[i, j])
    ratio_vec_sorted, order_sorted = sort_two_lists(ratio_vec, np.arange(len(ratio_vec)), reverse=True)

    input_keys = list(input_patches.keys())
    target_keys = list(target_patches.keys())

    # Update unexplained region with assignment:
    pair_list = []
    # Starting from the pair that has the largest explained_sum / cost:
    for mm in order_sorted:
        i, j = row_ind[mm], col_ind[mm]
        if val_min[i, j] < max_cost:
            min_ids = torch.where(cost_tensor[i,j] == val_min[i, j])[0]
            op_names = [list(cost_dict.keys())[id] for id in min_ids]
            Dict = distance_dict[(i, j, op_names[0])]
            assert len(Dict) == 1
            pos = list(Dict.keys())[0] # target patch position
            set_patch(input_unexplained, input_patches[input_keys[i]].get_node_value(), input_patches[input_keys[i]].get_node_value("pos"), 0)
            set_patch(target_unexplained, input_patch_trans[input_keys[i]][op_names[0]].get_node_value(), pos, 0)
            if input_given_objs is not None:
                for key in input.get_descendants(input_keys[i]):
                    if key in info["input_objs_unexplained"]:
                        info["input_objs_unexplained"].remove(key)
                info["input_objs_unexplained"].remove(input_keys[i])
            if target_given_objs is not None:
                for key in target.get_descendants(target_keys[j]):
                    if key in info["target_objs_unexplained"]:
                        info["target_objs_unexplained"].remove(key)
                if target_keys[j] in info["target_objs_unexplained"]:
                    info["target_objs_unexplained"].remove(target_keys[j])

            pair = {}
            pair["op_names"] = op_names
            pair["input"] = input_patches[input_keys[i]]
            target_mask = Dict[pos]["target_mask"]
            target_relpos = Dict[pos]["target_pos_rel"]
            pair["target"] = mask_target_patch(target_patches[target_keys[j]], target_mask, target_relpos)
            pair["relpos"] = get_relpos(input_patch_trans[input_keys[i]][op_names[0]].get_node_value("pos"), pair["target"].get_node_value("pos"))
            if pair["relpos"][0] != 0 or pair["relpos"][1] != 0:
                pair["op_names"].extend(["Trans", "Move"])
            if input_patch_trans[input_keys[i]][op_names[0]].get_node_value("color").item() !=             pair["target"].get_node_value("color"):
                pair["op_names"].extend(["Color"])
            pair["input_key"] = input_keys[i]
            pair["target_key"] = target_keys[j]
            pair_list.append(pair)

            # Early stop if everything is explained:
            if input_unexplained.sum() == 0 or target_unexplained.sum() == 0:
                break

    info["input_unexplained"] = input_unexplained
    info["target_unexplained"] = target_unexplained
    return pair_list, info


def parse_uni(input, target, input_unexplained=None, target_unexplained=None, is_color=True, input_given_objs=None, target_given_objs=None, isplot=False):
    """Uni-direction parsing: from input to target."""
    pair_list_all = []
    info = {"input_unexplained": input_unexplained, "target_unexplained": target_unexplained,
            "input_objs_unexplained": input_given_objs, "target_objs_unexplained": target_given_objs}
    for i in range(5):
        pair_list, info = segment(
            input, target, info["input_unexplained"], info["target_unexplained"], is_color=is_color,
            input_given_objs=info["input_objs_unexplained"], target_given_objs=info["target_objs_unexplained"])
        pair_list_all += pair_list
        if info["input_unexplained"].sum() == 0 or info["target_unexplained"].sum() == 0 or len(pair_list) == 0 or             ((input_given_objs is not None and len(info["input_objs_unexplained"]) == 0) and (target_given_objs is not None and len(info["target_objs_unexplained"]) == 0)):
            break
    if isplot:
        visualize_pair_list(pair_list_all, [input.get_node_value().shape, target.get_node_value().shape])
        print("Unexplained:")
        plot_matrices([info["input_unexplained"].long(), info["target_unexplained"].long()])
    return pair_list_all, info


def parse_bi(input, target, input_unexplained=None, target_unexplained=None, is_color=True, suffix="", input_given_objs=None, target_given_objs=None, isplot=False):
    """Bidirectional parsing. Both from input to target and from target to input."""
    pair_list_all, info = parse_uni(
        input, target, input_unexplained, target_unexplained, input_given_objs=input_given_objs, target_given_objs=target_given_objs, is_color=is_color)
    if info["input_unexplained"].sum() > 0 and info["target_unexplained"].sum() > 0         and ((input_given_objs is None or (info["input_objs_unexplained"] is not None and len(info["input_objs_unexplained"]) > 0))         and (target_given_objs is None or (info["target_objs_unexplained"] is not None and len(info["target_objs_unexplained"]) > 0))):
        pair_list_all_reverse, info = parse_uni(
            target, input, info["target_unexplained"], info["input_unexplained"], is_color=is_color,
            input_given_objs=info["target_objs_unexplained"], target_given_objs=info["input_objs_unexplained"],
        )
        switch_dict_keys(info, "input_unexplained", "target_unexplained")
        switch_dict_keys(info, "input_objs_unexplained", "target_objs_unexplained")
        pair_list_all += [{"input": pair["target"], "target": pair["input"], "op_names": [ele + "_reverse" for ele in pair["op_names"]], "relpos": pair["relpos"],
                           "input_key": pair["target_key"], "target_key": pair["input_key"]} 
                          for pair in pair_list_all_reverse]
    if len(suffix) > 0:
        for pair_list in pair_list_all:
            pair_list["op_names"] = [ele + suffix for ele in pair_list["op_names"]]
    if isplot:
        visualize_pair_list(pair_list_all, [input.get_node_value().shape, target.get_node_value().shape])
        print("Unexplained:")
        plot_matrices([info["input_unexplained"].long(), info["target_unexplained"].long()])
    return pair_list_all, info


def parse_pair_ele(input, target, use_given_objs=False, isplot=True, cache_dirname=None):
    # Color and shape matching via patches segmented with connected components:
    if cache_dirname is not None:
        string_repr = input.get_string_repr("obj") + "^^^" + target.get_string_repr("obj") + "^^^given_obj:{}".format(use_given_objs)
        filename_hash = get_hashing(string_repr)
        path = os.path.join(cache_dirname, filename_hash + ".p")
        try:
            pair_dict = pickle.load(open(path, "rb"))
            return pair_dict["pair_list_all"], pair_dict["info"]
        except:
            pass
    if use_given_objs:
        input_given_objs = ["$root"] + input.obj_names
        target_given_objs = ["$root"] + target.obj_names
    else:
        input_given_objs = target_given_objs = None
    pair_list_all, info = parse_bi(input, target, is_color=False, input_given_objs=input_given_objs, target_given_objs=target_given_objs)
    if info["input_unexplained"].sum() > 0 and info["target_unexplained"].sum() > 0 and not use_given_objs:
        # Color and shape matching via patches segmented with connected components of the same color:
        pair_list, info = parse_bi(input, target, info["input_unexplained"], info["target_unexplained"], is_color=True)
        pair_list_all += pair_list
    info["input_objs_unexplained"] = try_remove(info["input_objs_unexplained"], "$root")
    info["target_objs_unexplained"] = try_remove(info["target_objs_unexplained"], "$root")
    if info["input_unexplained"].sum() > 0 and info["target_unexplained"].sum() > 0         and (input_given_objs is None or (info["input_objs_unexplained"] is not None and len(info["input_objs_unexplained"]) > 0))         and (target_given_objs is None or (info["target_objs_unexplained"] is not None and len(info["target_objs_unexplained"]) > 0)):
        # Shape Matching via patches segmented with connected compnents:
        pair_list, info = parse_bi(input>0, target>0, info["input_unexplained"], info["target_unexplained"], suffix="_changeColor",
                                   input_given_objs=info["input_objs_unexplained"], 
                                   target_given_objs=info["target_objs_unexplained"])
        pair_list = get_pair_list_from_mask(pair_list, input, target)
        pair_list_all += pair_list
    if isplot:
        visualize_pair_list(pair_list_all, [input.get_node_value().shape, target.get_node_value().shape])
        print("Unexplained:")
        plot_matrices([info["input_unexplained"].long(), info["target_unexplained"].long()])
    if cache_dirname is not None:
        pair_dict = {}
        pair_dict["pair_list_all"] = pair_list_all
        pair_dict["info"] = info
        make_dir(path)
        pickle.dump(pair_dict, open(path, "wb"))
    return pair_list_all, info


def parse_pair(inputs, targets, use_given_objs=False, isplot=True, cache_dirname=None):
    """Parse input and target correspondence for multiple examples."""
    if not isinstance(inputs, dict):
        inputs = {0: inputs}
    if not isinstance(targets, dict):
        targets = {0: targets}
    parse_result = OrderedDict()
    for example_id, input in inputs.items():
        if isplot:
            print("=" * 30 + "\nExample {}:\n".format(example_id) + "=" * 30)
        target = targets[example_id]
        pair_list, info = parse_pair_ele(input, target, use_given_objs=use_given_objs, isplot=isplot, cache_dirname=cache_dirname)
        parse_result[example_id] = {"pair_list": pair_list, "info": info}
    return parse_result


# ## Relation functions:

# In[ ]:


def SameShape(obj1, obj2, is_NN_pathway=False):
    image1 = obj1.get_node_value()
    image2 = obj2.get_node_value()
    if np.prod(image1.shape) == 0:
        return False
    if np.prod(image2.shape) == 0:
        return False
    if image1.shape != image2.shape:
        return False
    else:
        return (image1.bool() == image2.bool()).all()


def SameColor(obj1, obj2, is_NN_pathway=False):
    color1 = obj1.get_node_value("color")
    color2 = obj2.get_node_value("color")
    if color1 == -1 or color2 == -1:
        return False
    else:
        return color1 == color2


def SameAll(obj1, obj2, is_NN_pathway=False):
    image1 = obj1.get_node_value()
    image2 = obj2.get_node_value()
    if np.prod(image1.shape) == 0:
        return False
    if np.prod(image2.shape) == 0:
        return False
    if image1.shape != image2.shape:
        return False
    else:
        return (image1 == image2).all()


def SameRow(obj1, obj2, is_NN_pathway=False):
    pos1 = obj1.get_node_value("pos")
    pos2 = obj2.get_node_value("pos")
    if pos1[0] == pos2[0] and pos1[2] == pos2[2]:
        return True
    else:
        return False


def SameCol(obj1, obj2, is_NN_pathway=False):
    pos1 = obj1.get_node_value("pos")
    pos2 = obj2.get_node_value("pos")
    if pos1[1] == pos2[1] and pos1[3] == pos2[3]:
        return True
    else:
        return False


def SubsetOf(obj1, obj2, is_NN_pathway=False):
    source = obj1.get_node_value()
    target = obj2.get_node_value()
    pos1 = obj1.get_node_value("pos")
    pos2 = obj2.get_node_value("pos")
    if np.prod(source.shape) == 0:
        return False
    if np.prod(target.shape) == 0:
        return False
    if source.shape[0] > target.shape[0]:
        return False
    if source.shape[1] > target.shape[1]:
        return False
    if not (pos1[0] >= pos2[0] and pos1[1] >= pos2[1] and pos1[0] + pos1[2] <= pos2[0] + pos2[2] and pos1[1] + pos1[3] <= pos2[1] + pos2[3]):
        return False

    is_subset_all = False
    for k in range(target.shape[-2] - source.shape[-2] + 1):
        for l in range(target.shape[-1] - source.shape[-1] + 1):
            is_subset = True
            for i in range(source.shape[-2]):
                for j in range(source.shape[-1]):
                    if source[..., i, j].bool().any():
                        if not (source[..., i, j] == target[..., i + k, j + l]).all():
                            is_subset = False
                            break
            if is_subset:
                is_subset_all = True
                break
    return is_subset_all


def IsInside(obj1, obj2, is_NN_pathway=False):
    """Whether obj1 is inside obj2."""
    pos1 = obj1.get_node_value("pos")
    pos2 = obj2.get_node_value("pos")
    if pos1[0] > pos2[0] and pos1[1] > pos2[1] and pos1[0] + pos1[2] < pos2[0] + pos2[2] and pos1[1] + pos1[3] < pos2[1] + pos2[3]:
        image1 = obj1.get_node_value()
        image2 = obj2.get_node_value()
        image2_patch = image2[..., int(pos1[0] - pos2[0]): int(pos1[0] + pos1[2] - pos2[0]), 
                              int(pos1[1] - pos2[1]): int(pos1[1] + pos1[3] - pos2[1])]
        overlap = (image1 != 0) & (image2_patch != 0)
        if overlap.any():
            return False
        else:
            return True
    else:
        return False


def IsTouch(obj, obj2, is_NN_pathway=False):
    """Whether the "obj"'s leftmost/rightmost/upmost/downmost part touches any other pixels (up, down, left, right) or boundary in the "image"."""
    obj_indices = get_indices(
        obj.get_node_value(),
        obj.get_node_value("pos"),
        includes_self=False,
        includes_neighbor=True,
    )
    obj2_indices = get_indices(obj2.get_node_value(), obj2.get_node_value("pos"))
    is_torch = len(set(obj_indices).intersection(set(obj2_indices))) > 0
    return is_torch


def OutOfBound(obj, image, is_NN_pathway=False):
    """Check if the obj is completely out of bound w.r.t. the image."""
    obj_pos = obj.get_node_value("pos")
    image_pos = image.get_node_value("pos")
    if image_pos[0] > 0 or image_pos[1] > 0:
        return False
    else:
        return obj_pos[0] + obj_pos[2] <= 0 or obj_pos[0] >= image_pos[2] or                obj_pos[1] + obj_pos[3] <= 0 or obj_pos[1] >= image_pos[3]


def isParallel(line1, line2, is_NN_pathway=False):
    """Whether line1 is parallel to line2."""
    lines = [line1.get_node_value(), line2.get_node_value()]
    directions = []
    for line in lines:
        shape = line.shape
        if shape[0] > shape[1]:
            directions.append("0")
        elif shape[0] < shape[1]:
            directions.append("1")
        else:
            raise Exception("Line must have unequal height and width!")
    return len(set(directions)) == 1


def isVertical(line1, line2, is_NN_pathway=False):
    """Whether line1 is vertical to line2."""
    lines = [line1.get_node_value(), line2.get_node_value()]
    directions = []
    for line in lines:
        shape = line.shape
        if shape[0] > shape[1]:
            directions.append("0")
        elif shape[0] < shape[1]:
            directions.append("1")
        else:
            raise Exception("Line must have unequal height and width!")
    return len(set(directions)) == 2


def IsNonOverlapXY(img1, img2):
    """If img1 and img2 have nooverlap."""
    pos1 = img1.get_node_value("pos")
    pos2 = img2.get_node_value("pos")
    if (pos2[0] < pos1[0] + pos1[2]) & (pos2[1] < pos1[1] + pos1[3]) & (pos1[0] < pos2[0] + pos2[2]) & (pos1[1] < pos2[1] + pos2[3]):
        return False
    else:
        return True


def IsEnclosed(img1, img2):
    return IsInside(img2, img1)


# ## Theory functions:

# In[ ]:


class Theory(Graph):
    def __init__(
        self,
        selector=None,
        inplace=True,
        **kwargs
    ):
        super(Theory, self).__init__(**kwargs)
        self.selector = selector
        self.inplace = inplace


    def backward_chaining(self):
        raise NotImplementedError


    def validate_theory(self, input, target):
        raise NotImplementedError


    def refine_selector(self):
        raise NotImplementedError


    def refine_predictor(self):
        raise NotImplementedError


# In[ ]:


def Identity(obj, is_NN_pathway=False):
    if is_NN_pathway:
        return obj
    else:
        return obj.copy()


def Trans(obj, relpos, is_NN_pathway=False):
    """Translate the object by given relative position (relpos)"""
    obj_o = obj.copy()
    obj_o.set_node_value(trans_pos(obj.get_node_value("pos"), relpos), "pos")
    # Also operate on the component objects:
    for obj_name in obj_o.obj_names:
        name_pos = obj_o.operator_name(obj_name) + "^pos"
        pos_ori = obj_o.get_node_value(name_pos)
        obj_o.set_node_value(trans_pos(pos_ori, relpos), name_pos)
    return obj_o


def hFlip(obj, is_NN_pathway=False):
    """Horizontal flip."""
    obj_o = obj.copy()
    pos_full = obj_o.get_node_value("pos")
    obj_o.set_node_value(obj.get_node_value().flip(-1))
    # Also operate on the component objects:
    for obj_name in obj_o.obj_names:
        obj_o.set_node_value(obj_o.get_node_value(obj_name).flip(-1), obj_name)
        name_pos = obj_o.operator_name(obj_name) + "^pos"
        pos_ori = obj_o.get_node_value(name_pos)
        pos = [pos_ori[0], (pos_full[1] + pos_full[3]) - (pos_ori[1] + pos_ori[3]), pos_ori[2], pos_ori[3]]
        obj_o.set_node_value(pos, name_pos)
    return obj_o


def vFlip(obj, is_NN_pathway=False):
    """Vertical flip."""
    obj_o = obj.copy()
    pos_full = obj_o.get_node_value("pos")
    obj_o.set_node_value(obj.get_node_value().flip(-2))
    # Also operate on the component objects:
    for obj_name in obj_o.obj_names:
        obj_o.set_node_value(obj_o.get_node_value(obj_name).flip(-2), obj_name)
        name_pos = obj_o.operator_name(obj_name) + "^pos"
        pos_ori = obj_o.get_node_value(name_pos)
        pos = [(pos_full[0] + pos_full[2]) - (pos_ori[0] + pos_ori[2]), pos_ori[1], pos_ori[2], pos_ori[3]]
        obj_o.set_node_value(pos, name_pos)
    return obj_o


def RotateA(obj, is_NN_pathway=False):
    """Rotate counter-clockwise by 90deg."""
    obj_o = obj.copy()
    pos_full = obj_o.get_node_value("pos")
    obj_o.set_node_value(torch.rot90(obj.get_node_value(), k=1, dims=(-2, -1)))
    obj_o.set_node_value(torch.stack([pos_full[0], pos_full[1], pos_full[3], pos_full[2]]), "pos")
    # Also operate on the component objects:
    for obj_name in obj_o.obj_names:
        name_pos = obj_o.operator_name(obj_name) + "^pos"
        pos_ori = obj_o.get_node_value(name_pos)
        obj_o.set_node_value(torch.rot90(obj_o.get_node_value(obj_name), k=1, dims=(-2, -1)), obj_name)
        pos = [(pos_full[1] + pos_full[3]) - (pos_ori[1] + pos_ori[3]),
               pos_ori[0] - pos_full[0],
               pos_ori[3],
               pos_ori[2]]
        obj_o.set_node_value(pos, name_pos)
    return obj_o


def RotateB(obj, is_NN_pathway=False):
    """Rotate counter-clockwise by 180deg."""
    obj_o = obj.copy()
    pos_full = obj_o.get_node_value("pos")
    obj_o.set_node_value(torch.rot90(obj.get_node_value(), k=2, dims=(-2, -1)))
    # Also operate on the component objects:
    for obj_name in obj_o.obj_names:
        name_pos = obj_o.operator_name(obj_name) + "^pos"
        pos_ori = obj_o.get_node_value(name_pos)
        obj_o.set_node_value(torch.rot90(obj_o.get_node_value(obj_name), k=2, dims=(-2, -1)), obj_name)
        pos = [(pos_full[0] + pos_full[2]) - (pos_ori[0] + pos_ori[2]),
               (pos_full[1] + pos_full[3]) - (pos_ori[1] + pos_ori[3]),
               pos_ori[2],
               pos_ori[3]]
        obj_o.set_node_value(pos, name_pos)
    return obj_o


def RotateC(obj, is_NN_pathway=False):
    """Rotate clockwise by 90deg."""
    obj_o = obj.copy()
    pos_full = obj_o.get_node_value("pos")
    obj_o.set_node_value(torch.rot90(obj.get_node_value(), k=3, dims=(-2, -1)))
    obj_o.set_node_value(torch.stack([pos_full[0], pos_full[1], pos_full[3], pos_full[2]]), "pos")
    # Also operate on the component objects:
    for obj_name in obj_o.obj_names:
        name_pos = obj_o.operator_name(obj_name) + "^pos"
        pos_ori = obj_o.get_node_value(name_pos)
        obj_o.set_node_value(torch.rot90(obj_o.get_node_value(obj_name), k=3, dims=(-2, -1)), obj_name)
        pos = [pos_ori[1] - pos_full[1],
               (pos_full[0] + pos_full[2]) - (pos_ori[0] + pos_ori[2]),
               pos_ori[3],
               pos_ori[2]]
        obj_o.set_node_value(pos, name_pos)
    return obj_o


def DiagFlipA(obj, is_NN_pathway=False):
    """Diagonally flip an object along the axis of lower-left and upper-right."""
    return hFlip(RotateA(obj))


def DiagFlipB(obj, is_NN_pathway=False):
    """Diagonally flip an object along the axis of upper-left and lower-right."""
    return vFlip(RotateA(obj))


def Draw(obj, color, is_NN_pathway=False):
    """Draw a certain color on an object."""
    obj = obj.copy()
    tensor = obj.get_node_value()
    if isinstance(color, Concept):
        color = color.get_node_value()
    if not isinstance(color, torch.LongTensor):
        color = color.round().long()
    tensor[tensor > 0] = color.type(tensor.dtype)
    obj.set_node_value(tensor)
    return obj


def DrawRect(obj, color, pos, is_NN_pathway=False):
    """Draw a rectangle of given color on the position of the object"""
    obj = obj.copy()
    tensor = obj.get_node_value()
    obj_pos = obj.get_node_value("pos")
    if isinstance(color, Concept):
        color = color.get_node_value()
    if not isinstance(color, torch.LongTensor):
        color = color.round().long()
    if isinstance(pos, Concept):
        pos = pos.get_node_value()
    pos_rel = (int(pos[0] - obj_pos[0]), int(pos[1] - obj_pos[1]), int(pos[2]), int(pos[3]))
    set_patch(tensor, torch.ones(int(pos[2]), int(pos[3])), pos_rel, color)
    obj.set_node_value(tensor)
    return obj


def ShrinkSize(pos, is_NN_pathway=False):
    """Shrink the size of the rectangle (specified by pos) by 2 in the inner."""
    pos = pos.copy()
    pos_value = pos.get_node_value()
    pos_value[:2] = pos_value[:2] + 1
    pos_value[2:] = pos_value[2:] - 2
    pos.set_node_value(pos_value)
    return pos


def Scale(obj, num, is_NN_pathway=False):
    """Scale the size of the object by num times."""
    obj = obj.copy()
    obj_value = obj.get_node_value()
    if isinstance(num, Concept):
        num = num.get_node_value()
    num = int(num)
    
    # Perform kronecker product
    scaler = torch.ones(num, num)
    obj_value_reshaped = torch.einsum("ab,cd->acbd", obj_value, scaler)
    obj_value = obj_value_reshaped.view(obj_value.size(0)*scaler.size(0), obj_value.size(1)*scaler.size(1))

    # Set the value back:
    obj.set_node_value(obj_value)
    return obj


def SelectObjOnSize(image, rank, is_NN_pathway=False):
    """Select the object based on its rank in length."""
    def get_size(obj):
        return int(max(to_np_array(obj.get_node_value("pos")[2:])))
    if isinstance(rank, Concept):
        rank = rank.get_node_value()
    rank = int(to_np_array(rank))
    lengths = {}
    objs = {}
    for obj_name, obj in image.objs.items():
        lengths[obj_name] = get_size(obj)
        objs[obj_name] = obj
    lengths_sorted, obj_name_sorted = sort_two_lists(list(lengths.values()), list(lengths.keys()), reverse=True)
    return objs[obj_name_sorted[rank]]


def SelectBiggest(image, is_NN_pathway=False):
    """Select the biggest object."""
    biggest_key = ''
    biggest_value = -1
    objs = image.objs
    for key, value in objs.items():
        pos = value.get_node_value("pos")
        area = pos[2] * pos[3]
        if area > biggest_value:
            biggest_key = key
            biggest_value = area
    return objs[biggest_key]


def Move(obj, obj1, relpos, is_NN_pathway=False):
    """Move obj1 to the relpos of the obj."""
    obj_copy = obj.copy()
    obj1_copy = obj1.copy()
    pos_obj1 = obj1.get_node_value("pos")
    if isinstance(relpos, Concept):
        relpos = relpos.get_node_value()
    obj1_copy.set_node_value([relpos[0], relpos[1], pos_obj1[2], pos_obj1[3]], "pos")
    obj_copy.add_obj(obj1_copy)
    return obj_copy


def Copy(obj, relpos, color, is_NN_pathway=False):
    """Copy obj1 to the relpos of the obj with the specified color."""
    obj_copy = obj.copy()
    pos_obj = obj.get_node_value("pos")
    if isinstance(relpos, Concept):
        relpos = relpos.get_node_value()
    obj_copy.set_node_value([pos_obj[0] + relpos[0], pos_obj[1] + relpos[1], pos_obj[2], pos_obj[3]], "pos")
    obj_copy = Draw(obj_copy, color)
    return obj_copy


def Combine(pos, *objs, is_NN_pathway=False):
    """Draw multiple objects in a blank image."""
    if isinstance(pos, Concept):
        pos = pos.get_node_value()
    tensor = torch.zeros(int(pos[2]), int(pos[3]))
    for obj in objs:
        set_patch(tensor, obj.get_node_value(), obj.get_node_value("pos"))
    image = CONCEPTS["Image"].copy().set_node_value(tensor)
    return image


def Remainder(obj, obj1, is_NN_pathway=False):
    """Obtain the remainder of obj1 inside obj."""
    obj_copy = obj.copy()
    obj1_copy = obj1.copy()
    final_score, info = score_fun_IoU(obj_copy, obj1_copy)
    assert info["pred_size_compare"] == ("larger", "larger")
    best_idx = info["best_idx"]
    tensor1 = obj1.get_node_value()
    pos_obj1 = [best_idx[0], best_idx[1], tensor1.shape[-2], tensor1.shape[-1]]
    tensor = set_patch(obj_copy.get_node_value(), tensor1, pos_obj1, 0)
    tensor, pos = shrink(tensor)
    obj_copy.set_node_value(tensor)
    obj_copy.set_node_value(pos, "pos")
    return obj_copy


def RemainderObj(obj, obj1, is_NN_pathway=False):
    """Obtain the remainder of obj1 inside obj."""
    obj_copy = obj.copy()
    obj1_copy = obj1.copy()
    obj_copy.remove_attr_with_value(obj1)
    return obj_copy


def is_horizontal(ref_pos, obj_pos):
    """
    Helper function for evaluating whether the object can be achieved through horizontal transformations. 
    If it is possible, return the number of pixels to move (positive means moving right).
    Otherwise, the function returns zero.
    """
    if obj_pos[0] >= ref_pos[0] and obj_pos[0] + obj_pos[2] <= ref_pos[0] + ref_pos[2]:
        return max(obj_pos[1] - ref_pos[1], obj_pos[1] + obj_pos[3] - ref_pos[1] - ref_pos[3],
                   key=lambda k: abs(k))
    return 0


def is_vertical(ref_pos, obj_pos):
    """
    Helper function for evaluating whether the object can be achieved through vertical transformations. 
    If it is possible, return the number of pixels to move (positive means moving down).
    Otherwise, the function returns zero.
    """
    if obj_pos[1] >= ref_pos[1] and obj_pos[1] + obj_pos[3] <= ref_pos[1] + ref_pos[3]:
        return max(obj_pos[0] - ref_pos[0], obj_pos[0] + obj_pos[2] - ref_pos[0] - ref_pos[2],
                   key=lambda k: abs(k))
    return 0


def is_diagonal(ref_pos, obj_pos):
    """
    Helper function for evaluating whether the object can be achieved through vertical transformations. 
    If it is possible, return the number of pixels to move in y and x direction
    (positive means moving down/right), respectively.
    Otherwise, the function returns zero.
    """
    dist = []
    if obj_pos[0] - ref_pos[0] == obj_pos[1] - ref_pos[1]:
        dist.append((obj_pos[0] - ref_pos[0], obj_pos[1] - ref_pos[1]))
    if obj_pos[0] + obj_pos[2] - ref_pos[0] - ref_pos[2] == obj_pos[1] + obj_pos[3] - ref_pos[1] - ref_pos[3]:
        dist.append((obj_pos[0] + obj_pos[2] - ref_pos[0] - ref_pos[2],
                     obj_pos[1] + obj_pos[3] - ref_pos[1] - ref_pos[3]))
    if obj_pos[0] + obj_pos[2] - ref_pos[0] - ref_pos[2] == -obj_pos[1] + ref_pos[1]:
        dist.append((obj_pos[0] + obj_pos[2] - ref_pos[0] - ref_pos[2], obj_pos[1] - ref_pos[1]))
    if -obj_pos[0] + ref_pos[0] == obj_pos[1] + obj_pos[3] - ref_pos[1] - ref_pos[3]:
        dist.append((obj_pos[0] - ref_pos[0], obj_pos[1] + obj_pos[3] - ref_pos[1] - ref_pos[3]))
    if dist:
        return max(dist, key = lambda k: k[0] ** 2 + k[1] ** 2)
    return 0


def GetRelPos(obj, obj_ref, is_NN_pathway=False):
    """Get the relative position (RelPos) of the obj w.r.t. obj_ref."""
    ref_pos = obj_ref.get_node_value("pos")
    obj_pos = obj.get_node_value("pos")
    diff = 0
    horizontal = is_horizontal(ref_pos, obj_pos)
    vertical = is_vertical(ref_pos, obj_pos)
    diagonal = is_diagonal(ref_pos, obj_pos)
    if horizontal:
        diff = [0, horizontal.item()]
    elif vertical:
        diff = [vertical.item(), 0]
    elif diagonal:
        diff = [diagonal[0].item(), diagonal[1].item()]
    return CONCEPTS["RelPos"].copy().set_node_value(diff)


# ### ParseRect:

# In[ ]:


# Object segmentation (Seperate object into rectangles, lines or pixels)
def get_end(matrix, i, j, dis_x, dis_y):
    """Get the end of the line containing an element with given direction.
    Return the index of the end.
    """
    if bool(dis_x) == bool(dis_y):
        raise ValueError("Change in only one of the dimensions can be nonzero.")
    m, n = matrix.shape
    if not(0 <= i < m and 0 <= j < n and matrix[i, j]):
        return i, j
    i, j = i - dis_x, j - dis_y
    while 0 <= i + dis_x < m and 0 <= j + dis_y < n and matrix[i + dis_x, j + dis_y]:
        i, j = i + dis_x, j + dis_y
        empty_neighbor, up, down, left, right = get_empty_neighbor(matrix, i, j)
        if empty_neighbor == 0:
            if dis_x:
                if (matrix[i + dis_x, j - 1] or matrix[i + dis_x, j + 1]):
                    return i - dis_x, j - dis_y
            elif matrix[i - 1, j + dis_y] or matrix[i + 1, j + dis_y]:
                return i - dis_x, j - dis_y
        elif empty_neighbor == 1:
            if (up and right and matrix[i - 1, j + 1]) or (right and down and matrix[i + 1, j + 1]) or                (down and left and matrix[i + 1, j - 1]) or (left and up and matrix[i - 1, j - 1]):
                return i, j
        elif empty_neighbor == 2:
            if not(up and down) and not(left and right):
                new_i = i - up + down
                new_j = j - left + right
                if 0 <= new_i < m and 0 <= new_j < n and matrix[new_i, new_j]:
                    return i, j
    return i, j


def parseRect(obj):
    """Operator. Parse an object into rectangles, lines, and pixels."""
    result = obj.copy()
    matrix = to_np_array(result.get_node_value())
    all_list = seperate_concept((matrix > 0).astype(int))
    cur_list = []
    # Combine three lists
    if "RectSolid" in all_list.keys():
        cur_list = all_list['RectSolid']
    if "Line" in all_list.keys():
        cur_list.extend(all_list['Line'])
    if "Pixel" in all_list.keys():
        cur_list.extend(all_list['Pixel'])

    # Add each seperated rectangles/lines/pixels to the current node
    for pos in cur_list:
        value = torch.FloatTensor(matrix[pos[0]][pos[1]] * np.ones((pos[2], pos[3])))
        cur_obj = CONCEPTS["Image"].copy().set_node_value(value)
        cur_obj.set_node_value(pos, "pos")
        result.add_obj(cur_obj)
    return result

def deleteParsing(obj):
    """Operator. Delete all children of the node (which was parsed into rectangles, lines, and pixels before)."""
    obj_copy = obj.copy()
    return obj_copy.get_root_value()


# In[ ]:


def find_max_hollow(matrix, rect, max_width=100):
    matrix = matrix.copy()
    matrix[rect[0]: rect[0] + rect[2], rect[1]: rect[1] + rect[3]] = np.ones((rect[2], rect[3]))
    m = len(matrix)
    n = len(matrix[0])

    left = [0] * n
    right = [n] * n
    height = [0] * n

    maxarea = 0
    result = (0, 0, 0, 0)

    for i in range(m):
        cur_left, cur_right = 0, n
        # update height
        for j in range(n):
            if matrix[i][j] == 0:
                height[j] = 0
            else:
                height[j] += 1
        # update left
        for j in range(n):
            if matrix[i][j] == 0:
                left[j] = 0
                cur_left = j + 1
            else:
                left[j] = max(left[j], cur_left)
        # update right
        for j in range(n-1, -1, -1):
            if matrix[i][j] == 0:
                right[j] = n
                cur_right = j
            else:
                right[j] = min(right[j], cur_right)
        # update the area
        for j in range(n):
            tmp = height[j] * (right[j] - left[j])
            loc = (i - height[j] + 1, left[j], height[j], right[j] - left[j])
            if tmp > maxarea and loc[0] < rect[0] < loc[0] + loc[2] and loc[1] < rect[1] < loc[1] + loc[3]                 and rect[0] + rect[2] <= loc[0] + loc[2] and rect[1] + rect[3] <= loc[1] + loc[3] and                 max(rect[0] - loc[0], rect[1] - loc[1], loc[0] + loc[2] - rect[0] - rect[2], 
                    loc[1] + loc[3] - rect[1] - rect[3]) <= max_width:
                maxarea = tmp
                result = loc
    square = np.ones((result[2], result[3]))
    output = np.zeros_like(matrix)
    output[result[0]: result[0] + result[2], result[1]: result[1] + result[3]] = square
    output[rect[0]: rect[0] + rect[2], rect[1]: rect[1] + rect[3]] = np.zeros((rect[2], rect[3]))
    return output

def no_nearby(matrix, loc):
    if loc[0] - 1 >= 0:
        for i in range(loc[3]):
            if matrix[loc[0] - 1][loc[1] + i] == 0:
                return False
        if loc[1] - 1 >= 0:
            if matrix[loc[0] - 1][loc[1] - 1] == 0:
                return False
        if loc[1] + loc[3] < matrix.shape[1]:
            if matrix[loc[0] - 1][loc[1] + loc[3]] == 0:
                return False
    if loc[0] + loc[2] < matrix.shape[0]:
        for i in range(loc[3]):
            if matrix[loc[0] + loc[2]][loc[1] + i] == 0:
                return False
        if loc[1] - 1 >= 0:
            if matrix[loc[0] + loc[2]][loc[1] - 1] == 0:
                return False
        if loc[1] + loc[3] < matrix.shape[1]:
            if matrix[loc[0] + loc[2]][loc[1] + loc[3]] == 0:
                return False
    if loc[1] - 1 >= 0:
        for i in range(loc[2]):
            if matrix[loc[0] + i][loc[1] - 1] == 0:
                return False
    if loc[1] + loc[3] < matrix.shape[1]:
        for i in range(loc[2]):
            if matrix[loc[0] + i][loc[1] + loc[3]] == 0:
                return False
    return True

def at_border(matrix, loc):
    return loc[0] == 0 or loc[1] == 0 or loc[0] + loc[2] == len(matrix) or loc[1] + loc[3] == len(matrix[0])

def get_hollow(obj):
    obj = obj.copy()
    if type(obj) is not np.ndarray:
        matrix = to_np_array(obj.get_node_value())
    else:
        matrix = obj
    all_dict = seperate_concept(np.logical_not(matrix > 0).astype(int))
    all_list = [x for v in all_dict.values() for x in v]
    result = []
    for loc in all_list:
        if not at_border(matrix, loc) and no_nearby(matrix, loc):
            result.append(loc)
    result = [find_max_hollow(matrix, rect) for rect in result]
    return result


# ## Operator and Concept definitions:

# In[ ]:


num_colors = 9
dim = 2

########################################################
# Concepts:
########################################################
CONCEPTS["Bool"] = Concept(name="Bool",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    value=Placeholder(Tensor(dtype="bool", shape=(1,), range=[True, False])))

CONCEPTS["Cat"] = Concept(name="Cat",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    value=Placeholder(Tensor(dtype="cat")))

CONCEPTS["Color"] = Concept(name="Color",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    value=Placeholder(Tensor(dtype="cat", shape=(1,), range=range(num_colors))))

CONCEPTS["Pos"] = Concept(name="Pos",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    value=Placeholder(Tensor(dtype="cat", shape=(4,), range=range(dim))))

CONCEPTS["RelPos"] = Concept(name="RelPos",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    value=Placeholder(Tensor(dtype="cat", shape=(2,), range=range(dim - 1))))

CONCEPTS["Image"] = Concept(name="Image",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    inherit_to=["Line"],
    value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
    attr={"pos": Placeholder("Pos"),
          "color": (Placeholder("Color"), get_color)})

CONCEPTS["Line"] = Concept(name="Line",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    inherit_from=["Image"],
    value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
    attr={"pos": Placeholder("Pos"),
          "color": (Placeholder("Color"), get_color)})

CONCEPTS["Lshape"] = Concept(name="Lshape",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    inherit_from=["Image"],
    value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
    attr={"pos": Placeholder("Pos"),
          "color": (Placeholder("Color"), get_color)})

CONCEPTS["Rect"] = Concept(name="Rect",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    inherit_from=["Image"],
    value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
    attr={"pos": Placeholder("Pos"),
          "color": (Placeholder("Color"), get_color)})

CONCEPTS["RectSolid"] = Concept(name="RectSolid",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    inherit_from=["Image"],
    value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
    attr={"pos": Placeholder("Pos"),
          "color": (Placeholder("Color"), get_color)})

# Newly added:
CONCEPTS["Randshape"] = Concept(name="Randshape",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    inherit_from=["Image"],
    value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
    attr={"pos": Placeholder("Pos"),
          "color": (Placeholder("Color"), get_color)})

CONCEPTS["ARCshape"] = Concept(name="ARCshape",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    inherit_from=["Image"],
    value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
    attr={"pos": Placeholder("Pos"),
          "color": (Placeholder("Color"), get_color)})

CONCEPTS["Tshape"] = Concept(name="Tshape",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    inherit_from=["Image"],
    value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
    attr={"pos": Placeholder("Pos"),
          "color": (Placeholder("Color"), get_color)})

CONCEPTS["Fshape"] = Concept(name="Fshape",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    inherit_from=["Image"],
    value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
    attr={"pos": Placeholder("Pos"),
          "color": (Placeholder("Color"), get_color)})

CONCEPTS["Eshape"] = Concept(name="Eshape",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    inherit_from=["Image"],
    value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
    attr={"pos": Placeholder("Pos"),
          "color": (Placeholder("Color"), get_color)})

CONCEPTS["Hshape"] = Concept(name="Hshape",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    inherit_from=["Image"],
    value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
    attr={"pos": Placeholder("Pos"),
          "color": (Placeholder("Color"), get_color)})

CONCEPTS["Cshape"] = Concept(name="Cshape",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    inherit_from=["Image"],
    value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
    attr={"pos": Placeholder("Pos"),
          "color": (Placeholder("Color"), get_color)})

CONCEPTS["Ashape"] = Concept(name="Ashape",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    inherit_from=["Image"],
    value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
    attr={"pos": Placeholder("Pos"),
          "color": (Placeholder("Color"), get_color)})

########################################################
# Operators:
########################################################
OPERATORS["SameShape"] = Graph(name="SameShape",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image"), Placeholder("Image")],
             "output": Placeholder("Bool"),
             "fun": SameShape,
            })

OPERATORS["SameColor"] = Graph(name="SameColor",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image"), Placeholder("Image")],
             "output": Placeholder("Bool"),
             "fun": SameColor,
            })

OPERATORS["SameAll"] = Graph(name="SameAll",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image"), Placeholder("Image")],
             "output": Placeholder("Bool"),
             "fun": SameAll,
            })

OPERATORS["SameRow"] = Graph(name="SameRow",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image"), Placeholder("Image")],
             "output": Placeholder("Bool"),
             "fun": SameRow,
            })

OPERATORS["SameCol"] = Graph(name="SameCol",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image"), Placeholder("Image")],
             "output": Placeholder("Bool"),
             "fun": SameCol,
            })

OPERATORS["SubsetOf"] = Graph(name="SubsetOf",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image"), Placeholder("Image")],
             "output": Placeholder("Bool"),
             "fun": SubsetOf,
            })

OPERATORS["IsInside"] = Graph(name="IsInside",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image"), Placeholder("Image")],
             "output": Placeholder("Bool"),
             "fun": IsInside,
            })

OPERATORS["IsTouch"] = Graph(name="IsTouch",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image"), Placeholder("Image")],
             "output": Placeholder("Bool"),
             "fun": IsTouch,
            })

OPERATORS["OutOfBound"] = Graph(name="OutOfBound",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image"), Placeholder("Image")],
             "output": Placeholder("Bool"),
             "fun": OutOfBound,
            })

OPERATORS["Parallel"] = Graph(name="Parallel",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Line"), Placeholder("Line")],
             "output": Placeholder("Bool"),
             "fun": isParallel,
            })

OPERATORS["Vertical"] = Graph(name="Vertical",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Line"), Placeholder("Line")],
             "output": Placeholder("Bool"),
             "fun": isVertical,
            })


OPERATORS["Identity"] = Graph(name="Identity",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image")],
             "output": Placeholder("Image"),
             "fun": Identity,
            })

OPERATORS["hFlip"] = Graph(name="hFlip",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image")],
             "output": Placeholder("Image"),
             "fun": hFlip,
            })

OPERATORS["vFlip"] = Graph(name="vFlip",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image")],
             "output": Placeholder("Image"),
             "fun": vFlip,
            })

OPERATORS["RotateA"] = Graph(name="RotateA",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image")],
             "output": Placeholder("Image"),
             "fun": RotateA,
            })

OPERATORS["RotateB"] = Graph(name="RotateB",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image")],
             "output": Placeholder("Image"),
             "fun": RotateB,
            })

OPERATORS["RotateC"] = Graph(name="RotateC",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image")],
             "output": Placeholder("Image"),
             "fun": RotateC,
            })

OPERATORS["DiagFlipA"] = Graph(name="DiagFlipA",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image")],
             "output": Placeholder("Image"),
             "fun": DiagFlipA,
            })

OPERATORS["DiagFlipB"] = Graph(name="DiagFlipB",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image")],
             "output": Placeholder("Image"),
             "fun": DiagFlipB,
            })

OPERATORS["Draw"] = Graph(name="Draw",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image"), Placeholder("Color")],
             "output": Placeholder("Image"),
             "fun": Draw,
            })

OPERATORS["DrawRect"] = Graph(name="DrawRect",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image"), Placeholder("Color"), Placeholder("Pos")],
             "output": Placeholder("Image"),
             "fun": DrawRect,
            })

OPERATORS["ShrinkSize"] = Graph(name="ShrinkSize",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Pos")],
             "output": Placeholder("Pos"),
             "fun": ShrinkSize,
            })

OPERATORS["Scale"] = Graph(name="Scale",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image"), Placeholder("Cat")],
             "output": Placeholder("Image"),
             "fun": Scale,
            })

OPERATORS["SelectObjOnSize"] = Graph(name="SelectObjOnSize",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image"), Placeholder("Cat")],
             "output": Placeholder("Image"),
             "fun": SelectObjOnSize,
            })

OPERATORS["SelectBiggest"] = Graph(name="SelectBiggest",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image")],
             "output": Placeholder("Image"),
             "fun": SelectBiggest,
            })

OPERATORS["Trans"] = Graph(name="Trans",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image"), Placeholder("RelPos")],
             "output": Placeholder("Image"),
             "fun": Trans,
            })

OPERATORS["Move"] = Graph(name="Move",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image"), Placeholder("Image"), Placeholder("RelPos")],
             "output": Placeholder("Image"),
             "fun": Move,
            })

OPERATORS["Copy"] = Graph(name="Copy",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image"), Placeholder("RelPos"), Placeholder("Color")],
             "output": Placeholder("Image"),
             "fun": Copy,
            })

OPERATORS["Combine"] = Graph(name="Combine",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Pos"), Placeholder("multi*Image")],
             "output": Placeholder("Image"),
             "fun": Combine,
            })

OPERATORS["Remainder"] = Graph(name="Remainder",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image"), Placeholder("Image")],
             "output": Placeholder("Image"),
             "fun": Remainder,
            })

OPERATORS["RemainderObj"] = Graph(name="RemainderObj",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image"), Placeholder("Image")],
             "output": Placeholder("Image"),
             "fun": RemainderObj,
            })

OPERATORS["GetRelPos"] = Graph(name="GetRelPos",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image"), Placeholder("Image")],
             "output": Placeholder("RelPos"),
             "fun": GetRelPos,
            })

OPERATORS["parseRect"] = Graph(name="parseRect",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image")],
             "output": Placeholder("Image"),
             "fun": parseRect,
            })

OPERATORS["deleteParsing"] = Graph(name="deleteParsing",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image")],
             "output": Placeholder("Image"),
             "fun": deleteParsing,
            })


OPERATORS["VerticalMid"] = Graph(name="VerticalMid",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Line"), Placeholder("Line")],
             "output": Placeholder("Bool"),
             "fun": isVertical,
            })

OPERATORS["VerticalEdge"] = Graph(name="VerticalEdge",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Line"), Placeholder("Line")],
             "output": Placeholder("Bool"),
             "fun": isVertical,
            })

OPERATORS["VerticalSepa"] = Graph(name="VerticalSepa",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Line"), Placeholder("Line")],
             "output": Placeholder("Bool"),
             "fun": isVertical,
            })

OPERATORS["IsNonOverlapXY"] = Graph(name="IsNonOverlapXY",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image"), Placeholder("Image")],
             "output": Placeholder("Bool"),
             "fun": IsNonOverlapXY,
            })

OPERATORS["IsEnclosed"] = Graph(name="IsEnclosed",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image"), Placeholder("Image")],
             "output": Placeholder("Bool"),
             "fun": IsEnclosed,
            })


CONCEPTS["Red"] = Concept(name="Red",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    inherit_from=["Image"],
    value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
    attr={"pos": Placeholder("Pos"),
          "color": (Placeholder("Color"), get_color)})

CONCEPTS["Green"] = Concept(name="Green",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    inherit_from=["Image"],
    value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
    attr={"pos": Placeholder("Pos"),
          "color": (Placeholder("Color"), get_color)})

CONCEPTS["Blue"] = Concept(name="Blue",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    inherit_from=["Image"],
    value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
    attr={"pos": Placeholder("Pos"),
          "color": (Placeholder("Color"), get_color)})

CONCEPTS["Cube"] = Concept(name="Cube",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    inherit_from=["Image"],
    value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
    attr={"pos": Placeholder("Pos"),
          "color": (Placeholder("Color"), get_color)})

CONCEPTS["Cylinder"] = Concept(name="Cylinder",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    inherit_from=["Image"],
    value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
    attr={"pos": Placeholder("Pos"),
          "color": (Placeholder("Color"), get_color)})

CONCEPTS["Large"] = Concept(name="Large",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    inherit_from=["Image"],
    value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
    attr={"pos": Placeholder("Pos"),
          "color": (Placeholder("Color"), get_color)})

CONCEPTS["Small"] = Concept(name="Small",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    inherit_from=["Image"],
    value=Placeholder(Tensor(dtype="cat", range=range(num_colors))),
    attr={"pos": Placeholder("Pos"),
          "color": (Placeholder("Color"), get_color)})

OPERATORS["SameSize"] = Graph(name="SameSize",
    repr=to_Variable(torch.rand(REPR_DIM), is_cuda=IS_CUDA),
    forward={"args": [Placeholder("Image"), Placeholder("Image")],
             "output": Placeholder("Bool"),
             "fun": Identity,
            })


# In[ ]:


if __name__ == "__main__":
    ####################
    # ARC:
    ####################
    print("=" * 50 + "\nARC:\n" + "=" * 50)
    task = '3.json'
    max_cost = 10
    print(task + ":")
    (inputs, targets), (inputs_test, targets_test) = load_task(task, directory="concept_env/task_files/lev1_0")

#     example_id = 0  # Choose example index
#     input, target = inputs[0][example_id], targets[example_id]
#     print("=" * 20 + "\nParsing:\n" + "=" * 20)
#     input, target = parse_obj([input, target])
#     pair_list_all, info = parse_pair_ele(input, target, use_given_objs=True, isplot=True)

    example_id = 0  # Choose example index
    input, target = inputs[0][example_id], targets[example_id]
    print("=" * 20 + "\nParsing:\n" + "=" * 20)
    input, target = parse_obj([input, target])
    pair_list_all, info = parse_pair_ele(input, target, use_given_objs=True, isplot=True)

    print("\nAfter parseRect:")
    input_parse_rect = OPERATORS["parseRect"](input)
    pair_list_all, info = parse_pair_ele(input_parse_rect, target, use_given_objs=True, isplot=True)


# In[ ]:


if __name__ == "__main__":
    ####################
    # ARC:
    ####################
    print("=" * 50 + "\nARC:\n" + "=" * 50)
    task = '0c53a9ac_syn.json'
    max_cost = 10
    print(task + ":")
    (inputs, targets), (inputs_test, targets_test) = load_task(task, directory="concept_env/task_files/lev1_1_aug")

#     example_id = 0  # Choose example index
#     input, target = inputs[0][example_id], targets[example_id]
#     print("=" * 20 + "\nParsing:\n" + "=" * 20)
#     input, target = parse_obj([input, target])
#     pair_list_all, info = parse_pair_ele(input, target, use_given_objs=True, isplot=True)

    example_id = 1  # Choose example index
    input, target = inputs[0][example_id], targets[example_id]
    print("=" * 20 + "\nParsing:\n" + "=" * 20)
    input, target = parse_obj([input, target])
    pair_list_all, info = parse_pair_ele(input, target, use_given_objs=True, isplot=True)


# In[ ]:


if __name__ == "__main__":
    ####################
    # ARC:
    ####################
    print("=" * 50 + "\nARC:\n" + "=" * 50)
    task_list = [
#         "9edfc990.json",
#         "7c008303.json",
#         "0dfd9992.json",
#         "6cdd2623.json",
#         "445eab21.json",
        "776ffc46.json", # 0
        "846bdb03.json", # 1
        "25d487eb.json", # 2
        "6d58a25d.json", # 3
        "2c608aff.json", # 4
        "47c1f68c.json", # 5
        "1f642eb9.json", # 6
        "d07ae81c.json", ### 7 Need line concept
        "3f7978a0.json",
        "0e206a2e.json",
        "08ed6ac7.json",
        "e8dc4411.json",
        "cbded52d.json",
        "ba97ae07.json", #
        "264363fd.json", 
        "6aa20dc0.json",
        "e9afcf9a.json",
        "8e1813be.json",
        # Easy:
        "85c4e7cd.json",
        "9dfd6313.json",
        "74dd1130.json", ## 15
        "846bdb03.json",
        "3c9b0459.json",
        "6150a2bd.json",
        "67a3c6ac.json",
        "68b16354.json",
        "ed36ccf7.json",
    ]
    task_id = 0     # Choose task index
    example_id = 0  # Choose example index
    max_cost = 10
    task = task_list[task_id]
    print(task + ":")
    (inputs, targets), (inputs_test, targets_test) = load_task(task)
    input, target = inputs[0][example_id], targets[example_id]

    # Parsing:
    print("=" * 20 + "\nParsing:\n" + "=" * 20)
    input, target = parse_obj([input, target])
    pair_list_all, info = parse_pair_ele(input, target, use_given_objs=True, isplot=True)


#     ####################
#     # Atari game:
#     ####################
#     print("=" * 50 + "\nAtari Breakout:\n" + "=" * 50)
#     dictionary = pickle.load(open("../results/Atari/breakout.p", "rb"))
#     obs_list = dictionary['obs_list']
#     t = 51
#     input = CONCEPTS["Image"].copy().set_node_value(obs_list[t])
#     target = CONCEPTS["Image"].copy().set_node_value(obs_list[t + 1])
#     plot_matrices([input.get_node_value(), target.get_node_value()])

#     # Parsing:
#     print("=" * 20 + "\nParsing:\n" + "=" * 20)
#     pair_list_all, (input_unexplained, target_unexplained) = parse_pair_ele(input, target)

    # Add objects:
    input_graph = input.copy()
    target_graph = target.copy()
    for pair in pair_list_all:
        input_obj = pair["input"]
        target_obj = pair["target"]
        op_names = pair["op_names"]
        relpos = pair["relpos"]
        input_obj_name = input_graph.add_obj(input_obj)
        target_obj_name = target_graph.add_obj(target_obj)
        pair["input_obj_name"] = input_obj_name
        pair["target_obj_name"] = target_obj_name

    # Add relations within input and target:
    input_graph.add_relations(OPERATORS)
    target_graph.add_relations(OPERATORS)

    # Add relations between input and target:
    concept_ensemble = Concept_Ensemble()
    concept_ensemble.add_concept(input_graph, "input")
    concept_ensemble.add_concept(target_graph, "target")
    concept_ensemble.add_relations("input", "target", OPERATORS)
    concept_ensemble.add_theories("input", "target", pair_list_all)

    ## Select objects:
    node_names = ["obj_0:Image", "obj_2:Image", "obj_5:Image"]
    concept_pattern = input_graph.get_concept_pattern(node_names)
    concept_pattern.set_pivot_nodes("obj_0").set_refer_nodes(["obj_5"])
    print(input_graph.get_refer_nodes(concept_pattern))

    ## Perform operation on subset of objects:
    op = OPERATORS["RotateA"].copy()
    op.set_selector(concept_pattern)
    op.set_inplace(True)
    op(input_graph).draw()

    ## Test the transformation operators:
    for op_name, op in OPERATORS.items():
        if len(op.input_placeholder_nodes) == 1 and len(op.dangling_nodes) == 0:
            output = op(input_graph)
            output_save = output.copy()
            for obj_name, obj in output.objs.items():
                output.add_obj(obj, change_root=True)
            is_same = output == output_save
            print(op_name, is_same)
            if not is_same:
                raise


# In[ ]:


if __name__ == "__main__":
    temp = np.array([[1,1,1,1,1], 
                 [1,0,0,1,1], 
                 [1,0,0,1,0], 
                 [0,1,1,1,1],
                 [0,1,0,1,1]])

    get_hollow(temp)

    temp = np.array([[0,1,1,1,0], 
                 [0,1,1,1,0], 
                 [0,0,1,0,0], 
                 [0,1,1,1,0],
                 [0,1,1,1,0]])

    get_hollow(temp)

    temp = np.array([[1,1,1,1,0], 
                 [1,0,0,1,0], 
                 [1,0,0,1,0], 
                 [1,0,0,1,0],
                 [1,1,1,1,0]])

    get_hollow(temp)

    temp = np.array([[0,1,1,1,0], 
                 [0,1,1,1,0], 
                 [0,0,1,0,0],
                 [0,0,1,0,0],
                 [0,1,1,0,0],
                 [0,1,1,0,0],
                 [0,1,1,0,0], 
                 [0,1,1,0,0], 
                 [0,0,1,0,0], 
                 [0,1,1,1,0],
                 [0,1,1,1,0],
                 [0,1,1,1,0], 
                 [0,1,1,1,0], 
                 [0,0,1,0,0],
                 [0,0,1,0,0],
                 [0,1,1,0,0],
                 [0,1,1,0,0],
                 [0,1,1,0,0], 
                 [0,1,1,0,0], 
                 [0,0,1,0,0], 
                 [0,1,1,1,0],
                 [0,1,1,1,0]])

    get_hollow(temp)

