## ZeroC: A Neuro-Symbolic Model for Zero-shot Concept Recognition and Acquisition at Inference Time

[Paper](https://arxiv.org/abs/2206.15049) | [Slides](https://docs.google.com/presentation/d/1WAR4dZ0J2E-u3V_DgYBYTF4mDCmRk0FXI8GPlM2kqdQ/edit?usp=sharing) | [Project page](https://snap.stanford.edu/zeroc/)

This is the official repo for [ZeroC: A Neuro-Symbolic Model for Zero-shot Concept Recognition and Acquisition at Inference Time](https://arxiv.org/abs/2206.15049) (Wu et al. NeurIPS 2022). ZeroC is a neuro-symbolic architecture that after pretrained with simpler concepts and relations, can recognize and acquire novel, more complex concepts in a zero-shot way. 

<a href="url"><img src="https://github.com/snap-stanford/zeroc/blob/master/assets/hierarchy.png" align="center" width="600" ></a>

ZeroC represents concepts as graphs of constituent concept models (as nodes) and their relations (as edges). We design ZeroC architecture so that it allows a **one-to-one mapping** between an abstract, symbolic graph structure of a concept and its corresponding energy-based model that provides a probability model for concepts and relations (by summing over constituent concept and relation EBMs according to the graph, see figure above). After pre-trained with simpler concepts and relations, at inference time, without further training, ZeroC can perform:

* **Zero-shot concept recognition**: recognize a more complex, hierarchical concept from image, given the symbolic graph structure of this hierarchical concept.

* **Zero-shot concept aquisition**: given a single demonstration of an image containing a hierarchical concept, infer its graph structure. Furthermore, this graph structure can be transferred across domains (e.g., from 2D image to 3D image), allowing the ZeroC in the second domain to directly recognize such hierarchical concept (see figure below):

<a href="url"><img src="https://github.com/snap-stanford/zeroc/blob/master/assets/2d3d_zeroc.png" align="center" width="500" ></a>

# Installation

1. First clone the directory. This repo depends on the [concept_library](https://github.com/tailintalent/concept_library) for functionality of concept classes and model architecture, and [BabyARC](https://github.com/frankaging/BabyARC) as a dataset-generating engine. Therefore, run the following command to initialize the submodules:

```code
git submodule init; git submodule update
```
(If showing error of no permission, need to first [add a new SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).)

2. Install dependencies.

Create a new environment using [conda](https://docs.conda.io/en/latest/miniconda.html), with Python >= 3.7. Install [PyTorch](https://pytorch.org/) (version >= 1.10.1). The repo is tested with PyTorch version of 1.10.1 and there is no guarentee that other version works. Then install other dependencies via:
```code
pip install -r requirements.txt
```

# Dataset:
The dataset files can be generated using the BabyARC engine with the `datasets/BabyARC` submodule, or directly downloaded via [this link](https://drive.google.com/drive/folders/1g0wNYb4JuwA1lcDxgv4yUDOyToszNYmQ?usp=share_link).  If download from the above link, put the downloaded data ("*.p" files) under the `./datasets/data/` folder.

# Structure
Here we detail the structure of this repo. The ".ipynb" files are equivalent to the corresponding ".py" version, where the ".ipynb" are easy to perform tests, while ".py" are suitable for long-running experiments. The .py are transformed from the corresponding ".ipynb" files via `jupyter nbconvert --to python {*.ipynb}`. Here we detail the repo's structure:
* The [concept_library](https://github.com/tailintalent/concept_library) contains the concept class definitions and EBM model architectures.
* The [train.py](https://github.com/snap-stanford/zeroc/blob/master/train.py) (or the corresponding "train.ipynb" file) is the script for training elementary concepts and relations.
* The [inference.ipynb](https://github.com/snap-stanford/zeroc/blob/master/train.ipynb) contains commands to perform zero-shot concept recognition and aquisition.
* The [concept_transfer.py](https://github.com/snap-stanford/zeroc/blob/master/concept_transfer.py) and [concepts_shapes.py](https://github.com/snap-stanford/zeroc/blob/master/concepts_shapes.py) contain helper functions and classes for training.
* The [datasets/data/](https://github.com/snap-stanford/zeroc/blob/master/datasets/data) folder contains the datasets used for training. The [datasets/babyarc_datasets.ipynb](https://github.com/snap-stanford/zeroc/blob/master/datasets/babyarc_datasets.ipynb) contains demonstrations about how to use the BabyARC engine to generate complex grid-world based reasoning tasks.
* The results are saved under [results/](https://github.com/snap-stanford/zeroc/blob/master/results) folder,



# Training:

Here we provide example commands for training energy-based models (EBMs) for concepts and relations. Full training command can be found under [results/README.md](https://github.com/snap-stanford/zeroc/blob/master/results/README.md). The results are saved under `./results/{--exp_id}_{--date_time}/`, as a pickle "*.p" file.

Training concepts for HD-Letter dataset (the `--dataset` argument specifies the dataset):
```code
python train.py --dataset=c-Line --exp_id=zeroc --date_time=11-24 --inspect_interval=5 --save_interval=5 --color_avail=1,2 --n_examples=40000 --max_n_distractors=2 --canvas_size=16 --train_mode=cd --model_type=CEBM --mask_mode=concat --c_repr_mode=c2 --c_repr_first=2 --kl_coef=1 --entropy_coef_mask=0 --entropy_coef_repr=0 --ebm_target=mask --ebm_target_mode=r-rmbx --is_pos_repr_learnable=False --sample_step=60 --step_size=20 --step_size_repr=2 --lambd_start=0.1 --p_buffer=0.2 --epochs=500 --channel_base=128 --two_branch_mode=concat --neg_mode=addrand+delrand --aggr_mode=max --neg_mode_coef=0.05 --epochs=500 --early_stopping_patience=-1 --n_workers=0 --seed=1 --self_attn_mode=None --transforms=color+flip+rotate+resize:0.5 --act_name=leakyrelu --pos_consistency_coef=0.1 --energy_mode=standard+center^stop:0.1 --emp_target_mode=r-mb --gpuid=0
```

Training relation for HD-Letter dataset (the "+" in the "--dataset" means the relation/concept are randomly sampled from the specified relations/concepts):
```code
python train.py --dataset=c-Parallel+VerticalMid+VerticalEdge --exp_id=zeroc --date_time=11-24 --inspect_interval=5 --save_interval=5 --color_avail=1,2 --n_examples=40000 --max_n_distractors=3 --canvas_size=16 --model_type=CEBM --mask_mode=concat --c_repr_mode=c2 --c_repr_first=2 --kl_coef=1 --entropy_coef_mask=0 --entropy_coef_repr=0 --ebm_target=mask --ebm_target_mode=r-rmbx --is_pos_repr_learnable=False --sample_step=60 --step_size=20 --step_size_repr=2 --lambd_start=0.1 --p_buffer=0.2 --epochs=500 --channel_base=128 --two_branch_mode=concat --neg_mode=addrand+permlabel --aggr_mode=max --neg_mode_coef=0.2 --epochs=500 --early_stopping_patience=-1 --n_workers=0 --seed=1 --self_attn_mode=None --transforms=color+flip+rotate+resize:0.5 --act_name=leakyrelu --pos_consistency_coef=1 --gpuid=0
```

Training concepts for HD-Concept dataset (the "Rect[4,16]" in the --dataset argument means that the size of "Rect" is within [4,16]):
```code
python train.py --dataset=c-Rect[4,16]+Eshape[3,10] --exp_id=zeroc --date_time=11-24 --inspect_interval=2 --save_interval=2 --color_avail=1,2 --n_examples=40000 --max_n_distractors=2 --canvas_size=20 --train_mode=cd --model_type=CEBM --mask_mode=concat --c_repr_mode=c2 --c_repr_first=2 --kl_coef=1 --entropy_coef_mask=0 --entropy_coef_repr=0 --ebm_target=mask --ebm_target_mode=r-rmbx --is_pos_repr_learnable=False --sample_step=60 --step_size=20 --step_size_repr=2 --lambd_start=0.1 --p_buffer=0.2 --epochs=200 --channel_base=128 --two_branch_mode=concat --neg_mode=addrand+delrand+permlabel --aggr_mode=max --neg_mode_coef=0.05 --epochs=500 --early_stopping_patience=-1 --n_workers=0 --seed=1 --self_attn_mode=None --transforms=color+flip+rotate+resize:0.5 --act_name=leakyrelu --pos_consistency_coef=0.1 --energy_mode=standard+center^stop:0.005 --emp_target_mode=r-mb --gpuid=0
```

Training relation for HD-Concept dataset
```code
python train.py --dataset=c-IsNonOverlapXY+IsInside+IsEnclosed\(Rect[4,16]+Randshape[3,8]+Lshape[3,10]+Tshape[3,10]\) --exp_id=zeroc --date_time=11-24 --inspect_interval=5 --save_interval=5  --color_avail=1,2 --n_examples=40000 --max_n_distractors=1 --canvas_size=20 --model_type=CEBM --mask_mode=concat --c_repr_mode=c2 --c_repr_first=2 --kl_coef=1 --entropy_coef_mask=0 --entropy_coef_repr=0 --ebm_target=mask --ebm_target_mode=r-rmbx --is_pos_repr_learnable=False --sample_step=60 --step_size=20 --step_size_repr=2 --lambd_start=0.1 --p_buffer=0.2 --epochs=300 --channel_base=128 --two_branch_mode=concat --neg_mode=addrand+permlabel --aggr_mode=max --neg_mode_coef=0.2 --epochs=500 --early_stopping_patience=-1 --n_workers=0 --seed=1 --self_attn_mode=None --transforms=color+flip+rotate+resize:0.5 --act_name=leakyrelu --pos_consistency_coef=1 --gpuid=0
```

Training 3D concepts (for experiments in the 2D->3D transfer):
```code
python train.py --dataset=y-Line --exp_id=zeroc --date_time=11-24 --inspect_interval=5 --save_interval=5 --color_avail=1,2 --n_examples=40000 --max_n_distractors=2 --canvas_size=16 --train_mode=cd --model_type=CEBM --mask_mode=concat --c_repr_mode=c2 --c_repr_first=2 --kl_coef=1 --entropy_coef_mask=0 --entropy_coef_repr=0 --ebm_target=mask --ebm_target_mode=r-rmbx --is_pos_repr_learnable=False --sample_step=60 --step_size=30 --step_size_repr=2 --lambd_start=0.1 --p_buffer=0.2 --channel_base=128 --two_branch_mode=concat --neg_mode=addallrand+delrand+permlabel --aggr_mode=max --neg_mode_coef=0.2 --epochs=500 --early_stopping_patience=-1 --n_workers=0 --seed=1 --self_attn_mode=None --transforms=color+flip+rotate+resize:0.5 --pos_consistency_coef=0.1 --is_res=True --lr=1e-4 --id=0 --act_name=leakyrelu --rescaled_size=32,32 --energy_mode=standard+center^stop:0.1 --emp_target_mode=r-mb --color_map_3d=same --seed_3d=42 --batch_size=128 --gpuid=0
```

Training 3D relations:
```code
python train.py --dataset=y-Parallel+VerticalMid+VerticalEdge --exp_id=zeroc --date_time=11-24 --inspect_interval=5 --save_interval=5 --color_avail=1,2 --n_examples=40000 --max_n_distractors=3 --canvas_size=16 --train_mode=cd --model_type=CEBM --mask_mode=concat --c_repr_mode=c2 --c_repr_first=2 --kl_coef=2 --entropy_coef_mask=0 --entropy_coef_repr=0 --ebm_target=mask --ebm_target_mode=r-rmbx --is_pos_repr_learnable=False --sample_step=60 --step_size=30 --step_size_repr=2 --lambd_start=0.1 --p_buffer=0.2 --channel_base=128 --two_branch_mode=concat --neg_mode=permlabel --aggr_mode=max --neg_mode_coef=0.2 --epochs=300 --early_stopping_patience=-1 --n_workers=0 --seed=3 --self_attn_mode=None --transforms=color+flip+rotate+resize:0.5 --pos_consistency_coef=1 --is_res=True --lr=1e-4 --id=0 --use_seed_2d=False --act_name=leakyrelu --rescaled_size=32,32 --color_map_3d=same --seed_3d=42 --batch_size=128 --gpuid=0
```

# Inference
The zero-shot concept recognition and aquisition experiments are provided in [inference_zero_shot.ipynb](https://github.com/snap-stanford/zeroc/blob/master/inference.ipynb).

# Citation
If you find our work and/or our code useful, please cite us via:

```bibtex
@inproceedings{wu2022zeroc,
title={ZeroC: A neuro-symbolic model for zero-shot concept recognition and acquisition at inference time},
author={Wu, Tailin and Tjandrasuwita, Megan and Wu, Zhengxuan and Yang, Xuelin and Liu, Kevin and Sosi{\v{c}}, Rok and Leskovec, Jure},
booktitle={Neural Information Processing Systems},
year={2022},
}
```



