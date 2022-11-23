## ZeroC: A Neuro-Symbolic Model for Zero-shot Concept Recognition and Acquisition at Inference Time

[Paper](https://arxiv.org/abs/2206.15049) | [Slides](https://docs.google.com/presentation/d/1WAR4dZ0J2E-u3V_DgYBYTF4mDCmRk0FXI8GPlM2kqdQ/edit?usp=sharing) | [Project page](https://snap.stanford.edu/zeroc/)

This is the official repo for [ZeroC: A Neuro-Symbolic Model for Zero-shot Concept Recognition and Acquisition at Inference Time](https://arxiv.org/abs/2206.15049) (Wu et al. NeurIPS 2022). ZeroC is a neuro-symbolic architecture that after pretrained with simpler concepts and relations, can recognize and acquire novel, more complex concepts in a zero-shot way. 

<a href="url"><img src="https://github.com/snap-stanford/zeroc/blob/master/assets/hierarchy.png" align="center" width="600" ></a>

ZeroC represents concepts as graphs of constituent concept models (as nodes) and their relations (as edges). We design ZeroC architecture so that it allows a **one-to-one mapping** between an abstract, symbolic graph structure of a concept and its corresponding energy-based model that provides a probability model for concepts and relations (by summing over constituent concept and relation EBMs according to the graph, see figure above). After pre-trained with simpler concepts and relations, at inference time, without further training, ZeroC can perform:

* **Zero-shot concept recognition**: recognize a more complex, hierarchical concept from image, given the symbolic graph structure of this hierarchical concept.

* **Zero-shot concept aquisition**: given a single demonstration of an image containing a hierarchical concept, infer its graph structure. Furthermore, this graph structure can be transferred across domains (e.g., from 2D image to 3D image), allowing the ZeroC in the second domain to directly recognize such hierarchical concept (see figure below):

<a href="url"><img src="https://github.com/snap-stanford/zeroc/blob/master/assets/2d3d_zeroc.png" align="center" width="500" ></a>

# Installation

1. First clone the directory. Then run the following command to initialize the submodules:

```code
git submodule init; git submodule update
```
(If showing error of no permission, need to first [add a new SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).)

2. Install dependencies.

Create a new environment using conda. Install pytorch (version >= 1.10.1). Then install other dependencies via:
```code
pip install -r requirements.txt
```

# Dataset:
The dataset files can be downloaded via [this link](https://drive.google.com/drive/folders/1g0wNYb4JuwA1lcDxgv4yUDOyToszNYmQ?usp=share_link). Download it to ./datasets/data/.


# Training:

Here we provide example commands for training concepts and relations. Full training command can be found [here](https://github.com/snap-stanford/zeroc/blob/master/results/README.md). The results are saved under `./results/{--exp_id}_{--date_time}/`.

Training concept (the `--dataset` specifies the dataset):
```code
python train.py --exp_id=ebm --date_time=10-24 --inspect_interval=5 --save_interval=5 --dataset=c-Line --color_avail=1,2 --n_examples=40000 --max_n_distractors=2 --canvas_size=16 --train_mode=cd --model_type=CEBM --mask_mode=concat --c_repr_mode=c2 --c_repr_first=2 --kl_coef=1 --entropy_coef_mask=0 --entropy_coef_repr=0 --ebm_target=mask --ebm_target_mode=r-rmbx --is_pos_repr_learnable=False --sample_step=60 --step_size=20 --step_size_repr=2 --lambd_start=0.1 --p_buffer=0.2 --epochs=500 --channel_base=128 --two_branch_mode=concat --neg_mode=addrand+delrand --aggr_mode=max --neg_mode_coef=0.05 --epochs=500 --early_stopping_patience=-1 --n_workers=0 --seed=1 --self_attn_mode=None --transforms=color+flip+rotate+resize:0.5 --act_name=leakyrelu --pos_consistency_coef=0.1 --energy_mode=standard+center^stop:0.1 --emp_target_mode=r-mb --gpuid=4 --id=0
```

Training relation:
```code
python train.py --exp_id=ebm --date_time=1-21 --inspect_interval=5 --save_interval=5 --dataset=c-Parallel+VerticalMid+VerticalEdge --color_avail=1,2 --n_examples=40000 --max_n_distractors=3 --canvas_size=16 --model_type=CEBM --mask_mode=concat --c_repr_mode=c2 --c_repr_first=2 --kl_coef=1 --entropy_coef_mask=0 --entropy_coef_repr=0 --ebm_target=mask --ebm_target_mode=r-rmbx --is_pos_repr_learnable=False --sample_step=60 --step_size=20 --step_size_repr=2 --lambd_start=0.1 --p_buffer=0.2 --epochs=500 --channel_base=128 --two_branch_mode=concat --neg_mode=addrand+permlabel --aggr_mode=max --neg_mode_coef=0.2 --epochs=500 --early_stopping_patience=-1 --n_workers=0 --seed=1 --self_attn_mode=None --transforms=color+flip+rotate+resize:0.5 --act_name=leakyrelu --pos_consistency_coef=1 --gpuid=0 --id=new3
```

# Citation
If you find our work and/or our code useful, please cite us via:

```bibtex
@inproceedings{wu2022zeroc,
title={Zeroc: A neuro-symbolic model for zero-shot concept recognition and acquisition at inference time},
author={Wu, Tailin and Tjandrasuwita, Megan and Wu, Zhengxuan and Yang, Xuelin and Liu, Kevin and Sosi{\v{c}}, Rok and Leskovec, Jure},
booktitle={Neural Information Processing Systems},
year={2022},
}
```



