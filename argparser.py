"""Arguments for EBM scripts: concept_energy and concept_energy_composite."""
import argparse
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..', '..'))
from zeroc.concept_library.util import str2bool


def get_args_EBM():
    parser = argparse.ArgumentParser(description='Concept argparse.')
    # Experiment management:
    parser.add_argument('--exp_id', type=str,
                        help='Experiment id')
    parser.add_argument('--date_time', type=str,
                        help='date and time')
    parser.add_argument(
        '--exp_name', default="None", 
        help='If not "None", will use asynchronous training, and the data_record of'
             'training will be saved under f"{exp_id}_{date_time}/{exp_name}/{filename}".')
    parser.add_argument('--inspect_interval', type=int,
                        help='Interval for inspecting and plotting.')
    parser.add_argument('--save_interval', type=int,
                        help='Interval for saving the model_dict.')
    parser.add_argument('--verbose', type=int,
                        help='verbose.')
    parser.add_argument('--seed', type=int,
                        help='seed')
    parser.add_argument('--gpuid', type=str,
                        help='gpu id.')
    parser.add_argument('--id', type=str,
                        help='id.')
    parser.add_argument(
        '--recent_record', type=int, default=-1, help='Number of most recent entries to keep in the data record. If -1, keeps all entries.')
    
    # Dataset:
    parser.add_argument('--dataset', type=str,
                        help='dataset name. Choose from "cifar10", "concept-{*}" and "arc-{*}"')
    parser.add_argument('--n_examples', type=int,
                        help='Number of examples.')
    parser.add_argument('--n_queries_per_class', type=int,
                        help='If generating fewshot, the number of queries per class.')
    parser.add_argument('--canvas_size', type=int,
                        help='Size of the canvas for concept dataset.')
    parser.add_argument('--rainbow_prob', type=float,
                        help='Probability of using rainbow color in BabyARC.')
    parser.add_argument('--max_n_distractors', type=int, default=-1,
                        help='Number of distractors in BabyARC. If set to -1, it will follow the default behavior.')
    parser.add_argument('--min_n_distractors', type=int, default=0,
                        help='Minimum number of distractors in BabyARC.')
    parser.add_argument('--allow_connect', type=str2bool, nargs='?', const=True, default=True,
                        help='Whether or not to allow objects to connect in the image.')
    parser.add_argument('--is_rewrite', type=str2bool, nargs='?', const=True, default=False,
                        help='If True, will rewrite the dataset.')
    parser.add_argument('--max_num_occur', type=int, default=10,
                        help='Max number of concepts (or relations) in an example.')
    parser.add_argument('--n_operators', type=int,
                        help='Number of operators in BabyARC.')
    parser.add_argument('--color_avail', type=str,
                        help='Available color in BabyARC separated by , (e.g., 1,2,3, -1 means any color).')
    parser.add_argument('--to_RGB', type=str2bool, nargs='?', const=True, default=False,
                        help='If dataset is BabyARC, convert from 10-channels to RGB')
    parser.add_argument('--is_load', type=str2bool, nargs='?', const=True, default=True,
                        help='Whether or not to load dataset from file if it exists.')
    parser.add_argument('--rescaled_size', type=str,
                        help='If dataset is BabyARC, produce the new shape for the dataset. Choose from "None" (no resizing), e.g. "16,16" (resizing to (16,16)).')
    parser.add_argument('--rescaled_mode', type=str,
                        help='Choose from "nearest", "default".')

    # 2D-to-3D conversion:
    parser.add_argument('--seed_3d', type=int, default=42,
                        help='seed when converting BabyARC to 3D,')
    parser.add_argument('--use_seed_2d', type=str2bool, default=False,
                        help='Use "seed" argument to generate 2D examples that are converted to 3D (instead of "seed_3d")')
    parser.add_argument('--image_size_3d', type=int, nargs=2, default=[256, 256],
                       help='Size of 3D image.')
    parser.add_argument('--num_processes_3d', type=int, default=20,
                        help='Number of processes to use for conversion.')
    parser.add_argument('--color_map_3d', type=str, default="same",
                        help='If "random", will randomly assign a color per object. Otherwise, use the color dictionary.')
    parser.add_argument('--add_thick_surf', type=int, nargs=2, default=[0, 0.5],
                       help='Range of values in which to uniformly sample addition of thickness in xy plane.')
    parser.add_argument('--add_thick_depth', type=int, nargs=2, default=[0, 0.5],
                       help='Range of values in which to uniformly sample addition of thickness in z dimension.')
    
    # Data augmentation
    parser.add_argument('--transforms', type=str,
                       help='Data augmentations to perform on initial negative samples from replay buffer. Example: "color+flip+rotate+resize" or "color+flip+rotate+resize", where the 0.5 is the probability of doing the transformation (default prob. of 1)')
    parser.add_argument('--transforms_pos', type=str,
                       help='Data augmentations to perform on initial positive samples from replay buffer. Example: "color+flip+rotate+resize:0.5" or "color+flip+rotate+resize", where the 0.5 is the probability of doing the transformation (default prob. of 1)')

    # Model:
    parser.add_argument('--model_type', type=str,
                        help='Model type. Choose from "CEBM", "GraphEBM", "IGEBM".')
    parser.add_argument('--w_type', type=str,
                        help='type of the first two arities of input. choose from "image", "mask", "image+mask", "obj", "image+obj"')
    parser.add_argument('--mask_mode', type=str,
                        help='mask_mode. Choose from "concat", "mulcat", "mul".')
    parser.add_argument('--channel_base', type=int,
                        help='Base n_channels for "CEBM".')
    parser.add_argument('--two_branch_mode', type=str,
                        help='Mode for the two branches of CEBM, if its mode is "operator". Choose from "concat", "imbal-{#indi-layers}".')
    parser.add_argument('--is_spec_norm', type=str,
                        help='If "True", each CNN block will have spectral norm. Choose from "True", "False", "ws" (with normalization).')
    parser.add_argument(
        '--is_res', type=str2bool, nargs='?', const=True, default=True, help='If True, will use residual layer for CResBlock.')

    parser.add_argument('--c_repr_mode', type=str,
                        help='How c_repr will be combined with the input. Choose from "None", l1", "l2", "c1", "c2", "c3".')
    parser.add_argument('--c_repr_first', type=int,
                        help='First block to pass in c_repr.')
    parser.add_argument('--c_repr_base', type=int,
                        help='Number of base channels for c_repr.')
    parser.add_argument('--z_mode', type=str,
                        help='How z will be combined with the input. Choose from "None", "c0", "c1", "c2", "c3".')
    parser.add_argument('--z_first', type=int,
                        help='First block to pass in z.')
    parser.add_argument('--z_dim', type=int,
                        help='Dimension for z.')
    parser.add_argument('--pos_embed_mode', type=str,
                        help='Whether or how to embed position. Choose from "None", "implicit", "sine", "learned".')
    parser.add_argument('--aggr_mode', type=str,
                        help='Aggregation mode for the last layer.')
    parser.add_argument('--act_name', type=str,
                        help='Activation name')
    parser.add_argument('--normalization_type', type=str,
                        help='Normalization type.')
    parser.add_argument('--dropout', type=float,
                        help='Dropout. If greater than 0, will have dropout for the CResBlock.')
    parser.add_argument('--self_attn_mode', type=str,
                        help='Choose from "None", "pixel".')
    parser.add_argument('--last_act_name', type=str,
                        help='Activation for last layer of ConceptEBM.')
    parser.add_argument('--n_avg_pool', type=int,
                        help='Number of average pooling for ConceptEBM at the beginning.')

    # Specific for EBM_composite:
    parser.add_argument('--cumu_mode', type=str,
                        help='cumu_mode for concept_energy_composite, for computing the loss that combines multiple solutions for the same task. Choose from "harmonic", "gm-{order}" (generalized-mean with specified order), "mean", "geometric", "sum".')
    parser.add_argument('--update_ebm_dict_interval', type=int,
                        help='Every {update_ebm_dict_interval} epochs, update the ebm_dict.'
                       )
    parser.add_argument('--min_n_tasks', type=int,
                        help='Wait until the number of tasks is above {args.min_n_tasks} in task_dict.p')
    parser.add_argument('--is_save', type=str2bool, nargs='?', const=True, default=True,
                        help='If True, will write to the ebm_dict.p and data_record for EBM_composite.')
    parser.add_argument('--train_coef', type=float,
                        help='train_coef.')
    parser.add_argument('--test_coef', type=float,
                        help='train_coef.')
    parser.add_argument('--mutual_exclusive_coef', type=float,
                        help='Coefficient for mutual-exclusive energy during composite training. Penalizes when two masks from multiple EBMs overlap in an image.')
    parser.add_argument('--obj_coef', type=float,
                        help='Coefficient for regularization to encourage each EBM to discover individual objects.')
    parser.add_argument('--channel_coef', type=float,
                        help='Coefficient for the main channel (1:10th channel) for the ARC/BabyARC tasks and all 3 channels for RGB images.')
    parser.add_argument('--empty_coef', type=float,
                        help='Coefficient for the empty channel (0th channel) for the ARC/BabyARC tasks.')
    parser.add_argument('--pixel_entropy_coef', type=float,
                        help='Coefficient for pixel-wise entropy.')
    parser.add_argument('--pixel_gm_coef', type=float,
                        help='Coefficient for pixel-wise generalize-mean distance w.r.t. 0 and 1.')
    parser.add_argument(
        '--iou_batch_consistency_coef', type=float, help='Encouraging consistency for distance of two masks across examples.')
    parser.add_argument(
        '--iou_attract_coef', type=float, help='Encouraging masks that are near to be nearer.')
    parser.add_argument(
        '--iou_concept_repel_coef', type=float, help='Repel masks that belong to different concepts that occupies one object slot.')
    parser.add_argument(
        '--iou_relation_repel_coef', type=float, help='Repel masks that belong to the same relation.')
    parser.add_argument(
        '--iou_relation_overlap_coef', type=float, help='Repel masks that belong to the same relation.')
    parser.add_argument(
        '--iou_target_matching_coef', type=float, help='Coefficient for relation tasks that if the IoU between one one discovered mask and the target mask is greater than 0.5, will further encourage it to be nearer.')
    parser.add_argument(
        '--connected_coef', type=float, help='Encourage each mask to be a single connected component.')
    parser.add_argument(
        '--connected_num_samples', type=int, help='Number of pairs of points to sample when computing connected loss.')
    # Specific for EBM + GNN:
    parser.add_argument(
        '--target_loss_type', type=str, help='Loss_type for ebm supervised learning. Choose from any valid loss_type. E.g. "mse", "Jaccard".')
    parser.add_argument(
        '--is_selector_gnn', type=str2bool, nargs='?', const=True, default=False, help='If True, will have GNN for the selector.')
    parser.add_argument(
        '--is_zgnn_node', type=str2bool, nargs='?', const=True, default=False, help='If True, have zgnn_node for the GNN (zgnn is a tuple of (zgnn_node, zgnn_edge). If is_zgnn_node is False, zgnn_node will be None). If False, will use forward_NN.')
    parser.add_argument(
        '--is_cross_validation', type=str2bool, nargs='?', const=True, default=True, help='If True, use cross-validation within a task.')
    parser.add_argument(
        '--load_pretrained_concepts', type=str, help='If not "None", will be a string including the dirname + filename for the data_record that contains the concept_model.')
    
    parser.add_argument(
        '--n_GN_layers', type=int, help='Number of GN layers.')
    parser.add_argument(
        '--gnn_normalization_type', type=str, help='Normalization_type for GNN.')
    parser.add_argument(
        '--gnn_pooling_dim', type=int, help='Pooling dimension for GNN.')
    parser.add_argument(
        '--edge_attr_size', type=int, help='Size of edge_attr.')
    parser.add_argument(
        '--cnn_output_size', type=int, help='CNN output_size.')
    parser.add_argument(
        '--cnn_is_spec_norm', type=str, help='If True, will have spectral norm for CNN inside GNN. Choose from "True", "False", "ws".')
    


    # Only affective for standalone mode of EBM_composite:
    parser.add_argument(
        '--is_ebm_share_param', type=str2bool, nargs='?', const=True, default=False, help='Whether or not to share parameter for different EBMs of the same EBM mode.')
    parser.add_argument('--T_id', type=str, help='T_id for the task for standalone mode. Examples'
                        'Tuc6: 6 random concepts; Tuc6r3: 6 random concepts and 3 random relations; Tuc6r3o2: 6 random concepts, 3 random relations and 2 operators.')
    parser.add_argument(
        '--image_value_range', type=str, help='Minimum and maximum value for the values of the image at each pixel. For BabyARC/ARC, use "0,1", for CLEVR, use "-1,1".')
    parser.add_argument(
        '--w_init_type', type=str, default='random', help='How to initialize w. Choose from "input", "random", "input-mask", "input-gaus", "k-means", "k-means^x" where x is the number of clusters')
    parser.add_argument(
        '--indiv_sample', type=int, default=-1, help='Number of sample steps for each EBM in selector when reconstructing image. If -1, do SGLD with all EBMs')
    parser.add_argument(
        '--n_tasks', type=int, help='Number of tasks.')
    parser.add_argument(
        '--is_concat_minibatch', type=str2bool, nargs='?', const=True, default=False, help='If True, will concatenate the tasks in a minibatch into a single tensor.')
    # Specific to relation-EBM:
    parser.add_argument(
        '--relation_merge_mode', type=str, help='How to merge graphs for relation graph discovery. Choose from "None", "threshold".')
    parser.add_argument(
        '--is_relation_z', type=str2bool, nargs='?', const=True, default=True, help='If True, will have z for relation-EBM and reconstruction on the 2nd SGLD.')

    # Specific for encouraging selector discovery:
    parser.add_argument(
        '--SGLD_is_anneal', type=str2bool, nargs='?', const=True, default=False, help='If True, will anneal the SGLD_ coefficients..')
    parser.add_argument(
        '--SGLD_anneal_power', type=float, help='Power to which annealing coefficient grows.')
    parser.add_argument(
        '--SGLD_is_penalize_lower', type=str, help='if True or "True", will penalize that the sum is less than 1. If "False" or False, will not. If "obj:0.001" e.g., will only penalize on the object locations (if n_channels==10), with coefficient of 0.001.')
    parser.add_argument(
        '--SGLD_iou_batch_consistency_coef', type=float, help='Encouraging consistency for distance of two masks across examples in SGLD.')
    parser.add_argument(
        '--SGLD_iou_attract_coef', type=float, help='Encouraging masks that are near to be nearer in SGLD.')
    parser.add_argument(
        '--SGLD_iou_concept_repel_coef', type=float, help='Repel masks that belong to different concepts that occupies one object slot in SGLD.')
    parser.add_argument(
        '--SGLD_iou_relation_repel_coef', type=float, help='Repel masks that belong to the same relation in SGLD.')
    parser.add_argument(
        '--SGLD_iou_relation_overlap_coef', type=float, help='Repel masks that belong to the same relation in SGLD.')
            
    # EBM training setting:
    parser.add_argument('--train_mode', type=str,
                        help='Training mode. Choose from "cd" (contrastive divergence) and "sl" (supervised learning).')
    parser.add_argument('--energy_mode', type=str,
                        help=' "standard:0.3": (E_pos - E_neg) * 0.3'
                        '"margin^0.2:0.3": max(0, 0.3 + E_pos - E_neg) * 0.2'
                        '"mid^0.2:0.3": (max(0, 0.2 + E_pos - E_empty) + max(0, 0.2 + E_empty - E_neg)) * 0.3'
                        '"mid^0.2^adapt:0.3": '
                            '(max(0, gamma + E_pos - E_empty) + max(0, gamma + E_empty - E_neg)) * 0.3'
                                'where gamma = max(0, StopGrad(E_neg - E_pos)/2) + 0.2'
                        '"standard:0.5+mid^0.2^adapt:0.3":'
                            '(E_pos - E_neg) * 0.5 + (max(0, gamma + E_pos - E_empty) + max(0, gamma + E_empty - E_neg)) * 0.3,'
                                'where gamma = max(0, StopGrad(E_neg - E_pos)/2) + 0.2.'
                        '"standard+center^stop": (E_pos - E_neg) * 1 + ((E_pos+E_neg).detach()/2 - E_empty).abs()'
                            '"stop": stop gradient, and each empty loss is computed per example'
                            '"stopgen": similar to "stop", but the negative energy is the mean of neg_out and neg_out_gen, per example.'
                            '"stopmean": stop gradient, and each empty loss is computed per minibatch'
                            '"stopgenmean": similar to "stopmean", but the negative energy is the mean of neg_out and neg_out_gen.')
    parser.add_argument('--supervised_loss_type', type=str,
                        help='Loss_type for ebm supervised learning. Choose from any valid loss_type. E.g. "mse", "l1", "l2".')
    parser.add_argument('--kl_all_step', type=str2bool, nargs='?', const=True, default=False,
                        help='If True, will compute the 2nd order kl for all steps.')
    parser.add_argument('--kl_coef', type=float,
                        help='Coefficient for kl regularization.')
    parser.add_argument('--entropy_coef_img', type=float,
                        help='Coefficient for entropy for image.')
    parser.add_argument('--entropy_coef_mask', type=float,
                        help='Coefficient for entropy for mask.')
    parser.add_argument('--entropy_coef_repr', type=float,
                        help='Coefficient for entropy for repr.')
    parser.add_argument('--pos_consistency_coef', type=float,
                        help='Coefficient for positive consistency loss.')
    parser.add_argument('--neg_consistency_coef', type=float,
                        help='Coefficient for negative consistency loss.')
    parser.add_argument('--emp_consistency_coef', type=float,
                        help='Coefficient for empty consistency loss.')
    parser.add_argument('--SGLD_mutual_exclusive_coef', type=float,
                        help='Coefficient for mutual-exclusive energy during SGLD. Penalizes when two masks from multiple EBMs overlap in an image.')
    parser.add_argument('--SGLD_fine_mutual_exclusive_coef', type=float,
                        help='Coefficient for mutual-exclusive energy during SGLD. Penalizes when two masks from multiple EBMs overlap in an image.')
    parser.add_argument('--SGLD_object_exceed_coef', type=float,
                        help='Coefficient for penalizing objects exceeding the ground truth mask during SGLD. Prevents a mask from an EBM from exceeding ground-truth boundaries.')
    parser.add_argument('--SGLD_pixel_entropy_coef', type=float,
                        help='Coefficient for pixel-wise entropy during SGLD.')
    parser.add_argument('--SGLD_mask_entropy_coef', type=float,
                        help='Coefficient for mask-level entropy during SGLD.')
    parser.add_argument('--SGLD_pixel_gm_coef', type=float,
                        help='Coefficient for pixel-wise generalize-mean distance w.r.t. 0 and 1, during SGLD')
    parser.add_argument('--epsilon_ent', type=float,
                        help='epsilon for adding to the entropy compuation to prevent Inf.')
    parser.add_argument('--ebm_target_mode', type=str,
                        help='Target input to perform SGD on. Choose from "None", "r-{}" where {} choose from subset of "r", "m", "b", "x".')
    parser.add_argument('--emp_target_mode', type=str,
                        help='Set of ebm_target mode in which the emp_out will participate in the loss. Choose from "all", "r-{}" where {} choose from subset of "r", "m", "b", "x".')
    parser.add_argument('--ebm_target', type=str,
                        help='Target input to perform SGD on. Choose from "mask", "mask+repr", "repr".')
    parser.add_argument('--is_pos_repr_learnable', type=str2bool, nargs='?', const=True, default=False,
                        help='Whether the positive concept_embeddings are learnable.')
    parser.add_argument('--neg_mode', type=str,
                        help='Modes for generated negative masks from pos images and pos_masks. Only valid when is_mask is True.')
    parser.add_argument('--neg_mode_coef', type=float,
                        help='Coefficient for negative mode. Only when it is > 0 and neg_mode is not "None" will neg_mode have effect.')
    parser.add_argument('--alpha', type=float,
                        help='Coefficient for the L2 loss.')
    parser.add_argument('--lambd_start', type=float,
                        help='Starting lambda for Gaussian Distribution.')
    parser.add_argument('--lambd', type=float,
                        help='Lambda for Gaussian Distribution.')
    parser.add_argument('--step_size_start', type=float,
                        help='Starting step size for sampling.')
    parser.add_argument('--step_size', type=float,
                        help='Step size for sampling.')
    parser.add_argument('--step_size_repr', type=float,
                        help='Step size for sampling c_repr.')
    parser.add_argument('--step_size_img', type=float,
                        help='Step size for sampling img.')
    parser.add_argument('--step_size_z', type=float,
                        help='Step size for sampling z.')
    parser.add_argument('--step_size_zgnn', type=float,
                        help='Step size for sampling zgnn.')
    parser.add_argument('--step_size_wtarget', type=float,
                        help='Step size for sampling wtarget.')
    parser.add_argument('--sample_step', type=int,
                        help='Number of steps for sampling.')
    parser.add_argument('--p_buffer', type=float,
                        help='Probability for using samples inside the buffer, as compared to using Gaussian.')
    parser.add_argument('--lr', type=float,
                        help='Learning rate.')
    parser.add_argument('--lr_pretrained_concepts', type=float,
                        help='Learning rate for pretrained concepts.')
    parser.add_argument('--parallel_mode', type=str,
                        help='Parallel mode. Choose from "None", "dp" (DataParallel) and "ddp" (DistributedDataParallel).')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size.')
    parser.add_argument('--epochs', type=int,
                        help='Number of epochs.')
    parser.add_argument('--early_stopping_patience', type=int,
                        help='Patience for early-stopping.')
    parser.add_argument('--n_workers', type=int,
                        help='Number of workers.')

    parser.set_defaults(
        # Exp management:
        exp_id="ebm",
        date_time="3-20",
        inspect_interval=5,
        save_interval=10,
        verbose=1,
        seed=-1,
        gpuid="3",
        id="1",

        # Dataset:
        dataset="c-line",
        n_examples=10000,
        n_queries_per_class=15,
        canvas_size=8,
        rainbow_prob=0.,
        max_n_distractors=0,
        min_n_distractors=0,
        allow_connect=True,
        is_rewrite=False,
        n_operators=1,
        color_avail="-1",
        transforms="None",
        transforms_pos="None",
        rescaled_size="None",
        rescale_mode="nearest",

        # Model:
        model_type="CEBM",
        w_type="image+mask",
        mask_mode="mul",
        channel_base=128,
        two_branch_mode="concat",
        is_spec_norm="True",
        is_res=True,
        c_repr_mode="c2",
        c_repr_first=2,
        c_repr_base=2,
        z_mode="None",
        z_first=2,
        z_dim=4,
        pos_embed_mode="None",
        aggr_mode="max",
        act_name="leakyrelu0.2",
        normalization_type="None",
        dropout=0,
        self_attn_mode="None",
        last_act_name="None",
        n_avg_pool=0,

        # Specific for EBM_composite:
        cumu_mode="harmonic",
        update_ebm_dict_interval=1,
        min_n_tasks=0,
        is_save=True,
        channel_coef=1.,
        empty_coef=0.02,
        obj_coef=0.1,
        mutual_exclusive_coef=0.1,
        pixel_entropy_coef=0.,
        pixel_gm_coef=0.,
        iou_batch_consistency_coef=0.,
        iou_concept_repel_coef=0.,
        iou_relation_repel_coef=0.,
        iou_relation_overlap_coef=0.,
        iou_attract_coef=0,
        iou_target_matching_coef=0,
        connected_coef=0,
        connected_num_samples=2,
        image_value_range='0,1',

        # Only valid for standalone EBM_composite:
        is_ebm_share_param=False,
        n_tasks=128,
        T_id="Tuc6",
        is_concat_minibatch=False,
        # Specific to relation-EBM:
        relation_merge_mode="None",
        is_relation_z=True,
        is_cross_validation=False,
        load_pretrained_concepts="None",

        # Specific for EBM + GNN:
        is_selector_gnn=False,
        is_zgnn_node=False,
        n_GN_layers=2,
        edge_attr_size=8,
        gnn_normalization_type="None",
        gnn_pooling_dim=16,
        cnn_output_size=32,
        cnn_is_spec_norm="True",
        train_coef=1,
        test_coef=1,

        # EBM training setting:
        train_mode="cd",
        energy_mode="standard",
        supervised_loss_type="mse",
        target_loss_type="mse",
        kl_all_step=False,
        kl_coef=0.,
        entropy_coef_img=0.,
        entropy_coef_mask=0.,
        entropy_coef_repr=0.,
        pos_consistency_coef=0.,
        neg_consistency_coef=0.,
        emp_consistency_coef=0.,
        # SGLD:
        SGLD_is_anneal=False,
        SGLD_anneal_power=2.0,
        SGLD_is_penalize_lower="True",
        SGLD_mutual_exclusive_coef=0.,
        SGLD_fine_mutual_exclusive_coef=0.,
        SGLD_object_exceed_coef=0.,
        SGLD_pixel_entropy_coef=0.,
        SGLD_mask_entropy_coef=0.,
        SGLD_pixel_gm_coef=0.,
        # For selector discovery:
        SGLD_iou_batch_consistency_coef=0.,
        SGLD_iou_concept_repel_coef=0.,
        SGLD_iou_relation_repel_coef=0.,
        SGLD_iou_relation_overlap_coef=0.,
        SGLD_iou_attract_coef=0,
        # Other settings:
        epsilon_ent=1e-5,
        ebm_target_mode="None",
        ebm_target="mask",
        emp_target_mode="all",
        is_pos_repr_learnable=False,
        neg_mode="None",
        neg_mode_coef=0.,
        alpha=1,
        lambd_start=-1,  # best: 0.1
        lambd=0.005,
        step_size_start=-1,
        step_size=20,
        step_size_img=-1,
        step_size_repr=-1,
        step_size_z=2,
        step_size_zgnn=2,
        step_size_wtarget=-1,
        sample_step=60,
        p_buffer=0.95,  # best: 0.2
        lr=1e-4,
        lr_pretrained_concepts=0,
        parallel_mode="None",
        batch_size=128,
        epochs=500,
        early_stopping_patience=-1,
        n_workers=4,
    )
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
        args = parser.parse_args([])
    except:
        args = parser.parse_args()
    if args.step_size_img == -1:
        args.step_size_img = args.step_size
    if args.step_size_repr == -1:
        args.step_size_repr = args.step_size
    if args.step_size_z == -1:
        args.step_size_z = args.step_size
    if args.step_size_zgnn == -1:
        args.step_size_zgnn = args.step_size
    if args.step_size_wtarget == -1:
        args.step_size_wtarget = args.step_size
    return args


def update_default_hyperparam(Dict):
    """Default hyperparameters for previous experiments, after adding these new options."""
    default_param = {
        "is_two_branch": False,
        "two_branch_mode": "concat",
        "rainbow_prob": 0,
        "max_n_distractors": 0,
        "min_n_distractors": 0,
        "allow_connect": True,
        "n_operators" : 1,
        "color_avail" : "-1",
        "transforms": "None",
        "transforms_pos": "None",
        # Training:
        "ebm_target_mode": "None",
        "ebm_target": "mask",
        "emp_target_mode": "all",
        "is_pos_repr_learnable": False,
        "p_buffer": 0.95,
        "lambd_start": -1,
        "lambd": 0.005,
        "neg_mode": "None",
        "neg_mode_coef": 0.,
        "early_stopping_patience": -1,
        "step_size_start": -1,
        "step_size_img": -1,
        "step_size_repr": -1,
        "step_size_z": 2,
        "step_size_zgnn": 2,
        "step_size_wtarget": -1,
        "is_spec_norm": "True",
        "is_res": True,
        "c_repr_mode": "l1",
        "c_repr_first": 0,
        "c_repr_base": 2,
        "aggr_mode": "sum",
        "act_name": "leakyrelu0.2",
        "normalization_type": "None",
        "dropout": 0,
        "self_attn_mode": "None",
        "last_act_name": "None",
        "n_avg_pool": 0,
        "kl_all_step": False,
        "kl_coef": 0.,
        "entropy_coef_img": 0.,
        "entropy_coef_mask": 0.,
        "entropy_coef_repr": 0.,
        "epsilon_ent": 1e-5,
        "pos_consistency_coef": 0.,
        "neg_consistency_coef": 0.,
        "emp_consistency_coef": 0.,
        # SGLD:
        "SGLD_is_anneal": False,
        "SGLD_anneal_power": 2.0,
        "SGLD_is_penalize_lower": "True",
        "SGLD_mutual_exclusive_coef": 0,
        "SGLD_fine_mutual_exclusive_coef": 0,
        "SGLD_object_exceed_coef": 0,
        "SGLD_pixel_entropy_coef": 0,
        "SGLD_mask_entropy_coef": 0,
        "SGLD_pixel_gm_coef": 0,
        # selector discovery:
        "SGLD_iou_batch_consistency_coef": 0,
        "SGLD_iou_concept_repel_coef": 0,
        "SGLD_iou_relation_repel_coef": 0,
        "SGLD_iou_relation_overlap_coef": 0,
        "SGLD_iou_attract_coef": 0,
        # Other settings:
        "w_type": "image+mask",
        "train_mode": "cd",
        "energy_mode": "standard",
        "supervised_loss_type": "mse",
        "target_loss_type": "mse",
        "cumu_mode": "harmonic",
        "channel_coef": 1,
        "empty_coef": 0.11,
        "obj_coef": 0,
        "mutual_exclusive_coef": 0,
        "pixel_entropy_coef": 0,
        "pixel_gm_coef": 0,
        "iou_batch_consistency_coef": 0,
        "iou_concept_repel_coef": 0,
        "iou_relation_repel_coef": 0,
        "iou_relation_overlap_coef": 0,
        "iou_attract_coef": 0,
        "iou_target_matching_coef": 0,
        "z_mode": "None",
        "z_first": 2,
        "z_dim": 4,
        "pos_embed_mode": "None",
        "image_value_range": "0,1",
        "w_init_type": "random",
        "indiv_sample": -1,
        "n_tasks": 128,
        "is_concat_minibatch": False,
        "to_RGB": False,
        "rescaled_size": "None",
        "rescale_mode": "nearest",
        "upsample": -1,
        "relation_merge_mode": "None",
        "is_relation_z": True,
        "connected_coef": 0,
        "connected_num_samples": 2,
        # Specific for EBM + GNN:
        "is_selector_gnn": False,
        "is_zgnn_node": False,
        "is_cross_validation": False,
        "load_pretrained_concepts": "None",
        "n_GN_layers": 2,
        "gnn_normalization_type": "None",
        "gnn_pooling_dim": 16,
        "edge_attr_size": 8,
        "cnn_output_size": 32,
        "cnn_is_spec_norm": "True",
        "train_coef": 1,
        "test_coef": 1,
        "lr_pretrained_concepts": 0,
        "parallel_mode": "None",
        "is_rewrite": False,
    }
    for key, item in default_param.items():
        if key not in Dict:
            Dict[key] = item
    return Dict


def get_SGLD_kwargs(args):
    kwargs = {}
    if isinstance(args, dict):
        args = init_args(args)
    if args.exp_name == "None":
        kwargs["lambd_start"] = args.lambd_start
        kwargs["lambd"] = args.lambd
        kwargs["SGLD_is_anneal"] = args.SGLD_is_anneal
        kwargs["SGLD_is_penalize_lower"] = args.SGLD_is_penalize_lower if hasattr(args, "SGLD_is_penalize_lower") else True
        kwargs["SGLD_mutual_exclusive_coef"] = args.SGLD_mutual_exclusive_coef
        kwargs["SGLD_pixel_entropy_coef"] = args.SGLD_pixel_entropy_coef
        kwargs["SGLD_pixel_gm_coef"] = args.SGLD_pixel_gm_coef
        kwargs["SGLD_iou_batch_consistency_coef"] = args.SGLD_iou_batch_consistency_coef
        kwargs["SGLD_iou_concept_repel_coef"] = args.SGLD_iou_concept_repel_coef
        kwargs["SGLD_iou_relation_repel_coef"] = args.SGLD_iou_relation_repel_coef
        kwargs["SGLD_iou_relation_overlap_coef"] = args.SGLD_iou_relation_overlap_coef
        kwargs["SGLD_iou_attract_coef"] = args.SGLD_iou_attract_coef
        kwargs["image_value_range"] = args.image_value_range
        kwargs["w_init_type"] = args.w_init_type
        kwargs["indiv_sample"] = args.indiv_sample
        kwargs["step_size"] = args.step_size
        kwargs["step_size_img"] = args.step_size_img
        kwargs["step_size_z"] = args.step_size_z
        kwargs["step_size_zgnn"] = args.step_size_zgnn
        kwargs["step_size_wtarget"] = args.step_size_wtarget
    return kwargs