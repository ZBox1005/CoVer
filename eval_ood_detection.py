import os
import argparse
import numpy as np
import torch
from scipy import stats

from utils.common import setup_seed, get_num_cls, get_test_labels
from utils.detection_util import print_measures, get_and_print_results, get_ood_scores_clip, get_ood_scores_resnet
from utils.file_ops import save_as_dataframe, setup_log
from utils.plot_util import plot_distribution
from utils.train_eval_util import set_model_clip, set_model_resnet50, set_id_loader, set_ood_loader_ImageNet


def process_args():
    parser = argparse.ArgumentParser(description='Evaluates CoVer Score',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # setting for each run
    parser.add_argument('--in_dataset', default='ImageNet', type=str,
                        choices=['ImageNet', 'ImageNet10', 'ImageNet20', 'ImageNet100'], help='in-distribution dataset')
    parser.add_argument('--root-dir', default="datasets", type=str,
                        help='root dir of datasets')
    parser.add_argument('--name', default="eval_ood",
                        type=str, help="unique ID for the run")
    parser.add_argument('--seed', default=5, type=int, help="random seed")
    parser.add_argument('--gpu', default=0, type=int,
                        help='the GPU indice to use')
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        help='mini-batch size')
    parser.add_argument('--T', type=int, default=1,
                        help='temperature parameter')
    parser.add_argument('--model', default='CLIP', choices=['ResNet50', 'CLIP'],
                        type=str, help='model architecture')
    parser.add_argument('--CLIP_ckpt', type=str, default='ViT-B-16',
                        choices=['RN50', 'ViT-B-32', 'ViT-B-16', 'ViT-L-14'],
                        help='which pretrained img encoder to use')
    parser.add_argument('--score', default='MCM', type=str, choices=['CoVer', 'energy'], help='score options')

    # for pretrained resnet and ASH
    parser.add_argument('--train_restore_file', default="resnet50-19c8e357.pth", type=str, help="which pth to use")
    parser.add_argument('--ash_method', default="ash_s@90", type=str, help="which pth to use")

    # for DICE and DICE + ReAct
    parser.add_argument('--p', default=None, type=int, help="p in dice")
    parser.add_argument('--clip_threshold', default=1.0, type=float, help="clip threshold in react")
    args = parser.parse_args()

    args.n_cls = get_num_cls(args)
    args.log_directory = f"results/{args.in_dataset}/{args.score}/{args.model}_{args.CLIP_ckpt}_T_{args.T}_ID_{args.name}/"
    
    for item in imagenet_c.items():
        method = item[0]
        args.log_directory += f"_{method}"
        severities = item[1]
        for severity in severities:
            args.log_directory += f"_{str(severity)}"
    os.makedirs(args.log_directory, exist_ok=True)

    return args


# --- Select the corruption types for input expansion ---
# --- Recommend corruption types from validation set ---
# imagenet_c = {
#     "origin": tuple([1]),
#     "brightness": tuple([1, 2]),
#     "fog": tuple([1, 2]),
#     "saturate": tuple([1, 2]),
#     "motion_blur": tuple([1, 2]),
#     "defocus_blur": tuple([1, 2]),
#     "gaussian_blur": tuple([1, 2]),
# }
# ------
imagenet_c = {
    # Fixed
    "origin": tuple([1]),

    # Selective
    "brightness": tuple([1, 2, 3, 4, 5]),
    "fog": tuple([1, 2, 3, 4, 5]),
    "contrast": tuple([1, 2, 3, 4, 5]),
    "motion_blur": tuple([1, 2, 3, 4, 5]),
    "defocus_blur": tuple([1, 2, 3, 4, 5]),
    "gaussian_blur": tuple([1, 2, 3, 4, 5]),
    "spatter": tuple([1, 2, 3, 4, 5]),
    "saturate": tuple([1, 2, 3, 4, 5]),
    "elastic_transform": tuple([1, 2, 3, 4, 5]),
    "jpeg_compression": tuple([1, 2, 3, 4, 5]),
    "pixelate": tuple([1, 2, 3, 4, 5]),
    "speckle_noise": tuple([1, 2, 3, 4, 5]),
    "glass_blur": tuple([1, 2, 3, 4, 5]),
    "gaussian_noise": tuple([1, 2, 3, 4, 5]),
    "shot_noise": tuple([1, 2, 3, 4, 5]),
    "zoom_blur": tuple([1, 2, 3, 4, 5]),
    "snow": tuple([1, 2, 3, 4, 5]),
    "impulse_noise": tuple([1, 2, 3, 4, 5]),
}


def main():
    args = process_args()
    setup_seed(args.seed)
    log = setup_log(args)
    assert torch.cuda.is_available()
    torch.cuda.set_device(args.gpu)

    # Load OOD detector
    if args.model == 'CLIP':
        net, preprocess = set_model_clip(args)
    elif 'ResNet' in args.model:
        net, preprocess = set_model_resnet50(args)
    else:
        raise ValueError(f"Unsupported model type: {args.model}")
    net.eval()

    # Following MCM
    if args.in_dataset in ['ImageNet10']:
        out_datasets = ['ImageNet20', 'ImageNet100']

    elif args.in_dataset in ['ImageNet20', 'ImageNet100']:
        out_datasets = ['ImageNet10']

    # ImageNet OOD detection benchmark
    elif args.in_dataset in ['ImageNet']:

        # Test
        out_datasets = ['iNaturalist', 'SUN', 'places365', 'dtd']

        # Validation
        # out_datasets = ['SVHN']

    # Get original loader and aug loader dict for ID dataset
    test_loader, in_aug_loader_dict = set_id_loader(args, preprocess)
    test_labels = get_test_labels(args, test_loader)

    # ID set
    if args.model == 'CLIP':
        in_score = get_ood_scores_clip(args, net, in_aug_loader_dict, test_labels, imagenet_c,
                                       dataset_name='ImageNet')
    elif 'ResNet' in args.model:
        in_score = get_ood_scores_resnet(args, net, in_aug_loader_dict, imagenet_c, dataset_name='ImageNet')

    # OOD set
    auroc_list, aupr_list, fpr_list = [], [], []
    for out_dataset in out_datasets:
        log.debug(f"Evaluating OOD dataset {out_dataset}")
        ood_loader, out_aug_loader_dict = set_ood_loader_ImageNet(args, out_dataset, preprocess,
                                                                  root=os.path.join(args.root_dir,
                                                                                    'ImageNet_OOD_dataset'))
        if args.model == 'CLIP':
            out_score = get_ood_scores_clip(args, net, out_aug_loader_dict, test_labels, imagenet_c,
                                            dataset_name=out_dataset)
        elif 'ResNet' in args.model:
            out_score = get_ood_scores_resnet(args, net, out_aug_loader_dict, imagenet_c, dataset_name=out_dataset)
        log.debug(f"in scores: {stats.describe(in_score)}")
        log.debug(f"out scores: {stats.describe(out_score)}")
        plot_distribution(args, in_score, out_score, out_dataset)
        get_and_print_results(args, log, in_score, out_score,
                              auroc_list, aupr_list, fpr_list)

    log.debug('\n\nMean Test Results')
    print_measures(log, np.mean(auroc_list), np.mean(aupr_list),
                   np.mean(fpr_list), method_name=args.score)
    save_as_dataframe(args, out_datasets, fpr_list, auroc_list, aupr_list)


if __name__ == '__main__':
    main()
