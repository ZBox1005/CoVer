import os
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
import open_clip

from DNNs.resnet_ash import ResNet50, ResNet101
from DNNs.resnet_dice import resnet50

import utils.svhn_loader as svhn


imagenet_c = ["fog", "motion_blur", "brightness", "snow", "defocus_blur", "glass_blur", \
                "gaussian_noise", "shot_noise", "impulse_noise", "contrast", "elastic_transform", "pixelate", \
                "jpeg_compression", "speckle_noise", "spatter", "gaussian_blur", "saturate", "zoom_blur"]

def set_model_resnet50(args):
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
    # For ASH
    # if args.train_restore_file:
    #     checkpoint = os.path.join("DNNs/checkpoints", args.train_restore_file)
    #     checkpoint = torch.load(checkpoint, map_location='cpu')
    #     model = ResNet50() 
    #     model.load_state_dict(checkpoint)
    # else:
    #     print('Warning: train_restore_file config not specified')
    # setattr(model, 'ash_method', args.ash_method)

    # For DICE
    info = np.load(f"models/checkpoints/{args.in_dataset}_{args.model}_feat_stat.npy")
    model = resnet50(num_classes=1000, pretrained=True, p=args.p, info=info, clip_threshold=args.clip_threshold)

    model = model.cuda()

    normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                         std=(0.229, 0.224, 0.225))  # for ResNet
    val_preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    
    return model, val_preprocess

def set_model_clip(args):
    '''
    load Huggingface CLIP
    '''
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.CLIP_ckpt in ['ViT-B-16', 'ViT-B-32', 'ViT-L-14', 'RN50']:
        model, _, _ = open_clip.create_model_and_transforms(args.CLIP_ckpt, pretrained='openai', device=device)
    else:
        model, _, _ = open_clip.create_model_and_transforms(args.CLIP_ckpt, pretrained='metaclip_400m', device=device)
    # model = model.cuda()
    normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
    val_preprocess = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])
    
    return model, val_preprocess

def set_id_loader(args, preprocess=None):
    root = args.root_dir
    if preprocess == None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                         std=(0.26862954, 0.26130258, 0.27577711))  # for CLIP
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    aug_id_loader_dict = {}
    if args.in_dataset == "ImageNet":
        path = os.path.join(root, 'ImageNet', 'val')

        path_aug = ''
        if args.model == 'CLIP':
            path_aug = os.path.join(root, 'DistortedImageNet', 'val_224')
        elif 'ResNet' in args.model:
            path_aug = os.path.join(root, 'DistortedImageNet', 'val')

        # Original Loader
        id_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(path, transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)

        # Corruption Loaders
        for data in imagenet_c:
            aug_loaders = []
            aug_sets = []
            for aug_level in range(1,6):
                aug_level = str(aug_level)
                testset = torchvision.datasets.ImageFolder(root=f"{path_aug}" + "/" + data + "/" + aug_level, transform=preprocess)
                testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)

                aug_sets.append(testset)
                aug_loaders.append(testloader)
            aug_id_loader_dict[data] = aug_loaders

    elif args.in_dataset in ["ImageNet10", "ImageNet20", "ImageNet100"]:
        path = os.path.join(root, args.in_dataset, 'val')
        path_aug = os.path.join(root, args.in_dataset, 'Distorted', 'val')

        id_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(path, transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)

        for data in imagenet_c:
            aug_loaders = []
            aug_sets = []
            for aug_level in range(1,6):
                aug_level = str(aug_level)
                testset = torchvision.datasets.ImageFolder(root=f"{path_aug}" + "/" + data + "/" + aug_level, transform=preprocess)
                testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)

                aug_sets.append(testset)
                aug_loaders.append(testloader)
            aug_id_loader_dict[data] = aug_loaders
        
    aug_id_loader_dict["origin"] = [id_loader]

    return id_loader, aug_id_loader_dict


def set_ood_loader_ImageNet(args, out_dataset, preprocess, root):
    '''
    set OOD loader for ImageNet scale datasets
    '''

    if args.model == 'CLIP':
        root_aug = os.path.join(root, 'Distorted_CLIP')
    elif 'ResNet' in args.model:
        root_aug = os.path.join(root, 'Distorted_ResNet')

    # Store the Original and Corrupted loaders
    aug_ood_loader_dict = {}

    if out_dataset == 'iNaturalist':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'iNaturalist'), transform=preprocess)
        for data in imagenet_c:
            aug_ood_sets = []
            aug_ood_loaders = []
            for aug_level in range(1,6):
                aug_level = str(aug_level)
                testset = torchvision.datasets.ImageFolder(root=f"{root_aug}" + "/iNaturalist/" + data + "/" + aug_level,
                                                        transform=preprocess)
                testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
                aug_ood_sets.append(testset)
                aug_ood_loaders.append(testloader)
            aug_ood_loader_dict[data] = aug_ood_loaders

    elif out_dataset == 'SUN':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'SUN'), transform=preprocess)
        for data in imagenet_c:
            aug_ood_sets = []
            aug_ood_loaders = []
            for aug_level in range(1,6):
                aug_level = str(aug_level)
                testset = torchvision.datasets.ImageFolder(root=f"{root_aug}" + "/SUN/" + data + "/" + aug_level,
                                                        transform=preprocess)
                testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
                aug_ood_sets.append(testset)
                aug_ood_loaders.append(testloader)
            aug_ood_loader_dict[data] = aug_ood_loaders

    elif out_dataset == 'places365':  # filtered places
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'Places'), transform=preprocess)
        for data in imagenet_c:
            aug_ood_sets = []
            aug_ood_loaders = []
            for aug_level in range(1,6):
                aug_level = str(aug_level)
                testset = torchvision.datasets.ImageFolder(root=f"{root_aug}" + "/Places/" + data + "/" + aug_level,
                                                        transform=preprocess)
                testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
                aug_ood_sets.append(testset)
                aug_ood_loaders.append(testloader)
            aug_ood_loader_dict[data] = aug_ood_loaders

    elif out_dataset == 'placesbg':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'placesbg'), transform=preprocess)
        for data in imagenet_c:
            aug_ood_sets = []
            aug_ood_loaders = []
            for aug_level in range(1,6):
                aug_level = str(aug_level)
                testset = torchvision.datasets.ImageFolder(root=f"{root_aug}" + "/placesbg/" + data + "/" + aug_level,
                                                        transform=preprocess)
                testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
                aug_ood_sets.append(testset)
                aug_ood_loaders.append(testloader)
            aug_ood_loader_dict[data] = aug_ood_loaders

    elif out_dataset == 'dtd':
        testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'dtd', 'images'),
                                                      transform=preprocess)
        for data in imagenet_c:
            aug_ood_sets = []
            aug_ood_loaders = []
            for aug_level in range(1,6):
                aug_level = str(aug_level)
                testset = torchvision.datasets.ImageFolder(root=f"{root_aug}" + "/dtd/images/" + data + "/" + aug_level,
                                                        transform=preprocess)
                testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
                aug_ood_sets.append(testset)
                aug_ood_loaders.append(testloader)
            aug_ood_loader_dict[data] = aug_ood_loaders
    
    elif out_dataset == 'SVHN':
        # testsetout = torchvision.datasets.ImageFolder(root=os.path.join(root, 'svhn'),
        #                                               transform=preprocess)
        testsetout = svhn.SVHN(root=os.path.join(root, 'svhn'), split='test', transform=preprocess, download=False)
        for data in imagenet_c:
            aug_ood_sets = []
            aug_ood_loaders = []
            for aug_level in range(1,6):
                aug_level = str(aug_level)
                testset = torchvision.datasets.ImageFolder(root=f"{root_aug}" + "/svhn/" + data + "/" + aug_level,
                                                        transform=preprocess)
                testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
                aug_ood_sets.append(testset)
                aug_ood_loaders.append(testloader)
            aug_ood_loader_dict[data] = aug_ood_loaders

    elif out_dataset == 'ImageNet10':
        path = os.path.join(args.root_dir, out_dataset, 'train')
        path_aug = os.path.join(args.root_dir, out_dataset, 'Distorted', 'train')
        testsetout = datasets.ImageFolder(path, transform=preprocess)
        for data in imagenet_c:
            aug_ood_sets = []
            aug_ood_loaders = []
            for aug_level in range(1,6):
                aug_level = str(aug_level)
                testset = torchvision.datasets.ImageFolder(root=f"{path_aug}" + "/" + data + "/" + aug_level, transform=preprocess)
                testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

                aug_ood_sets.append(testset)
                aug_ood_loaders.append(testloader)
            aug_ood_loader_dict[data] = aug_ood_loaders

    elif out_dataset in ["ImageNet20", "ImageNet100"]:
        path = os.path.join(args.root_dir, out_dataset, 'val')
        path_aug = os.path.join(args.root_dir, out_dataset, 'Distorted', 'val')
        testsetout = datasets.ImageFolder(path, transform=preprocess)
        for data in imagenet_c:
            aug_ood_sets = []
            aug_ood_loaders = []
            for aug_level in range(1,6):
                aug_level = str(aug_level)
                testset = torchvision.datasets.ImageFolder(root=f"{path_aug}" + "/" + data + "/" + aug_level, transform=preprocess)
                testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)
                aug_ood_sets.append(testset)
                aug_ood_loaders.append(testloader)
            aug_ood_loader_dict[data] = aug_ood_loaders

    ood_loader = torch.utils.data.DataLoader(testsetout, batch_size=args.batch_size,
                                                shuffle=False, num_workers=4)
    aug_ood_loader_dict["origin"] = [ood_loader]

    return ood_loader, aug_ood_loader_dict

