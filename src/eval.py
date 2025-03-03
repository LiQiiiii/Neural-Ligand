import os
import json
import tqdm
import torch
import numpy as np
import utils
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.templates import get_templates
from heads import get_classification_head, build_classification_head
from modeling import ImageClassifier, ImageEncoder, ClassificationHead
from src.datasets.registry import get_dataset_cifar_mnist
import torchvision.utils as vutils
from src.utils import *

def eval_single_dataset(image_encoder, dataset_name, args, backdoor_info=None):
    print("")
    #
    classification_head = get_classification_head(args, dataset_name)
    model = ImageClassifier(image_encoder, classification_head)
    model.eval()

    #
    test_dataset, test_loader = get_dataset_cifar_mnist(
        dataset_name,
        'shadowtest',
        model.val_preprocess,
        location=args.data_location,
        batch_size=args.batch_size
    )
    normalizer = model.val_preprocess.transforms[-1]
    inv_normalizer = NormalizeInverse(normalizer.mean, normalizer.std)
    print("Evaluation Size:", len(test_dataset))

    device = args.device

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(test_loader)):
            data = maybe_dictionarize(data)
            x = data['images']
            y = data['labels']

            x = x.cuda()
            y = y.cuda()
            logits = utils.get_logits(x, model)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)

        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Accuracy: {100*top1:.2f}%')

    return metrics

def eval_single_dataset_head(image_encoder, head, dataset_name, args):
    model = ImageClassifier(image_encoder, head)
    model.eval()
    test_dataset, test_loader = get_dataset_cifar_mnist(dataset_name, 'test', model.val_preprocess, location=args.data_location,  batch_size=args.batch_size)
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(test_loader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)
            logits = utils.get_logits(x, model)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)
        top1 = correct / n

    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100 * top1:.2f}%')
    return metrics

def eval_single_dataset_preprocess_head(image_encoder, head, dataset_name, args):
    model = ImageClassifier(image_encoder, head)
    model.eval()
    test_dataset, test_loader = get_dataset_cifar_mnist(dataset_name, model.val_preprocess, 'test', location=args.data_location,  batch_size=args.batch_size)
    device = args.device

    with torch.no_grad():
        top1, correct, n = 0., 0., 0.
        for i, data in enumerate(tqdm.tqdm(test_loader)):
            data = maybe_dictionarize(data)
            x = data['images'].to(device)
            y = data['labels'].to(device)
            logits = utils.get_logits(x, model)
            pred = logits.argmax(dim=1, keepdim=True).to(device)
            correct += pred.eq(y.view_as(pred)).sum().item()
            n += y.size(0)
        top1 = correct / n
    metrics = {'top1': top1}
    print(f'Done evaluating on {dataset_name}. Accuracy: {100 * top1:.2f}%')
    return metrics

def evaluate(image_encoder, args, backdoor_info=None):
    if args.eval_datasets is None:
        return
    info = vars(args)
    for i, dataset_name in enumerate(args.eval_datasets):
        print('Evaluating on', dataset_name)

        results = eval_single_dataset(image_encoder, dataset_name, args, backdoor_info)

        for key, val in results.items():
            if 'worst' in key or 'f1' in key.lower() or 'pm0' in key:
                print(f"{dataset_name} {key}: {val:.4f}")
            if backdoor_info is not None:
                info[dataset_name + '-B:' + key] = val # trigger
            else:
                info[dataset_name + ':' + key] = val # clean
    return info