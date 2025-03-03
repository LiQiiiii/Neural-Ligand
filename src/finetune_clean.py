import os
import time
import sys
sys.path.append(os.path.abspath('.'))
import torch
from src.args import parse_arguments
from src.datasets.common import get_dataloader, maybe_dictionarize
from src.datasets.registry import get_dataset_cifar_mnist
from src.eval import evaluate
from src.modeling import ImageEncoder, ImageClassifier, MultiHeadImageClassifier
from src.utils import cosine_lr, LabelSmoothing
from src.heads import get_classification_head
import src.datasets as datasets
from torch.utils.data import Subset
import pickle
import numpy as np
import copy
import tqdm

def save_dataset_splits(train_dataset, test_dataset, shadowtrain_dataset, shadowtest_dataset, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'train_dataset.pkl'), 'wb') as f:
        pickle.dump(train_dataset, f)
    
    with open(os.path.join(save_dir, 'test_dataset.pkl'), 'wb') as f:
        pickle.dump(test_dataset, f)
    
    with open(os.path.join(save_dir, 'shadowtrain_dataset.pkl'), 'wb') as f:
        pickle.dump(shadowtrain_dataset, f)

    with open(os.path.join(save_dir, 'shadowtest_dataset.pkl'), 'wb') as f:
        pickle.dump(shadowtest_dataset, f)

def load_datasets(save_dir):
    with open(os.path.join(save_dir, 'train_dataset.pkl'), 'rb') as f:
        train_dataset = pickle.load(f)
    
    with open(os.path.join(save_dir, 'test_dataset.pkl'), 'rb') as f:
        test_dataset = pickle.load(f)
    
    with open(os.path.join(save_dir, 'shadowtrain_dataset.pkl'), 'rb') as f:
        shadowtrain_dataset = pickle.load(f)
    
    with open(os.path.join(save_dir, 'shadowtest_dataset.pkl'), 'rb') as f:
        shadowtest_dataset = pickle.load(f)

    
    return train_dataset, test_dataset, shadowtrain_dataset, shadowtest_dataset
def check_datasets_exist(save_dir):
    return (os.path.exists(os.path.join(save_dir, 'train_dataset.pkl')) and
            os.path.exists(os.path.join(save_dir, 'test_dataset.pkl')) and
            os.path.exists(os.path.join(save_dir, 'shadowtrain_dataset.pkl')) and 
            os.path.exists(os.path.join(save_dir, 'shadowtest_dataset.pkl')))

def load_dataset_splits(save_dir):
    with open(os.path.join(save_dir, 'train_indices.pkl'), 'rb') as f:
        train_indices = pickle.load(f)
    with open(os.path.join(save_dir, 'test_indices.pkl'), 'rb') as f:
        test_indices = pickle.load(f)
    with open(os.path.join(save_dir, 'shadowtrain_indices.pkl'), 'rb') as f:
        shadowtrain_indices = pickle.load(f)
    with open(os.path.join(save_dir, 'shadowtest_indices.pkl'), 'rb') as f:
        shadowtest_indices = pickle.load(f)

    return train_indices, test_indices, shadowtrain_indices, shadowtest_indices

def finetune(model, args):
    dataset = args.dataset
    preprocess_fn = model.train_preprocess

    print_every = 100
    dataset_save_dir = os.path.join("{}/{}/dataset_splits".format(args.save, args.dataset))

    if check_datasets_exist(dataset_save_dir):
        print("Subsets already exits...")
        from torch.utils.data import DataLoader
        train_dataset, test_dataset, shadowtrain_dataset, shadowtest_dataset = load_datasets(dataset_save_dir)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        shadowtrain_loader = DataLoader(shadowtrain_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        print("Subsets do not exist...")
        train_dataset, train_loader = get_dataset_cifar_mnist(
            dataset,
            'train',
            preprocess_fn,
            location=args.data_location,
            batch_size=args.batch_size
        )
        test_dataset, test_loader = get_dataset_cifar_mnist(
            dataset,
            'test',
            preprocess_fn,
            location=args.data_location,
            batch_size=args.batch_size
        )
        shadowtrain_dataset, shadowtrain_loader = get_dataset_cifar_mnist(
            dataset,
            'shadowtrain',
            preprocess_fn,
            location=args.data_location,
            batch_size=args.batch_size
        )
        shadowtest_dataset, shadowtest_loader = get_dataset_cifar_mnist(
            dataset,
            'shadowtest',
            preprocess_fn,
            location=args.data_location,
            batch_size=args.batch_size
        )
        save_dataset_splits(train_dataset, test_dataset, shadowtrain_dataset, shadowtest_dataset, dataset_save_dir)

    num_batches = len(train_loader)
    print("train_length: {}, val_length: {}, shadowtrain_length: {}, shadowtest_length: {}".format(len(train_dataset), len(test_dataset), len(shadowtrain_dataset), len(shadowtest_dataset)))
    # save pre-trained model

    # dataset_dir = dataset + '_1epoch'
    # ckpdir = os.path.join(args.save, dataset_dir)

    ckpdir = os.path.join(args.save, dataset)

    if args.save is not None:
        os.makedirs(ckpdir, exist_ok=True)
        model_path = os.path.join(args.save, f'zeroshot.pt')
        if not os.path.exists(model_path):
            model.image_encoder.save(model_path)
    # evaluate pre-trained model
    print("Initial evaluation:")
    image_encoder = model.image_encoder
    args.eval_datasets = [dataset]
    evaluate(image_encoder, args)

    # test_loaders = [test_loader]
    # evaluate_single(model, test_loaders, args.device)

    # train model for target train set
    loss_fn = torch.nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    scheduler = cosine_lr(optimizer, args.lr, args.warmup_length, args.epochs * num_batches)
    for epoch in range(args.epochs):
        model = model.cuda()
        model.train()
        for i, batch in enumerate(train_loader):
            start_time = time.time()
            step = i + epoch * num_batches
            scheduler(step)
            optimizer.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch['images'].to('cuda:0')
            labels = batch['labels'].to('cuda:0')
            data_time = time.time() - start_time

            logits = model(inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            batch_time = time.time() - start_time

            if step % print_every == 0:
                percent_complete = 100 * i / len(train_loader)
                print(
                    f"Target Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(train_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )
    # evaluate target model
    image_encoder = model.image_encoder
    args.eval_datasets = [dataset] # eval dataset
    evaluate(image_encoder, args)
    
    # Save the finetuned model
    if args.save is not None:
        ft_path = os.path.join(ckpdir, 'finetuned.pt')
        image_encoder.save(ft_path)

def finetune_dev(model_shadow, args):
    dataset = args.dataset
    preprocess_fn = model_shadow.train_preprocess

    print_every = 100
    dataset_save_dir = os.path.join("{}/{}/dataset_splits".format(args.save, args.dataset))

    if check_datasets_exist(dataset_save_dir):
        print("Subsets already exits...")
        from torch.utils.data import DataLoader
        train_dataset, test_dataset, shadowtrain_dataset, shadowtest_dataset = load_datasets(dataset_save_dir)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        shadowtrain_loader = DataLoader(shadowtrain_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        print("Subsets do not exist...")
        train_dataset, train_loader = get_dataset_cifar_mnist(
            dataset,
            'train',
            preprocess_fn,
            location=args.data_location,
            batch_size=args.batch_size
        )
        test_dataset, test_loader = get_dataset_cifar_mnist(
            dataset,
            'test',
            preprocess_fn,
            location=args.data_location,
            batch_size=args.batch_size
        )
        shadowtrain_dataset, shadowtrain_loader = get_dataset_cifar_mnist(
            dataset,
            'shadowtrain',
            preprocess_fn,
            location=args.data_location,
            batch_size=args.batch_size
        )
        shadowtest_dataset, shadowtest_loader = get_dataset_cifar_mnist(
            dataset,
            'shadowtest',
            preprocess_fn,
            location=args.data_location,
            batch_size=args.batch_size
        )
        save_dataset_splits(train_dataset, test_dataset, shadowtrain_dataset, shadowtest_dataset, dataset_save_dir)

    num_batches = len(train_loader)
    print("train_length: {}, val_length: {}, shadowtrain_length: {}, shadowtest_length: {}".format(len(train_dataset), len(test_dataset), len(shadowtrain_dataset), len(shadowtest_dataset)))

    # train model for shadow train set
    model_shadow = model_shadow.to(args.device)
    loss_fn_shadow = torch.nn.CrossEntropyLoss()
    params_shadow = [p for p in model_shadow.parameters() if p.requires_grad]
    optimizer_shadow = torch.optim.AdamW(params_shadow, lr=args.lr, weight_decay=args.wd)
    scheduler_shadow = cosine_lr(optimizer_shadow, args.lr, args.warmup_length, args.epochs * num_batches)
    for epoch in range(args.epochs):
        model_shadow = model_shadow.cuda()
        model_shadow.train()
        for i, batch in enumerate(shadowtrain_loader):
            start_time = time.time()
            step = i + epoch * num_batches
            scheduler_shadow(step)
            optimizer_shadow.zero_grad()

            batch = maybe_dictionarize(batch)
            inputs = batch['images'].to('cuda:0')
            labels = batch['labels'].to('cuda:0')
            data_time = time.time() - start_time

            logits = model_shadow(inputs)
            loss = loss_fn_shadow(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_shadow, 1.0)
            optimizer_shadow.step()
            batch_time = time.time() - start_time

            if step % print_every == 0:
                percent_complete = 100 * i / len(shadowtrain_loader)
                print(
                    f"Shadow Train Epoch: {epoch} [{percent_complete:.0f}% {i}/{len(shadowtrain_loader)}]\t"
                    f"Loss: {loss.item():.6f}\tData (t) {data_time:.3f}\tBatch (t) {batch_time:.3f}", flush=True
                )

    # evaluate shadow model
    image_encoder_shadow = model_shadow.image_encoder
    args.eval_datasets = [dataset] # eval dataset
    evaluate(image_encoder_shadow, args)
    
    ckpdir = os.path.join(args.save, dataset)
    if args.save is not None:
        dev_ft_path = os.path.join(ckpdir, 'finetuned_dev.pt')
        image_encoder_shadow.save(dev_ft_path)

if __name__ == '__main__':
    data_location = "./data"
    models = ['RN50']
    # datasets = ['CIFAR10', 'MNIST', 'GTSRB', 'RESISC45', 'CIFAR100', 'SVHN', 'STL10']
    datasets = ['MNIST']
    
    epochs = {
        'GTSRB': 11,
        'MNIST': 5,
        'RESISC45': 15,
        'SVHN': 4,
        'STL10': 50,
        'CIFAR100': 5,
        'CIFAR10': 5,
    }

    for model_name in models:
        for dataset in datasets:
            print('='*100)
            print(f'Finetuning {model_name} on {dataset}')
            print('='*100)
            args = parse_arguments()

            args.lr = 1e-5
            args.epochs = epochs[dataset]
            args.data_location = data_location
            args.dataset = dataset
            args.batch_size = 32

            args.model = model_name
            args.save = f'./checkpoints/{args.model}'
            args.cache_dir = ''
            args.openclip_cachedir = './open_clip'
            image_encoder = ImageEncoder(args, keep_lang=False)
            classification_head = get_classification_head(args, dataset)
            model = ImageClassifier(image_encoder, classification_head)
            model.freeze_head()
            model_shadow = copy.deepcopy(model)
            
            finetune(model, args)
            finetune_dev(model_shadow, args)