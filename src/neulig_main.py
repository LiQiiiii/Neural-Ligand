import os
import numpy as np
import time
import sys
import tqdm
sys.path.append('.')
sys.path.append('./src')
from src.modeling import ImageEncoder
from task_vectors import TaskVector
# from eval import eval_single_dataset
from args import parse_arguments
from utils import *
import torchvision.transforms as transforms
from PIL import Image
import time
import torchvision.utils as vutils
# from src.datasets.registry import get_dataset
from src.heads import get_classification_head
import torch
from collections import Counter
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from src.datasets.common import get_dataloader, maybe_dictionarize
import timm
from itertools import cycle
from modeling import ImageClassifier, ImageEncoder, ClassificationHead
from open_clip import create_model_and_transforms
from torch.utils.data import DataLoader, TensorDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def merge_ckps(fusion_model, sample_weights):
    flat_ft = torch.vstack([state_dict_to_vector(check, []).to('cpu') for check in fusion_model.ckpts]).to('cpu')
    tv_flat_checks = flat_ft
    final_ck = None 
    for j in range(fusion_model.num_models):
        weighted_value = sample_weights[0, j].to('cpu') * tv_flat_checks[j]
        if final_ck is None:
            final_ck = weighted_value
        else:
            final_ck += weighted_value
    final_ck = final_ck.to(device)
    return final_ck


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
            os.path.exists(os.path.join(save_dir, 'shadowtest_dataset.pkl'))
            )
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


def evaluate_ori(fusion_model, test_loaders, criterion, device):
    fusion_model.eval()  
    total_loss = 0.0
    merged_total_loss = 0.0
    total_correct = []
    total_samples = []
    merged_total_correct = []

    with torch.no_grad():  
        for loader_idx, test_loader in enumerate(test_loaders):
            cur_correct = 0
            cur_samples = 0
            merged_cur_correct = 0
            for i, data in enumerate(tqdm.tqdm(test_loader)):
                data = maybe_dictionarize(data)
                inputs = data['images'].to(device)
                labels = data['labels'].to(device)

                outputs, _ = fusion_model(inputs, dataset_index=loader_idx)
                
                model_outputs = []
                for i, model in enumerate(fusion_model.models):
                    model.eval() 
                    with torch.no_grad():
                        output = model(inputs)
                        model_outputs.append(output)
                weighting_model = fusion_model.get_weighting_model()
                stacked_outputs = torch.cat(model_outputs, dim=1)
                merge_weights = weighting_model(stacked_outputs)
                                
                merged_checks = merge_ckps(fusion_model, merge_weights)
                merged_state_dict = vector_to_state_dict(merged_checks, ptm_check, remove_keys=[])
                image_encoder.load_state_dict(merged_state_dict, strict=False)
                image_encoder.to(device)
                merged_model = ImageClassifier(image_encoder, fusion_model.prediction_heads[loader_idx])
                merged_outputs = merged_model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                merged_loss = criterion(merged_outputs, labels)
                merged_total_loss += merged_loss.item()

                cur_samples += labels.size(0)

                _, predicted = torch.max(outputs.data, 1)
                cur_correct += (predicted == labels).sum().item()

                _, merged_predicted = torch.max(merged_outputs.data, 1)
                merged_cur_correct += (merged_predicted == labels).sum().item()

            total_samples.append(cur_samples)

            total_correct.append(cur_correct)
            merged_total_correct.append(merged_cur_correct)

    accuracies = [100.0 * total_correct[i] / total_samples[i] for i in range(len(total_samples))]
    print("accuracy per task: ", accuracies)
    merged_accuracies = [100.0 * merged_total_correct[i] / total_samples[i] for i in range(len(total_samples))]
    print("merged_accuracy per task: ", merged_accuracies)
    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_loss = total_loss / sum(total_samples)
    
    merged_avg_accuracy = sum(merged_accuracies) / len(merged_accuracies)
    merged_avg_loss = merged_total_loss / sum(total_samples)

    return avg_loss, avg_accuracy, merged_avg_loss, merged_avg_accuracy

class WeightingModel(nn.Module):
    def __init__(self, input_dim=512, num_models=6):
        super(WeightingModel, self).__init__()
        self.num_models = num_models
        self.fc = nn.Linear(input_dim * num_models, num_models)
    def forward(self, x):

        logits = self.fc(x)
        weights = F.softmax(logits, dim=1)  
        return weights


class FusionModel(nn.Module):
    def __init__(self, ckpts, models, prediction_heads, input_dim=1024): # ViT-L-14: 768, RN50: 1024, ViT-B-32: 512
        super(FusionModel, self).__init__()
        self.models = models
        self.prediction_heads = prediction_heads
        self.num_models = len(models)
        self.weighting_model = WeightingModel(input_dim=input_dim, num_models=self.num_models)
        
        self.ckpts = ckpts
        self.flat_ft = torch.vstack([state_dict_to_vector(check, []).to('cpu') for check in self.ckpts]).to('cpu')
        self.mean_ft = torch.mean(self.flat_ft, dim=0)
        self.diff_ft = self.flat_ft - self.mean_ft.unsqueeze(0) # ksi
        self.sum_ft = torch.sum(self.diff_ft, dim=1).to(device)
        
        mean = torch.mean(self.sum_ft)
        std = torch.std(self.sum_ft)
        self.sum_ft = (self.sum_ft - mean) / std

    def forward(self, inputs, dataset_index):
        model_outputs = []
        self.weighting_model.train()
        for i, model in enumerate(self.models):
            model.eval()  
            with torch.no_grad():
                output = model(inputs)
                model_outputs.append(output)

        stacked_outputs = torch.cat(model_outputs, dim=1) 

        weights = self.weighting_model(stacked_outputs) 
        reg_loss = torch.matmul(weights, self.sum_ft)

        tensor_sum = torch.sum(weights)

        weighted_sum = 0
        for i in range(self.num_models):
            weighted_output = model_outputs[i] * weights[:, i].unsqueeze(1)
            weighted_sum += weighted_output
        final_output = self.prediction_heads[dataset_index](weighted_sum)
        return final_output, reg_loss
    
    def get_weighting_model(self):
        return self.weighting_model

args = parse_arguments()
args.save = './checkpoints/{}'.format(args.model)

exam_datasets = ['GTSRB', 'CIFAR100', 'RESISC45', 'CIFAR10', 'MNIST', 'STL10', 'SVHN']
num_classes = [43, 100, 45, 10, 10, 10, 10]
use_merged_model = True

classification_heads = [get_classification_head(args, dataset_name).to(device) for dataset_name in exam_datasets]

import itertools
exam_datasets_list = [list(comb) for comb in itertools.combinations(exam_datasets, args.num_co_models)]
num_classes_list = [list(comb) for comb in itertools.combinations(num_classes, args.num_co_models)]
classification_heads_list = [list(comb) for comb in itertools.combinations(classification_heads, args.num_co_models)]

for mm in range(len(exam_datasets_list)):
    exam_datasets = exam_datasets_list[mm]
    num_classes = num_classes_list[mm]
    classification_heads = classification_heads_list[mm]

    args.save = os.path.join(args.ckpt_dir,args.model)
    args.save = './checkpoints/{}'.format(args.model)
    pretrained_checkpoint = os.path.join(args.save, 'zeroshot.pt')
    image_encoder = torch.load(pretrained_checkpoint)
    image_encoder_shadow = torch.load(pretrained_checkpoint)
    ptm_check = torch.load(pretrained_checkpoint).state_dict()

    from tm_utils import *
    ft_checks, ft_checks_shadow = [], []
    ft_archs, ft_archs_shadow = [], []

    for dataset_name in exam_datasets:
        ckpt_name = os.path.join(args.save, dataset_name, 'finetuned.pt')
        ckpt_name_shadow = os.path.join(args.save, dataset_name, 'finetuned_dev.pt')
        ft_archs.append(torch.load(ckpt_name).to(device))
        ft_archs_shadow.append(torch.load(ckpt_name_shadow).to(device))
        ft_checks.append(torch.load(ckpt_name).state_dict())
        ft_checks_shadow.append(torch.load(ckpt_name_shadow).state_dict())
        print(ckpt_name)
        print(ckpt_name_shadow)

    if args.model == 'RN50': 
        fusion_model = FusionModel(ft_checks, ft_archs, classification_heads, input_dim=1024)
        fusion_model_shadow = FusionModel(ft_checks_shadow, ft_archs_shadow, classification_heads, input_dim=1024)
    elif args.model == 'ViT-B-32':
        fusion_model = FusionModel(ft_checks, ft_archs, classification_heads, input_dim=512)
        fusion_model_shadow = FusionModel(ft_checks_shadow, ft_archs_shadow, classification_heads, input_dim=512)
    elif args.model == 'ViT-L-14':
        fusion_model = FusionModel(ft_checks, ft_archs, classification_heads, input_dim=768)
        fusion_model_shadow = FusionModel(ft_checks_shadow, ft_archs_shadow, classification_heads, input_dim=768)
    test_loaders, train_loaders, shadowtrain_loaders, shadowtest_loaders, adv_test_loaders = [], [], [], [], []

    for num_ld in range(len(exam_datasets)):
        dataset_save_dir = os.path.join("{}/{}/dataset_splits".format(args.save, exam_datasets[num_ld]))
        print("cur_process_dataset: ", dataset_save_dir)
        if check_datasets_exist(dataset_save_dir):
            print("Subsets already exits...")
            from torch.utils.data import DataLoader

            train_dataset, test_dataset, shadowtrain_dataset, shadowtest_dataset = load_datasets(dataset_save_dir)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

            shadowtrain_loader = DataLoader(shadowtrain_dataset, batch_size=args.batch_size, shuffle=True)

            shadowtest_loader = DataLoader(shadowtest_dataset, batch_size=args.batch_size, shuffle=True)

            test_loaders.append(test_loader) 
            train_loaders.append(train_loader)

        print("dataset: {}, train_length: {}, test_length: {}, shadowtrain_length: {}, shadowtest_length: {}".format(exam_datasets[num_ld], len(train_dataset), len(test_dataset), len(shadowtrain_dataset), len(shadowtest_dataset)))

    fusion_model = fusion_model.to(device)
    fusion_model_shadow = fusion_model_shadow.to(device)

    optimizer = optim.Adam(fusion_model.weighting_model.parameters(), lr=0.001)
    optimizer_shadow = optim.Adam(fusion_model_shadow.weighting_model.parameters(), lr=0.001)

    criterion = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()

    print("#########################################################")
    print("###############PortLand Training Begins##################")
    print("#########################################################")
    # avg_loss, accuracy, merged_avg_loss, merged_accuracy = evaluate_ori(fusion_model, test_loaders, criterion, device)
    # print(f"Initial Evaluation - Avg Loss: {avg_loss:.4f}, Merged Avg Loss: {merged_avg_loss:.4f}, Ensembling Accuracy: {accuracy:.2f}%, Merging Accuracy: {merged_accuracy:.2f}%")
    best_accuracy = 0.0
    for glb_ep in range(args.global_epoch):
        fusion_model.train()
        loaders_cycle = [cycle(loader) for loader in train_loaders]
        total_batches = min(len(loader) for loader in train_loaders)

        for batch_idx in range(total_batches):
            for loader_idx, loader in enumerate(loaders_cycle):
                data = next(loader)

                data = maybe_dictionarize(data)
                inputs = data['images'].to(device)
                labels = data['labels'].to(device)

                outputs, reg_loss = fusion_model(inputs, dataset_index=loader_idx)

                target = torch.zeros_like(reg_loss).to(device)
                loss_reg = criterion_reg(reg_loss, target) / args.scaling
                
                if args.alignment_type == 'sup':
                    loss_ce = criterion(outputs, labels)
                elif args.alignment_type == 'semi': # semi-supervised (entropy minimization)
                    probs = F.softmax(outputs, dim=1)
                    loss_ce = -torch.mean(torch.sum(probs * torch.log(probs + 1e-6), dim=1))

                loss = loss_ce + loss_reg
                
                print(f"Epoch: {glb_ep}, Current Dataset Index: {loader_idx}, Batch: {batch_idx + 1}/{total_batches}, Loss: {loss.item():.4f}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if (glb_ep+1)%1==0:
            avg_loss, accuracy, merged_avg_loss, merged_accuracy = evaluate_ori(
                fusion_model, test_loaders, criterion, device
            )
            print("portland original performance test: ")
            print(
                f"Epoch [{glb_ep + 1}/{args.global_epoch}] Evaluation - "
                f"Ensembling Avg Loss: {avg_loss:.4f}, Merging Avg Loss: {merged_avg_loss:.4f}, "
                f"Ensembling Accuracy: {accuracy:.2f}%, Merging Accuracy: {merged_accuracy:.2f}%"
            )

            if merged_accuracy > best_accuracy:
                best_accuracy = merged_accuracy 
                print(f"New best model found with accuracy: {best_accuracy:.2f}%")

    print("#########################################################")
    print("################PortLand Training Ends###################")
    print("#########################################################")

