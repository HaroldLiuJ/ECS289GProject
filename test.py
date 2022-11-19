import argparse
import os.path
import random
import numpy as np
import torch
from torchmetrics import F1Score, Precision, Recall
from torchmetrics.functional import precision_recall
from torch import optim
from tqdm import tqdm

from clip import clip
from dataset import COCODataset, get_dataloader, CifarDataset, OxfordDataset
from train_utils import train, val
import datasets

parser = argparse.ArgumentParser()

parser.add_argument("--model_path", type=str, default='./checkpoints')
parser.add_argument("--test_data", type=str, default='cifar10')
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--image_encoder", type=str, default='ViT-B/16')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

test_data: str = args.test_data

assert test_data in ['cifar10', 'cifar100', 'oxford']

print("Loading Model")
model, preprocess = clip.load(args.image_encoder, device=device)
model.load_state_dict(torch.load(args.model_path))
    # ds = COCODataset(preprocess, clip.tokenize, 'validation')
    #
    # dataloader = get_dataloader(ds, 16)
print("Loading {}".format(test_data))
if test_data.startswith('cifar'):
    ds = CifarDataset(test_data, preprocess, clip.tokenize, 'test')
elif test_data.startswith('oxford'):
    ds = OxfordDataset(preprocess, clip.tokenize, 'train')

dataloader = get_dataloader(ds, args.batch_size)

model.eval()

with torch.no_grad():
    pred_labels = []
    gt_labels = []
    for text, image, gt_label in tqdm(dataloader):
        text = text[0, :, :].squeeze()
        logits_per_image, logits_per_text = model(image.cuda(), text.cuda())
        pred_label = torch.topk(torch.softmax(logits_per_image, dim=-1), k=1, dim=-1).indices.squeeze().cpu().tolist()

        pred_labels += pred_label
        gt_labels += gt_label.tolist()
        # break


    pred_labels = torch.tensor(pred_labels)
    gt_labels = torch.tensor(gt_labels)

    # print(pred_labels)
    # print(torch.unique(gt_labels))
    # print(len(ds.class_labels))
    # print(ds.class_labels)
    f1 = F1Score(num_classes=len(ds.class_labels), average='macro')
    precision, recall = precision_recall(pred_labels, gt_labels, average='macro', num_classes=len(ds.class_labels))
    f1_score = f1(pred_labels, gt_labels)



    print("Result on {}".format(test_data))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1: {}".format(f1_score))







