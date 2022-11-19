import argparse
import os.path
import random
import numpy as np
import torch
from torch import optim

from clip import clip
from dataset import COCODataset, get_dataloader
from train_utils import train, val

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=20, help='Random seed.')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate.')
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--image_encoder", type=str, default='ViT-B/16')
parser.add_argument("--save_root_path", type=str, default='./checkpoints')


args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

save_root_path = args.save_root_path

model_save_root_path = os.path.join(save_root_path, "{}".format(args.image_encoder))

if not os.path.exists(model_save_root_path):
    os.makedirs(model_save_root_path)

epochs = args.epochs
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

torch.cuda.manual_seed(args.seed)

print(clip.available_models())

print("Loading Model")
model, preprocess = clip.load(args.image_encoder, device=device, from_pretrained=False)

print("Reading Dataset")
train_dataset = COCODataset(preprocess, clip.tokenize, 'train')
val_dataset = COCODataset(preprocess, clip.tokenize, 'validation')

train_dataloader = get_dataloader(train_dataset, args.batch_size)
val_dataloader = get_dataloader(val_dataset, args.batch_size)


print("Building Optimizer")
optimizer = optim.AdamW(model.parameters(), lr=args.lr)


step = 0
val_loss_best = 100
for i in range(epochs):
    train_loss, step = train(train_dataloader, model, optimizer, step)
    # train_loss.append(loss)
    print("At Epoch {}, Train Loss: {}".format(i, train_loss))

    torch.cuda.empty_cache()

    print("Validating")
    val_loss = val(val_dataloader, model)
    print("At Epoch {}, Val Loss: {}".format(i, val_loss))
    if val_loss < val_loss_best:
        print("Saving Model...")
        model_save_path = os.path.join(model_save_root_path, "epoch_{}_loss_{}.pt".format(i, val_loss))
        torch.save(model.state_dict(), model_save_path)
        val_loss_best = val_loss


    # rouge1_score, loss = val(val_dataloader, hgs)
    # scheduler.step(rouge1_score)
    # writer.add_scalar('Loss/val', loss, i)
    # writer.add_scalar('Rouge 1/val', rouge1_score, i)
    # torch.cuda.empty_cache()

    # print("At Epoch {}, Val Loss: {}, Val R1: {}".format(i, loss, rouge1_score))