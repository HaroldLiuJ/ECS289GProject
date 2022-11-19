import torch
import torch.nn.functional as F
import numpy as np
import sys
import datasets
from tqdm import tqdm


def train(train_dataloader, model, optimizer, step):
    model.train()
    batch_num = 0
    loss = 0
    print_interval = 100

    for i, batch in enumerate(tqdm(train_dataloader)):
        batch_loss, batch_size = train_batch(batch, model, optimizer)

        loss += batch_loss
        batch_num += 1
        step += 1

        if i % print_interval == 0 and i > 0:
            print("Batch {}, Loss: {}".format(i, loss / batch_num))
            sys.stdout.flush()

        # if i > 100:
        #     break

    return loss / batch_num, step


def train_batch(batch, model, optimizer):

    text_input_ids, image = batch
    # print("text_input_ids: {}".format(text_input_ids))
    # print("image: {}".format(image))
    logits_per_image, logits_per_text = model(image.cuda(), text_input_ids.cuda())

    # print("logits_per_image: {}".format(logits_per_image))
    # print("logits_per_text: {}".format(logits_per_text))
    gt = torch.arange(len(logits_per_image)).to(logits_per_image.device)
    # print("gt: {}".format(gt))
    loss_func = torch.nn.CrossEntropyLoss()
    image_loss = loss_func(logits_per_image, gt)
    text_loss = loss_func(logits_per_text, gt)

    loss = (image_loss + text_loss) / 2
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        3
    )

    optimizer.step()
    #
    # print("loss: {}".format(loss))

    return loss.data, text_input_ids.shape[0]
    # print(logits_per_image.shape)
    # print(logits_per_text.shape)



def val(val_dataloader, model):
    model.eval()
    batch_num = 0
    loss = 0
    print_interval = 1000

    for i, batch in enumerate(tqdm(val_dataloader)):
        batch_loss, batch_size = val_batch(batch, model)

        loss += batch_loss
        batch_num += 1

        if i % print_interval == 0 and i > 0:
            print("Batch {}, Loss: {}".format(i, loss / batch_num))
            sys.stdout.flush()
        #
        # if i > 10:
        #     break

    return loss / batch_num


def val_batch(batch, model):
    with torch.no_grad():
        text_input_ids, image = batch
    # print("text_input_ids: {}".format(text_input_ids))
    # print("image: {}".format(image))
        logits_per_image, logits_per_text = model(image.cuda(), text_input_ids.cuda())

    # print("logits_per_image: {}".format(logits_per_image))
    # print("logits_per_text: {}".format(logits_per_text))
        gt = torch.arange(len(logits_per_image)).to(logits_per_image.device)
    # print("gt: {}".format(gt))
        loss_func = torch.nn.CrossEntropyLoss()
        image_loss = loss_func(logits_per_image, gt)
        text_loss = loss_func(logits_per_text, gt)

        loss = (image_loss + text_loss) / 2


    return loss.data, text_input_ids.shape[0]