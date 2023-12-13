import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(
        figsize=(12, 18 + 2)
    )  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(
        top=0.8
    )  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n**0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join(
            [
                f"{task} - gt: {gt_label}, pred: {pred_label}"
                for gt_label, pred_label, task in zip(
                    gt_decoded_labels, pred_decoded_labels, tasks
                )
            ]
        )

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(path, exist_ok=False):
    """Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"


def train(data_dir, model_dir, args):
    seed_everything(args.seed)

    save_dir = increment_path(os.path.join(model_dir, args.name))

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(
        import_module("dataset"), args.dataset
    )  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform_module = getattr(
        import_module("dataset"), args.augmentation
    )  # default: BaseAugmentation
    transform = transform_module(
        resize=args.resize,
        mean=dataset.mean,
        std=dataset.std,
    )
    dataset.set_transform(transform)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # -- model
    #model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    # model = model_module(num_classes=num_classes).to(device)

    #resnext50
    model_age = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
    model_mask = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
    model_gender = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)

    # #resnext101
    # model_age = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resneXt')#'pytorch/vision:v0.10.0', 'resnext101_32x4d', pretrained=True)
    # model_mask = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resneXt')
    # model_gender = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resneXt')
    # freezing
    # for param in model_age.parameters():
    #     param.requires_grad = False
    # for param in model_mask.parameters():
    #     param.requires_grad = False
    # for param in model_gender.parameters():
    #     param.requires_grad = False
    # fc layer 수정
    fc_in_features = model_age.fc.in_features
    model_age.fc = nn.Linear(fc_in_features, 3)
    model_mask.fc = nn.Linear(fc_in_features, 3)
    model_gender.fc = nn.Linear(fc_in_features, 2)
    model_age = model_age.to(device)
    model_mask = model_mask.to(device)
    model_gender = model_gender.to(device)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer_age = opt_module(
        filter(lambda p: p.requires_grad, model_age.parameters()),
        lr=args.lr,
        weight_decay=5e-4,
    )
    optimizer_mask = opt_module(
        filter(lambda p: p.requires_grad, model_mask.parameters()),
        lr=args.lr,
        weight_decay=5e-4,
    )
    optimizer_gender = opt_module(
        filter(lambda p: p.requires_grad, model_gender.parameters()),
        lr=args.lr,
        weight_decay=5e-4,
    )
    scheduler_age = StepLR(optimizer_age, args.lr_decay_step, gamma=0.5)
    scheduler_mask = StepLR(optimizer_mask, args.lr_decay_step, gamma=0.5)   
    scheduler_gender = StepLR(optimizer_gender, args.lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        # train loop
        model_age.train()
        model_mask.train()
        model_gender.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, age_label,mask_label,gender_label = train_batch
            
            inputs = inputs.to(device)
            age_label,mask_label,gender_label = torch.tensor(age_label).to(device),torch.tensor(mask_label).to(device),torch.tensor(gender_label).to(device)
            
            ###age###
            optimizer_age.zero_grad()

            logits = model_age(inputs)
            preds_age = torch.argmax(logits, dim=-1)
            loss_age = criterion(logits, age_label)

            loss_age.backward()
            optimizer_age.step()
            
            ###mask###
            optimizer_mask.zero_grad()

            logits = model_mask(inputs)
            preds_mask = torch.argmax(logits, dim=-1)
            loss_mask = criterion(logits, mask_label)

            loss_mask.backward()
            optimizer_mask.step()
            
            ###gender###
            optimizer_gender.zero_grad()

            logits = model_gender(inputs)
            preds_gender = torch.argmax(logits, dim=-1)
            loss_gender = criterion(logits, gender_label)

            loss_gender.backward()
            optimizer_gender.step()
            

            loss_value += (loss_age.item()+loss_mask.item()+loss_gender.item())/3
            correct_predictions = (preds_age == age_label) & (preds_mask == mask_label) & (preds_gender == gender_label)
            matches += correct_predictions.sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer_age)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar(
                    "Train/loss", train_loss, epoch * len(train_loader) + idx
                )
                logger.add_scalar(
                    "Train/accuracy", train_acc, epoch * len(train_loader) + idx
                )

                loss_value = 0
                matches = 0

        scheduler_age.step()
        scheduler_mask.step()
        scheduler_gender.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model_age.eval()
            model_mask.eval()
            model_gender.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            for val_batch in val_loader:
                inputs, age_label,mask_label,gender_label = val_batch
                inputs = inputs.to(device)
                age_label,mask_label,gender_label = torch.tensor(age_label).to(device),torch.tensor(mask_label).to(device),torch.tensor(gender_label).to(device)

                logits_age, logits_mask, logits_gender = model_age(inputs),model_mask(inputs),model_gender(inputs)
                preds_age,preds_mask,preds_gender = torch.argmax(logits_age, dim=-1),torch.argmax(logits_mask, dim=-1),torch.argmax(logits_gender, dim=-1)

                loss_item_age, loss_item_mask, loss_item_gender = criterion(logits_age, age_label).item(),criterion(logits_mask, mask_label).item(),criterion(logits_gender, gender_label).item()
                acc=(age_label==preds_age) & (mask_label==preds_mask) & (gender_label==preds_gender)
                acc_item = acc.sum().item()
                val_loss_items.append((loss_item_age+loss_item_mask+loss_item_gender)/3)
                val_acc_items.append(acc_item)

                if figure is None:
                    inputs_np = (
                        torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    )
                    inputs_np = dataset_module.denormalize_image(
                        inputs_np, dataset.mean, dataset.std
                    )
                    figure = grid_image(
                        inputs_np,
                        age_label,
                        preds_age,
                        n=16,
                        shuffle=args.dataset != "MaskSplitByProfileDataset",
                    )

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                print(
                    f"New best model for val accuracy : {val_acc:4.2%}! saving the best model.."
                )
                torch.save({
                    'model_age': model_age.state_dict(),  
                    'model_mask': model_mask.state_dict(),
                    'model_gender': model_gender.state_dict(),
                    }, f"{save_dir}/best.pth")
                best_val_acc = val_acc
            torch.save({
                    'model_age': model_age.state_dict(),  
                    'model_mask': model_mask.state_dict(),
                    'model_gender': model_gender.state_dict(),
                    }, f"{save_dir}/last.pth")
            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_figure("results", figure, epoch)
            print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 1)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MaskBaseDataset",
        help="dataset augmentation type (default: MaskBaseDataset)",
    )
    parser.add_argument(
        "--augmentation",
        type=str,
        default="BaseAugmentation",
        help="data augmentation type (default: BaseAugmentation)",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        default=[128, 96],
        help="resize size for image when training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--valid_batch_size",
        type=int,
        default=1000,
        help="input batch size for validing (default: 1000)",
    )
    parser.add_argument(
        "--model", type=str, default="MyModel", help="model type (default: BaseModel)"
    )
    parser.add_argument(
        "--optimizer", type=str, default="Adam", help="optimizer type (default: SGD)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="ratio for validaton (default: 0.2)",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="cross_entropy",
        help="criterion type (default: cross_entropy)",
    )
    parser.add_argument(
        "--lr_decay_step",
        type=int,
        default=20,
        help="learning rate scheduler deacy step (default: 20)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=20,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--name", default="exp", help="model save at {SM_MODEL_DIR}/{name}"
    )

    # Container environment
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAIN", "../../../train/images"),
    )
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model")
    )

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
