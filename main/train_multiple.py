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
import torchvision.models as models

from dataset import MaskBaseDataset
from loss import create_criterion

from accuracy_loss_print import AccuracyLoss


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
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model_age = model_module(num_classes=3).to(device)
    model_mask = model_module(num_classes=3).to(device)
    model_gender = model_module(num_classes=2).to(device)


    # -- loss & metric
    criterion_age = create_criterion(args.criterion_age)  # default: cross_entropy
    criterion_mask = create_criterion(args.criterion_mask)
    criterion_gender = create_criterion(args.criterion_gender)
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
        torch.cuda.empty_cache()
        # train loop
        model_age.train()
        model_mask.train()
        model_gender.train()
        loss_val_age = 0
        loss_val_mask = 0
        loss_val_gender = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            mask_label,gender_label, age_label = MaskBaseDataset.decode_multi_class(labels)
            
            inputs = inputs.to(device)
            age_label,mask_label,gender_label = torch.tensor(age_label).to(device),torch.tensor(mask_label).to(device),torch.tensor(gender_label).to(device)
            
            ###age###
            optimizer_age.zero_grad()

            logits = model_age(inputs)
            preds_age = torch.argmax(logits, dim=-1)
            loss_age = criterion_age(logits, age_label)

            loss_age.backward()
            optimizer_age.step()
            
            ###mask###
            optimizer_mask.zero_grad()

            logits = model_mask(inputs)
            preds_mask = torch.argmax(logits, dim=-1)
            loss_mask = criterion_mask(logits, mask_label)

            loss_mask.backward()
            optimizer_mask.step()
            
            ###gender###
            optimizer_gender.zero_grad()

            logits = model_gender(inputs)
            preds_gender = torch.argmax(logits, dim=-1)
            loss_gender = criterion_gender(logits, gender_label)

            loss_gender.backward()
            optimizer_gender.step()
            

            loss_val_age += loss_age.item()
            loss_val_mask += loss_mask.item()
            loss_val_gender += loss_gender.item()
            correct_predictions = (preds_age == age_label) & (preds_mask == mask_label) & (preds_gender == gender_label)
            matches += correct_predictions.sum().item()
            if (idx + 1) % args.log_interval == 0:
                train_loss_age = loss_val_age / args.log_interval
                train_loss_mask = loss_val_mask / args.log_interval
                train_loss_gender = loss_val_gender / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer_age)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training age loss {train_loss_age:4.4} ||training mask loss {train_loss_mask:4.4} || training gender loss {train_loss_gender:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar(
                    "Train/loss", train_loss_age, epoch * len(train_loader) + idx
                )
                logger.add_scalar(
                    "Train/accuracy", train_acc, epoch * len(train_loader) + idx
                )

                loss_val_age =0
                loss_val_mask=0
                loss_val_gender=0
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
            val_loss_items_age = []
            val_loss_items_mask = []
            val_loss_items_gender = []
            val_acc_items = []
            figure = None
            for val_batch in val_loader:
                inputs, labels = val_batch
                mask_label,gender_label, age_label = MaskBaseDataset.decode_multi_class(labels)
                
                inputs = inputs.to(device)
                age_label,mask_label,gender_label = torch.tensor(age_label).to(device),torch.tensor(mask_label).to(device),torch.tensor(gender_label).to(device)

                logits_age, logits_mask, logits_gender = model_age(inputs),model_mask(inputs),model_gender(inputs)
                preds_age,preds_mask,preds_gender = torch.argmax(logits_age, dim=-1),torch.argmax(logits_mask, dim=-1),torch.argmax(logits_gender, dim=-1)

                loss_item_age, loss_item_mask, loss_item_gender = criterion_age(logits_age, age_label).item(),criterion_mask(logits_mask, mask_label).item(),criterion_gender(logits_gender, gender_label).item()
                acc=(age_label==preds_age) & (mask_label==preds_mask) & (gender_label==preds_gender)
                acc_item = acc.sum().item()
                val_loss_items_age.append(loss_item_age)
                val_loss_items_mask.append(loss_item_mask)
                val_loss_items_gender.append(loss_item_gender)
                val_acc_items.append(acc_item)
                
                preds=MaskBaseDataset.encode_multi_class(preds_mask,preds_gender,preds_age)

                if figure is None:
                    inputs_np = (
                        torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    )
                    inputs_np = dataset_module.denormalize_image(
                        inputs_np, dataset.mean, dataset.std
                    )
                    figure = grid_image(
                        inputs_np,
                        labels,
                        preds,
                        n=16,
                        shuffle=args.dataset != "MaskSplitByProfileDataset",
                    )

            val_loss_age = np.sum(val_loss_items_age) / len(val_loader)
            val_loss_mask = np.sum(val_loss_items_mask) / len(val_loader)
            val_loss_gender = np.sum(val_loss_items_gender) / len(val_loader)
            
            val_loss = loss_mask + loss_age + loss_gender
            
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, (val_loss_age+val_loss_mask+val_loss_gender)/3)
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
                f"[Val] acc : {val_acc:4.2%}, age loss: {val_loss_age:4.2} || val loss: {val_loss_mask:4.2} || gender loss: {val_loss_gender:4.2} "
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
        "--resume_from", type=str, help="path of model to resume training"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed (default: 42)"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="number of epochs to train (default: 10)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MaskSplitByProfileDataset",
        help="dataset augmentation type (default: MaskSplitByProfileDataset)",
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
        "--model", type=str, default="BaseModel", help="model type (default: BaseModel)"
    )
    parser.add_argument(
        "--optimizer", type=str, default="SGD", help="optimizer type (default: SGD)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="ratio for validaton (default: 0.2)",
    )
    parser.add_argument(
        "--criterion_age",
        type=str,
        default="f1",
        help="criterion type for age(default: f1)",
    )
    parser.add_argument(
        "--criterion_mask",
        type=str,
        default="cross_entropy",
        help="criterion type for mask(default: f1)",
    )
    parser.add_argument(
        "--criterion_gender",
        type=str,
        default="cross_entropy",
        help="criterion type for gender(default: f1)",
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
