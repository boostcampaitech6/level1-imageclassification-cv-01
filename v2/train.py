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
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
    model = model_module(num_classes=num_classes).to(device)
    model = torch.nn.DataParallel(model)
        
    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4,
    )
        
    
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_acc = 0
    best_val_loss = np.inf
    best_epoch = 0 
    
    start_epoch = 0
    
    if args.resume_from:
        model_data = torch.load(args.resume_from)
        model.load_state_dict(model_data['model_state_dict'])
        optimizer.load_state_dict(model_data['optimizer_state_dict'])
        start_epoch = model_data['epoch'] + 1
    
    
    for epoch in range(start_epoch, args.epochs):
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)

            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()
            
            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            train_accloss = AccuracyLoss(labels, preds, outs, criterion)
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                train_loss_dict, train_acc_dict = train_accloss.loss_acc(args.log_interval, 1)

                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training total loss {train_loss:4.4} || training total accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                print(
                    f"training mask loss {train_loss_dict['mask_wear_loss']:4.4}, {train_loss_dict['mask_incorrect_loss']:4.4}, {train_loss_dict['mask_not_wear_loss']:4.4} || training mask accuracy {train_acc_dict['mask_wear_acc']:4.4%}, {train_acc_dict['mask_incorrect_acc']:4.4%}, {train_acc_dict['mask_not_wear_acc']:4.4%}\n"
                    f"training gender loss {train_loss_dict['male_loss']:4.4}, {train_loss_dict['female_loss']:4.4} || training gender accuracy {train_acc_dict['mask_not_wear_acc']:4.4%}, {train_acc_dict['female_acc'] :4.4%}\n"
                    f"training age loss {train_loss_dict['age_0_30_loss']:4.4}, {train_loss_dict['age_30_60_loss']:4.4}, {train_loss_dict['age_60_loss']:4.4} || training age accuracy {train_acc_dict['age_0_30_acc']:4.4%}, {train_acc_dict['age_30_60_acc']:4.4%}, {train_acc_dict['age_60_acc']:4.4%}\n"
                )

                logger.add_scalar(
                    "Train/loss", train_loss, epoch * len(train_loader) + idx
                )
                logger.add_scalar(
                    "Train/accuracy", train_acc, epoch * len(train_loader) + idx
                )

                for key, value in train_loss_dict.items():
                    logger.add_scalar(
                        "Train_cls/"+key, value, epoch * len(train_loader) + idx
                    )
                for key, value in train_acc_dict.items():
                    logger.add_scalar(
                        "Train_cls/"+key, value, epoch * len(train_loader) + idx
                    )

                loss_value = 0
                matches = 0

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            val_loss_dict = {
                'mask_wear_loss' : 0,
                'mask_incorrect_loss' : 0,
                'mask_not_wear_loss' : 0,
                'male_loss' : 0,
                'female_loss' : 0,
                'age_0_30_loss' : 0,
                'age_30_60_loss' : 0,
                'age_60_loss' : 0,
            }
            val_acc_dict = {
                'mask_wear_acc' : 0,
                'mask_incorrect_acc' : 0,
                'mask_not_wear_acc' : 0,
                'male_acc' : 0,
                'female_acc' : 0,
                'age_0_30_acc' : 0,
                'age_30_60_acc' : 0,
                'age_60_acc' : 0,
            }
            for val_batch in val_loader:
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                
                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                val_accloss = AccuracyLoss(labels, preds, outs, criterion)
                val_loss_cls, val_acc_cls = val_accloss.loss_acc(len(val_loader), len(val_loader))
                for key, value in val_loss_cls.items():
                    val_loss_dict[key] += value
                for key, value in val_acc_cls.items():
                    val_acc_dict[key] += value

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

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.sum(val_acc_items) / len(val_set)
            best_val_loss = min(best_val_loss, val_loss)
            if val_acc > best_val_acc:
                best_epoch = epoch
                print(
                    f"New best model for val accuracy : {val_acc:4.2%}! saving the best model.."
                )
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss,
                        'accuracy': val_acc,
                    }
                    , f"{save_dir}/best.pth")
                best_val_acc = val_acc
            torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss,
                        'accuracy': val_acc,
                    }
                    , f"{save_dir}/last_epoch{epoch:03d}.pth")


            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}"
            )
            print(
                f"[Val] mask loss {val_loss_dict['mask_wear_loss']:4.2}, {val_loss_dict['mask_incorrect_loss']:4.2}, {val_loss_dict['mask_not_wear_loss']:4.4} || training mask accuracy {val_acc_dict['mask_wear_acc']:4.2%}, {val_acc_dict['mask_incorrect_acc']:4.2%}, {val_acc_dict['mask_not_wear_acc']:4.2%}\n"
                f"[Val] gender loss {val_loss_dict['male_loss']:4.2}, {val_loss_dict['female_loss']:4.2} || training gender accuracy {val_acc_dict['mask_not_wear_acc']:4.2%}, {val_acc_dict['female_acc']:4.2%}\n"
                f"[Val] age loss {val_loss_dict['age_0_30_loss']:4.2}, {val_loss_dict['age_30_60_loss']:4.2}, {val_loss_dict['age_60_loss']:4.2} || training age accuracy {val_acc_dict['age_0_30_acc']:4.2%}, {val_acc_dict['age_30_60_acc']:4.2%}, {val_acc_dict['age_60_acc']:4.2%}\n"
            )
            
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_figure("results", figure, epoch)

            for key, value in val_loss_dict.items():
                    logger.add_scalar("Val_cls/"+key, value, epoch)
            for key, value in val_acc_dict.items():
                logger.add_scalar("Val_cls/"+key, value, epoch)
            print()

    ################## 
    os.rename(f"{save_dir}/best.pth",f"{save_dir}/best_epoch{best_epoch:03d}.pth")
    ##################

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
        "--epochs", type=int, default=1, help="number of epochs to train (default: 1)"
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
        default=os.environ.get("SM_CHANNEL_TRAIN", "../data/train/images"),
    )
    parser.add_argument(
        "--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "../model")
    )

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)
