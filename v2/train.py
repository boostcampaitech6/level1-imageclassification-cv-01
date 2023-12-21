import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion

from accuracy_loss_print import AccuracyLoss, AgeBoundaryAcc
from collections import OrderedDict



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
    
    #mixup
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        #print("Mixup is activated!")
        mixup_fn = Mixup(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.label_smoothing, num_classes=num_classes)

    # -- data_loader
    train_set, val_set = dataset.split_dataset()
    weights=dataset.weights
    sample_weights=[weights[train_set[i][1]] for i in range(len(train_set))]
    print(len(sample_weights))
    sampler = torch.utils.data.sampler.WeightedRandomSampler(sample_weights, 9000,replacement=True)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        # shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
        sampler=sampler,
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
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.label_smoothing > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    else:
        criterion = create_criterion(args.criterion)
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

    best_val_age60_acc = 0
    best_epoch_age60 = 0
    
    start_epoch = 0
    
    if args.resume_from:
        model_data = torch.load(args.resume_from)
        model.load_state_dict(model_data['model_state_dict'])
        optimizer.load_state_dict(model_data['optimizer_state_dict'])
        start_epoch = model_data['epoch'] + 1
    
    
    for epoch in range(start_epoch, args.epochs):
        torch.cuda.empty_cache()
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in enumerate(train_loader):
            inputs, labels = train_batch

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            if mixup_fn is not None:
                inputs, labels = mixup_fn(inputs, labels)
                

            optimizer.zero_grad()

            outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)
            
            loss = criterion(outs, labels)

            loss.backward()
            optimizer.step()
            
            loss_value += loss.item()

            if mixup_fn is None:
                matches += (preds == labels).sum().item()
                train_accloss = AccuracyLoss(labels, preds, outs, criterion)

                
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                current_lr = get_lr(optimizer)
                
                if  mixup_fn is not None:
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training total loss {train_loss:4.4} || lr {current_lr}"
                    )                    
                    
                else:
                    train_acc = matches / args.batch_size / args.log_interval
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
                    "Train/accuracy", train_acc, epoch * len(train_loader) + idx
                    )

                logger.add_scalar(
                    "Train/loss", train_loss, epoch * len(train_loader) + idx
                )

                if mixup_fn is None:
                    for key, value in train_loss_dict.items():
                        logger.add_scalar(
                            "Train_cls/"+key, value, epoch * len(train_loader) + idx
                        )
                    for key, value in train_acc_dict.items():
                        logger.add_scalar(
                            "Train_cls/"+key, value, epoch * len(train_loader) + idx
                        )

                logger.add_scalars('Train_cls/Mask Loss', dict(OrderedDict(list(train_loss_dict.items())[:3])), epoch * len(train_loader) + idx)
                logger.add_scalars('Train_cls/Gender Loss', dict(OrderedDict(list(train_loss_dict.items())[3:5])), epoch * len(train_loader) + idx)
                logger.add_scalars('Train_cls/Age Loss', dict(OrderedDict(list(train_loss_dict.items())[5:])), epoch * len(train_loader) + idx)
                logger.add_scalars('Train_cls/Mask Accuracy', dict(OrderedDict(list(train_acc_dict.items())[:3])), epoch * len(train_loader) + idx)
                logger.add_scalars('Train_cls/Gender Accuracy', dict(OrderedDict(list(train_acc_dict.items())[3:5])), epoch * len(train_loader) + idx)
                logger.add_scalars('Train_cls/Age Accuracy', dict(OrderedDict(list(train_acc_dict.items())[5:])), epoch * len(train_loader) + idx)

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
            acc_dict = {
            'late_20s_acc' : 0,
            'late_20s_MIDDLE' : 0,
            'late_50s_acc' : 0,
            'late_50s_OLD' : 0,
            'age_60s_acc' : 0,
            'age_60s_MIDDLE' : 0,

            }
        
            for val_batch,indices in zip(val_loader, val_loader.batch_sampler):
                inputs, labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                ages = [dataset.age[i] for i in indices]

                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                criterion = create_criterion(args.criterion)
                loss_item = criterion(outs, labels).item()
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item)

                val_accloss = AccuracyLoss(labels, preds, outs, criterion)
                val_loss_cls, val_acc_cls = val_accloss.loss_acc(len(val_set), len(val_loader))
                for key, value in val_loss_cls.items():
                    val_loss_dict[key] += value
                for key, value in val_acc_cls.items():
                    val_acc_dict[key] += value


                val_accloss = AccuracyLoss(labels, preds, outs, criterion)
                val_loss_cls, val_acc_cls = val_accloss.loss_acc(len(val_loader), len(val_loader))
                for key, value in val_loss_cls.items():
                    val_loss_dict[key] += value
                for key, value in val_acc_cls.items():
                    val_acc_dict[key] += value

                last_acc = AgeBoundaryAcc(labels,preds,ages)
                last_acc = last_acc.cal_acc(len(val_loader))
                for key, value in last_acc.items():
                    acc_dict[key] += value

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
                    , f"{save_dir}/last.pth")
            
            if val_acc_dict['age_60_acc'] > best_val_age60_acc:
                best_epoch_age60 = epoch
                best_val_age60_acc = val_acc_dict['age_60_acc']
                torch.save(
                    {
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': val_loss,
                        'accuracy': val_acc,
                    }
                    , f"{save_dir}/best_age60.pth")


            print(
                f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2} || "
                f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}\n"
            )
            print(
                f"[Val] mask loss {val_loss_dict['mask_wear_loss']:4.2}, {val_loss_dict['mask_incorrect_loss']:4.2}, {val_loss_dict['mask_not_wear_loss']:4.4} || training mask accuracy {val_acc_dict['mask_wear_acc']:4.2%}, {val_acc_dict['mask_incorrect_acc']:4.2%}, {val_acc_dict['mask_not_wear_acc']:4.2%}\n"
                f"[Val] gender loss {val_loss_dict['male_loss']:4.2}, {val_loss_dict['female_loss']:4.2} || training gender accuracy {val_acc_dict['mask_not_wear_acc']:4.2%}, {val_acc_dict['female_acc']:4.2%}\n"
                f"[Val] age loss {val_loss_dict['age_0_30_loss']:4.2}, {val_loss_dict['age_30_60_loss']:4.2}, {val_loss_dict['age_60_loss']:4.2} || training age accuracy {val_acc_dict['age_0_30_acc']:4.2%}, {val_acc_dict['age_30_60_acc']:4.2%}, {val_acc_dict['age_60_acc']:4.2%}\n"
            )
            print(
                f"[Val] Late 20s YOUNG  {acc_dict['late_20s_acc']:4.2%} || MIDDLE(Error) {acc_dict['late_20s_MIDDLE']:4.2%}\n"
                f"[Val] Late 50s MIDDLE {acc_dict['late_50s_acc']:4.2%} || OLD(Error)    {acc_dict['late_50s_OLD']:4.2%}\n"
                f"[Val] Age  60s OLD    {acc_dict['age_60s_acc']:4.2%} || MIDDLE(Error) {acc_dict['age_60s_MIDDLE']:4.2%}\n"
            )
            
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/accuracy", val_acc, epoch)
            logger.add_figure("results", figure, epoch)

            # for key, value in val_loss_dict.items():
            #         logger.add_scalar("Val_cls/"+key, value, epoch)
            # for key, value in val_acc_dict.items():
            #     logger.add_scalar("Val_cls/"+key, value, epoch)
            for key, value in acc_dict.items():
                if key in ['late_20s_acc','late_50s_acc','age_60s_acc']:
                    logger.add_scalar("Val_ageboundary/"+key, value, epoch)


            logger.add_scalars('Val_cls/Mask Loss', dict(OrderedDict(list(val_loss_dict.items())[:3])), epoch * len(train_loader) + idx)
            logger.add_scalars('Val_cls/Gender Loss', dict(OrderedDict(list(val_loss_dict.items())[3:5])), epoch * len(train_loader) + idx)
            logger.add_scalars('Val_cls/Age Loss', dict(OrderedDict(list(val_loss_dict.items())[5:])), epoch * len(train_loader) + idx)
            logger.add_scalars('Val_cls/Mask Accuracy', dict(OrderedDict(list(val_acc_dict.items())[:3])), epoch * len(train_loader) + idx)
            logger.add_scalars('Val_cls/Gender Accuracy', dict(OrderedDict(list(val_acc_dict.items())[3:5])), epoch * len(train_loader) + idx)
            logger.add_scalars('Val_cls/Age Accuracy', dict(OrderedDict(list(val_acc_dict.items())[5:])), epoch * len(train_loader) + idx)


            print()

    ################## 
    os.rename(f"{save_dir}/best.pth",f"{save_dir}/best_epoch{best_epoch:03d}.pth")
    os.rename(f"{save_dir}/last.pth",f"{save_dir}/last_epoch{args.epochs-1:03d}.pth")
    os.rename(f"{save_dir}/best_age60.pth",f"{save_dir}/best_age60_epoch{best_epoch_age60:03d}.pth")
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
        default=[236,236],#[128, 96],
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
        "--optimizer", type=str, default="AdamW", help="optimizer type (default: AdamW)"
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
        default="f1",
        help="criterion type (default: f1)",
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
    
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--label_smoothing', type=float, default=0.0)

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)