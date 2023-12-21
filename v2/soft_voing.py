import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch

from dataset import TestDataset, MaskBaseDataset

import glob
from datetime import datetime, timezone, timedelta



def load_model(saved_model, num_classes, device, model_name):
    """
    저장된 모델의 가중치를 로드하는 함수입니다.

    Args:
        saved_model (str): 모델 가중치가 저장된 디렉토리 경로
        num_classes (int): 모델의 클래수 수
        device (torch.device): 모델이 로드될 장치 (CPU 또는 CUDA)

    Returns:
        model (nn.Module): 가중치가 로드된 모델
    """
    model_cls = getattr(import_module("model"), args.models)
    model = model_cls(num_classes=num_classes)

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    # 모델 가중치를 로드한다.
    ####################
    model_path = os.path.join(saved_model, model_name)
    select_model_path = glob.glob(model_path)
    model.load_state_dict(torch.load(select_model_path[-1], map_location=device)['model_state_dict'])
    ####################

    return model


def voting(data_dir, model_dir, output_dir, args):
    """
    soft한 방식으로 ensemble 하는 함수

    Args:
        data_dir (str): 테스트 데이터가 있는 디렉토리 경로
        model_dir (str): 모델 가중치가 저장된 디렉토리 경로
        output_dir (str): 결과 CSV를 저장할 디렉토리 경로
        args (argparse.Namespace): 커맨드 라인 인자
    
    Returns:
        None
    """
    # CUDA를 사용할 수 있는지 확인
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # 클래스의 개수를 설정한다. (마스크, 성별, 나이의 조합으로 18)
    num_classes = MaskBaseDataset.num_classes  # 18
    
    model_1 = load_model(model_dir, num_classes, device, args.model_files[0]).to(device)
    model_2 = load_model(model_dir, num_classes, device, args.model_files[1]).to(device)
    model_3 = load_model(model_dir, num_classes, device, args.model_files[2]).to(device)
    model_1.eval()
    model_2.eval()
    model_3.eval()

    # 이미지 파일 경로와 정보 파일을 읽어온다.
    img_root = os.path.join(data_dir, "images")
    info_path = os.path.join(data_dir, "info.csv")
    info = pd.read_csv(info_path)

    # 이미지 경로를 리스트로 생성한다.
    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset_1 = TestDataset(img_paths, args.resize[:2])
    dataset_2 = TestDataset(img_paths, args.resize[2:4])
    dataset_3 = TestDataset(img_paths, args.resize[4:])

    loader_1 = torch.utils.data.DataLoader(
        dataset_1,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )
    loader_2 = torch.utils.data.DataLoader(
        dataset_2,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )
    loader_3 = torch.utils.data.DataLoader(
        dataset_3,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for images_1, images_2, images_3 in zip(loader_1, loader_2, loader_3):
            images_1 = images_1.to(device)
            images_2 = images_2.to(device)
            images_3 = images_3.to(device)
            
            pred_1 = model_1(images_1)
            pred_2 = model_2(images_2)
            pred_3 = model_3(images_3)

            pred = pred_1 * 0.4 + pred_2 * 0.3 + pred_3 * 0.3
            pred = pred.argmax(dim=-1)

            preds.extend(pred.cpu().numpy())

    ############################# 
    KST = timezone(timedelta(hours=9))
    current_time = datetime.now(KST).strftime("%Y-%m-%d_%H-%M-%S")
    #############################

    # 예측 결과를 데이터프레임에 저장하고 csv 파일로 출력한다.
    info["ans"] = preds
    save_path = os.path.join(output_dir, f"soft_voting_{current_time}.csv")
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    """
    batch_size : 배치 사이즈
    resize : 이미지 사이즈
    models : 모델 이름(1등, 2등, 3등)
    model_dir : checkpoint 경로
    model_files : checkpoint 파일 이름
    data_dir : 데이터 경로
    save_dir : 저장할 경로
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="input batch size for validing (default: 1000)",
    )
    parser.add_argument(
        "--resize",
        nargs='+',
        type=int,
        default=[236, 236, 224, 224, 236, 236],
        help="resize size for image when you trained (default: (96, 128))",
    )
    parser.add_argument(
        "--models", nargs='+', type=str,
        default=["ConvNextModel", "MyModel_efficient_v2_s", "ViT"],
        help="model type (default: [\"ConvNextModel\", \"MyModel_efficient_v2_s\", \"ViT\"])"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_MODEL", "./checkpoint/"),
    )
    parser.add_argument(
        "--model_files", nargs='+', type=str,
        default=["convnext.pth", "efficient.pth", "vit.pth"]
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_EVAL", "../data/eval"),
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./voting",
        help="save type (default: ./voting)"
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    save_dir = args.save_dir
    
    os.makedirs(args.save_dir, exist_ok=True)

    voting(data_dir, model_dir, save_dir, args)