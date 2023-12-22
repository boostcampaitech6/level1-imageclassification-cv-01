import argparse
import multiprocessing
import os
import glob
import pandas as pd
import torch
import torch.nn as nn

from importlib import import_module
from torch.utils.data import DataLoader
from dataset import TestDataset, MaskBaseDataset
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
    model_cls = getattr(import_module("model"), args.model)
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

def load_model_mul(saved_model, num_classes, device,model_name):
    """
    저장된 모델의 가중치를 로드하는 함수입니다.

    Args:
        saved_model (str): 모델 가중치가 저장된 디렉토리 경로
        num_classes (int): 모델의 클래수 수
        device (torch.device): 모델이 로드될 장치 (CPU 또는 CUDA)

    Returns:
        model (nn.Module): 가중치가 로드된 모델
    """
    # model_cls = getattr(import_module("model"), args.model)
    # model = model_cls()
    model_module = getattr(import_module("model"), args.model)  # default: BaseModel
    model_age = model_module(num_classes=3)
    model_mask = model_module(num_classes=3)
    model_gender = model_module(num_classes=2)

    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)

    # 모델 가중치를 로드한다.
    model_path = os.path.join(saved_model, model_name)#"best.pth")
    checkpoint = torch.load(model_path, map_location=device)

    age_state_dict = checkpoint.get("model_age", None)
    mask_state_dict = checkpoint.get("model_mask", None)
    gender_state_dict = checkpoint.get("model_gender", None)
            
    model_age.load_state_dict(age_state_dict)
    model_mask.load_state_dict(mask_state_dict)
    model_gender.load_state_dict(gender_state_dict)

    return model_age,model_mask,model_gender


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    모델 추론을 수행하는 함수

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
    if(args.model_mode=="multiple"):
        model_age,model_mask,model_gender = load_model_mul(model_dir, num_classes, device,args.model_file_name)
        model_age,model_mask,model_gender = model_age.to(device),model_mask.to(device),model_gender.to(device)
        model_age.eval()
        model_mask.eval()
        model_gender.eval()
    else:
        model=load_model(model_dir, num_classes, device,args.model_file_name)
        model=model.to(device)
        model.eval()

    # 이미지 파일 경로와 정보 파일을 읽어온다.
    img_root = os.path.join(data_dir, "images")
    info_path = os.path.join(data_dir, "info.csv")
    info = pd.read_csv(info_path)

    # 이미지 경로를 리스트로 생성한다.
    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            if(args.model_mode=="multiple"):
                logits_age,logits_mask,logits_gender = model_age(images),model_mask(images),model_gender(images)
                pred_age,pred_mask,pred_gender = logits_age.argmax(dim=-1),logits_mask.argmax(dim=-1),logits_gender.argmax(dim=-1)
                
                preds.extend(MaskBaseDataset.encode_multi_class(pred_mask,pred_gender,pred_age).cpu().numpy())
            elif(args.model_mode=="single_multiple"):
                logits_age,logits_mask,logits_gender = model(images)
                pred_age,pred_mask,pred_gender = logits_age.argmax(dim=-1),logits_mask.argmax(dim=-1),logits_gender.argmax(dim=-1)
                
                preds.extend(MaskBaseDataset.encode_multi_class(pred_mask,pred_gender,pred_age).cpu().numpy())
                
            else:
                logits=model(images)
                preds1=logits.argmax(dim=-1)
                
                preds.extend(preds1.cpu().numpy())

    ############################# 
    KST = timezone(timedelta(hours=9))
    current_time = datetime.now(KST).strftime("%Y-%m-%d_%H-%M-%S")
    #############################

    # 예측 결과를 데이터프레임에 저장하고 csv 파일로 출력한다.
    info["ans"] = preds
    save_path = os.path.join(output_dir, f"{current_time}.csv")
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == "__main__":
    # 커맨드 라인 인자를 파싱한다.
    parser = argparse.ArgumentParser()

    # 데이터와 모델 체크포인트 디렉터리 관련 인자
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="input batch size for validing (default: 1000)",
    )
    parser.add_argument(
        "--resize",
        nargs=2,
        type=int,
        default=(236,236),#(96, 128),
        help="resize size for image when you trained (default: (96,128))",
    )
    parser.add_argument(
        "--model", type=str, default="BaseModel", help="model type (default: BaseModel)"
    )
    parser.add_argument(
        "--model_mode",
        type=str,
        default="single", help="single/multiple/single_multiple",
    )

    # 컨테이너 환경 변수
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_EVAL", "../../../eval"),
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=os.environ.get("SM_CHANNEL_MODEL", "../../../models/exp154"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "../../../output"),
    )
    parser.add_argument( # 입력 안하면 best_epoch.pth를 사용한다.
        "--model_file_name",
        type=str,
        default=os.environ.get("SM_INPUT_MODEL", "best_epoch*.pth"),
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir
    model_file_name = args.model_file_name

    os.makedirs(output_dir, exist_ok=True)

    # 모델 추론을 수행한다.
    inference(data_dir, model_dir, output_dir, args)