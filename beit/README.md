# Fine-tuning BEiT-3 on Mask Dataset (Image Classification)

Mask dataset 구조

```
/path/to/mask_data/
  train/
    images/
      profile1/
        incorrect_mask.jpg/
        mask1.jpg
        mask2.jpg
        mask3.jpg
        mask4.jpg
        mask5.jpg
        normal.jpg
      profile2/
        incorrect_mask.jpg
        mask1.jpg
        mask2.jpg
        mask3.jpg
        mask4.jpg
        mask5.jpg
        normal.jpg
    train.csv
  eval/
    images/
      img1.jpg
      img2.jpg
    info.csv
```

Profile 기준으로 Train data와 Validation data로 구분
Imagenet을 이용한 finetuning과 같은 방식을 이용하기 위해 데이터 저장 구조를 아래와 같이 맞춘다
해당 구조는 torchvision의 [datasets.ImageFolder](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html)를 따른다

```
/path/to/split_data/
  train/
    class1/
      img1.jpg
    class2/
      img2.jpg
  val/
    class1/
      img3.jpg
    class2/
      img4.jpg
```
아래 코드를 실행해서 위의 데이터를 구성한다

```
from datasets import MaskDataset

MaskDataset.split_dataset(
    data_dir = "/path/to/your_data/train",
    output_dir = "/path/to/your_split_data",
    val_ratio = (Default)0.2
)
```

이후, 다음 코드를 통해 index json 파일을 만든다
tokenize를 위해 사용한 sentencepiece model은 [beit3.spm](https://conversationhub.blob.core.windows.net/beit-share-public/beit3/sentencepiece/beit3.spm?sv=2021-10-04&st=2023-06-08T11%3A16%3A02Z&se=2033-06-09T11%3A16%3A00Z&sr=c&sp=r&sig=N4pfCVmSeq4L4tS8QbrFVsX6f6q844eft8xSuXdxU48%3D) 이다.

```
from datasets import MaskDataset

MaskDataset.make_dataset_index(
    train_data_path = "/path/to/your_split_data/train",
    val_data_path = "/path/to/your_split_data/val",
    index_path = "/path/to/your_data"
)
```



마지막으로 아래 명령어를 통해 run_beit3_finetuning.py를 실행시켜 학습을 진행한다

```bash       
python -m torch.distributed.launch --nproc_per_node=8 run_beit3_finetuning.py \
        --model beit3_base_patch16_224 \
        --task mask \
        --batch_size 128 \
        --layer_decay 0.65 \
        --lr 7e-4 \
        --update_freq 1 \
        --epochs 50 \
        --warmup_epochs 5 \
        --drop_path 0.15 \
        --sentencepiece_model /your_beit3_model_path/beit3.spm \
        --finetune /your_beit3_model_path/beit3_base_patch16_224.pth \
        --data_path /path/to/your_data \
        --output_dir /path/to/save/your_model \
        --log_dir /path/to/save/your_model/log \
        --weight_decay 0.05 \
        --seed 83 \
        --save_ckpt_freq 5 \
        --dist_eval \
        --mixup 0.8 \
        --cutmix 1.0
```
