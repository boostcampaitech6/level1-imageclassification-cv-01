# Use ViT 16
## data location
data는 그 level1-imageclassification-cv-01에 뒀음.
근데 거의 절대경로 찍어서 상관없을듯 

## train command 
### data_dir, modelname, model_dir, resize, creterion, epoch, dataset, usekfold, kfold_num, current_fold
```
python train.py \
       --data_dir /~data/train/images \
       --model VITmodel \
       --model_dir ./vit16_k10_b64 \
       --resize 224 224 \
       --criterion focal \
       --epochs 15 \
       --dataset MaskSplitByProfileDataset \
       --use_stratified_kfold \
       --num_splits 5 \
       --current_fold 0
```

## inference command
### modelname, data_dir, model_dir

```
python inference.py \
       --model VITmodel \
       --model_dir /~level1-imageclassification-cv-01/vit_base_patch16_224/vit16_k5_b64/exp/re \
       --data_dir /~data/eval \
       --resize 224 224
```

## Result
### vit16_batch32_epoch10_Basedataset
[val] acc : 76.01%, loss : 0.13 || best acc : 76.01%, best loss : 0.13
### vit16_batch64_epoch40_splitbyprofiledataset
[Val] acc : 71.24%, loss: 0.46 || best acc : 71.80%, best loss: 0.32
### vit16_batch32_epoch40_splitbyprofiledataset
[val] acc : 71.01%, loss : 0.38 || best acc : 71.01%, best loss : 0.33
### vit16_batch64_epoch15_5kfold
[Val] acc : 66.51%, loss: 0.38 || best acc : 67.28%, best loss: 0.24

## best 2
### just best result - vit16_batch32_epoch10_Basedataset
[val] acc : 76.01%, loss : 0.13 || best acc : 76.01%, best loss : 0.13
### useful StratifiedKfold - vit16_batch64_epoch15_5kfold
[Val] acc : 66.51%, loss: 0.38 || best acc : 67.28%, best loss: 0.24
