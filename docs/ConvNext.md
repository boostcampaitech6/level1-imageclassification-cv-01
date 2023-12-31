# Use ConvNext Tiny

## train command
#### data_dir, modelname, model_dir, resize, creterion, epoch, dataset, epochs, optimizer
```
python train.py \
       --data_dir ~/data/train/images \
       --batch_size 32 \
       --valid_batch_size 32 \
       --model ConvNextModel \
       --resize 236 236 \
       --epochs 10 \
       --dataset MaskSplitByProfileDataset \
       --optimizer Adam
```

## inference command
#### modelname, data_dir, model_dir
```
python inference.py 
       --model ConvNextModel \
       --model_dir ~/ConvNextModel/exp \
       --data_dir ~/data/eval \
       --resize 236 236 \
       --model_file_name best_model.pth \
       --batch_size 32
```

## Result
#### ConvNextModel_batch64_epoch10_Basedataset
|F1-score|Accuracy|
|:---:|:---:|
|0.7583|81.2698|