# Use EfficientNet_v2_s

## Data Directory Setting
Set the `Data dir`
```
export SM_CHANNEL_TRAIN="ADD DATASET PATH"/train/images
export SM_CHANNEL_EVAL="ADD DATASET PATH"/eval
export SM_MODEL_DIR="ADD MODEL DIR PATH TO SAVE THE MODEL"
export SM_CHANNEL_MODEL="ADD MODEL DIR PATH TO LOAD THE MODEL"
export SM_OUTPUT_DATA_DIR="ADD RESULT CSV FILE DIR TO SAVE"
```

## Training
```
cd ~/main
python train.py \
       --epochs 15 \
       --dataset MaskSplitByProfileDataset \
       --resize 224 224 \
       --model MyModel_efficient_v2_s \
       --optimizer AdamW \
       --lr 0.00001 \
       --criterion crossentropy \
       --augmentation BaseAugmentation_efficientNet
```
 
 
## Testing
```
python inference.py \
       --model MyModel_efficient_v2_s \
       --resize 224 224 
```
You can view CSV files in folders stored in the SM_OUTPUT_DATA_DIR variable.