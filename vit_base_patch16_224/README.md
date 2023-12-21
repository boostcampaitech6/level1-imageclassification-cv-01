<<<<<<< HEAD
# How to use vit 16
=======
# use vit 16

>>>>>>> 5e6f37d261f68bfd72e52ad98732f9f88bcb970a

## data location
data는 그 level1-imageclassification-cv-01에 뒀음.
근데 거의 절대경로 찍어서 상관없을듯 

## train command 
python train.py --data_dir /home/level1-imageclassification-cv-01/data/train/images --model VITmodel --model_dir ./exp2 --num_classes 18 --resize 224 224 --epochs 10 --dataset MaskBaseDataset

## inference command
python inference.py --model VITmodel --model_dir ./exp2 --data_dir /home/level1-imageclassification-cv-01/data/eval --resize 224 224

## result
<<<<<<< HEAD
### vit16_batch32_epoch10_Basedataset
[val] acc : 76.01%, loss : 0.13 || best acc : 76.01%, best loss : 0.13
### vit16_batch64_epoch40_splitbyprofiledataset
[Val] acc : 71.24%, loss: 0.46 || best acc : 71.80%, best loss: 0.32
### vit16_batch32_epoch40_splitbyprofiledataset
[val] acc : 71.01%, loss : 0.38 || best acc : 71.01%, best loss : 0.33
=======
[val] acc : 76.01%, loss : 0.13 || best acc : 76.01%, best loss : 0.13
>>>>>>> 5e6f37d261f68bfd72e52ad98732f9f88bcb970a
