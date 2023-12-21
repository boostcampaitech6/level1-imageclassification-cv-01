## Training
- `--resume_from` : model의 checkpoint 불러오기
- `--seed` : data split을 위한 seed
- `--epochs` : epoch 수
- `--dataset` : 사용할 dataset class 이름
- `--augmentation` : 사용할 augmentation class 이름
- `--resize` : 이미지 resize 크기
- `--batch_size` : batch size
- `--valid_batch_size` : validation batch size
- `--model` : 사용할 model 이름
- `--optimizer` : 사용할 optimizer 이름
- `--lr` : learning rate
- `--val_ratio` : validation set의 비율
- `--criterion` : loss function 이름
    - `focal` : focal loss
    - `cross_entropy` : cross entropy loss
    - `label_smoothing` : label smoothing loss
    - `f1` : f1 loss
    - `MSE` : mean squared error loss
- `--lr_decay_step` : learning rate decay step
- `--log_interval` : logging interval
- `--name` : model 저장할 때 사용할 이름
- `--data_dir` : data가 있는 directory
- `--model_dir` : model을 저장할 directory
- `--use_stratified_kfold` : stratified kfold 사용 여부
- `--num_splits` : kfold의 k
- `--current_fold` : 현재 몇번째 fold인지


## Inference
- `--batch_size` : batch size
- `--resize` : 이미지 resize 크기
- `--model` : 사용할 model 이름
- `--model_mode`: single or single_multiple
- `--data_dir` : data가 있는 directory
- `--model_dir` : model을 저장할 directory
- `--output_dir`: output을 저장할 directory
- `--model_file_name` : model file name


## Ensemble
hard voting
- `--file_dir` : csv 파일이 있는 디렉토리 경로
- `--csv1` : csv 파일 이름
- `--csv2` : csv 파일 이름
- `--csv3` : csv 파일 이름
- `--save_dir` : 저장할 디렉토리 경로

soft voting
- `--batch_size` : batch size
- `--resize` : 이미지 resize 크기
- `--models` : 사용할 model 이름
- `--model_dir` : model을 저장할 directory
- `--model_files` : model file name
- `--data_dir` : data가 있는 directory
- `--save_dir` : 저장할 directory