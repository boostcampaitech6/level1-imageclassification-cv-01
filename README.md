# CV 기초대회 베이스라인 코드

## Project Structure

```
${PROJECT}
├── dataset.py
├── inference.py
├── loss.py
├── model.py
├── README.md
├── requirements.txt
├── sample_submission.ipynb
└── train.py
```

- dataset.py : This file contains dataset class for model training and validation
- inference.py : This file used for predict the model
- loss.py : This file defines the loss functions used during training
- model.py : This file defines the model
- README.md
- requirements.txt : contains the necessary packages to be installed
- sample_submission.ipynb : an example notebook for submission
- train.py : This file used for training the model

## Getting Started

### Stteing up Vitual Enviornment

1. Install `virtualenv` if you haven't yet:

```
pip install virtualenv
```

2. Create a virtual environment in the project directory

```
cd ${PROJECT}
python -m venv /path/to/venv
```

3. Activate the virtual environment

- On Windows:

```
.\venv\Scripts\activate
```

- On Unix or MacOS:

```
source venv/bin/activate
```

4. To deactivate and exit the virtual environment, simply run:

```
deactivate
```

### Install Requirements

To Insall the necessary packages liksted in `requirements.txt`, run the following command while your virtual environment is activated:


```
pip install -r requirements.txt
```

### Usage

#### Training

To train the model with your custom dataset, set the appropriate directories for the training images and model saving, then run the training script.

```
SM_CHANNEL_TRAIN=/path/to/images SM_MODEL_DIR=/path/to/model python train.py
```

or 

```
python train.py --data_dir /path/to/images --model_dir /path/to/model
```

#### Inference

For generating predictions with a trained model, provide directories for evaluation data, the trained model, and output, then run the inference script.

```
SM_CHANNEL_EVAL=/path/to/images SM_CHANNEL_MODEL=/path/to/model SM_OUTPUT_DATA_DIR=/path/to/output python inference.py
```

or 

```
python inference.py --data_dir /path/to/images --model_dir /path/to/model --output_dir /path/to/model
```


# use vit 16
## data location
data는 그 level1-imageclassification-cv-01에 뒀음.
근데 거의 절대경로 찍어서 상관없을듯 

## train command 
### data_dir, modelname, model_dir, resize, creterion, epoch, dataset, usekfold, kfold_num, current_fold
python train.py --data_dir /data/ephemeral/home/data/train/images --model VITmodel --model_dir ./vit16_k10_b64 --resize 224 224 --criterion focal --epochs 15 --dataset MaskSplitByProfileDataset --use_stratified_kfold --num_splits 5 --current_fold 0

## inference command
### modelname, data_dir, model_dir
python inference.py --model VITmodel --model_dir /data/ephemeral/home/level1-imageclassification-cv-01/vit_base_patch16_224/vit16_k5_b64/exp/re --data_dir /data/ephemeral/home/data/eval --resize 224 224

## result
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
