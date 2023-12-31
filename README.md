# Level1-Classification Competetion
🌟**CV-01**조🌟 **Supershy**
성주희, 한주희, 정재웅, 김혜지, 류경엽, 임서현

## Project Structure

```
${PROJECT}
├── eda
│   ├── data_eda.ipynb
├── main
│   ├── README.md
│   ├── accuracy_loss_print.py
│   ├── dataset.py
│   ├── inference.py
│   ├── loss.py
│   ├── model.py
│   ├── requiremets.txt
│   ├── train.py
│   ├── train_single_multiple.py
│   ├── hard_voting.py
│   └── soft_voting.py
├── README.md
└── requiremets.txt
```

- dataset.py : This file contains dataset class for model training and validation
- inference.py : This file used for predict the model
- loss.py : This file defines the loss functions used during training
- model.py : This file defines the model
- README.md
- requirements.txt : contains the necessary packages to be installed
- train.py : This file used for training the model

## Code Structure Description 

**Data EDA**

 - Use MaskSplitByProfileDataset
 - Downsampling
 - Stratified Kfold

**Model**
 - Ensemble `Soft Voting`
 - Learn additional Fine Tuning based on the public pretrained model
	 `EfficientNet` + `ConvNext` + `ConvNext(Stratified Kfold)`


## Getting Started

### Setting up Vitual Enviornment

1. Initialize and update the server
	```
    su -
    source .bashrc
    ```

2. Create and Activate a virtual environment in the project directory

	```
    conda create -n env python=3.8
    conda activate env
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
[Description of all arguments]()

#### Training

To train the model with your custom dataset, set the appropriate directories for the training images and model saving, then run the training script.
- **single model**

```
python train.py --data_dir /path/to/images --model_dir /path/to/model --model MODEL_NAME
```

- **single multiple model**

```
python train_single_multiple.py --data_dir /path/to/images --model_dir /path/to/model --model MODEL_NAME
```


#### Inference

For generating predictions with a trained model, provide directories for evaluation data, the trained model, and output, then run the inference script.
- **single model**
```
python inference.py --data_dir /path/to/images --model_dir /path/to/model --output_dir /path/to/model --model MODEL_NAME
```
- **single multiple model**
```
python inference.py --data_dir /path/to/images --model_dir /path/to/model --output_dir /path/to/model --model_mode single_multiple --model MODEL_NAME
```

#### Ensemble
- **ensemble (hard voting)**
```
python hard_voting.py --file_dir ./csv --csv1 file1.csv --csv2 file2.csv --csv3 file3.csv
```

- **ensemble (soft voting)**
```
python soft_voting.py --models MODEL_NAME1 MODEL_NAME2 MODEL_NAME3 --model_dir ./checkpoint --model_files file1.pth file2.pth file3.pth --data_dir ./data/eval
```


### [Arguments](./docs/arguments.md)

## Model
- [ConvNext Tiny](./docs/ConvNext.md)
- [EfficientNet v2 small](./docs/EfficientNet_v2_small.md)
- [ViT16](./docs/ViT16.md)

## Dataloader
- [oversampling with data augmentation](./docs/data_sampling.md)
