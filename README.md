# Level1-Classification Competetion
🌟**CV-01**조 **Supershy**
성주희, 한주희, 정재웅, 김혜지, 류경엽, 임서현

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

#### Training

To train the model with your custom dataset, set the appropriate directories for the training images and model saving, then run the training script.
```
python train.py --data_dir /path/to/images --model_dir /path/to/model
```

#### Inference

For generating predictions with a trained model, provide directories for evaluation data, the trained model, and output, then run the inference script.
```
python inference.py --data_dir /path/to/images --model_dir /path/to/model --output_dir /path/to/model
```

