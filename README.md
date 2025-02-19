# Contrastive Model for Ataxia Severity Assessment

This repository contains the official implementation of the paper "Contrastive Learning Model for Wearable-based Ataxia Assessment."

## Acknowledgments
The model code is based on the implementation of the paper [TS2Vec: Towards Universal Representation of Time Series (AAAI 22)](https://github.com/zhihanyue/ts2vec), licensed under the MIT License.

## Requirements

* Python 3.9
* numpy==1.21.2
* pandas==1.3.5
* PyYAML==6.0
* scikit_learn==1.0.2
* scipy==1.7.3
* seaborn==0.11.2
* torch==1.11.0


To install the dependencies, run:


```bash

pip install -r requirements.txt

```

## Data

The dataset is not publicly available due to privacy concerns and the requirement for IRB approval. 

Please contact us if you are interested in accessing the dataset.


## Usage


###Training the Contrastive Learning Model
Run the following command to train the contrastive learning model:


```
python train.py 
```

You can customize the training settings by modifying the configuration file in ```config.yaml```.



### Evaluation on Downstream Tasks
To evaluate the trained model on downstream tasks, such as ataxia severity assessment and classification of ataxia vs. healthy subjects, refer to: 

```
Downstream_task.ipynb
```
