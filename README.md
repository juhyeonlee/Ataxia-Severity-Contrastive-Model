# Contrastive Model for Ataxia Severity Assessment

This repository contains the official implementation of the paper "Contrastive Learning Model for Wearable-based Ataxia Assessment."

## Acknowledgments
The model code is based on the implementation of the paper [TS2Vec: Towards Universal Representation of Time Series (AAAI 22)](https://github.com/zhihanyue/ts2vec), licensed under the MIT License.

[//]: # ()
[//]: # ()
[//]: # (## Requirements)

[//]: # ()
[//]: # (The recommended environment setup for TS2Vec is as follows:)

[//]: # ()
[//]: # (* Python 3.8)

[//]: # (* torch==1.8.1)

[//]: # (* scipy==1.6.1)

[//]: # (* numpy==1.19.2)

[//]: # (* pandas==1.0.1)

[//]: # (* scikit-learn==0.24.2)

[//]: # (* statsmodels==0.12.2)

[//]: # (* Bottleneck==1.3.2)

[//]: # ()
[//]: # (To install the dependencies, run:)

[//]: # ()
[//]: # (```bash)

[//]: # (pip install -r requirements.txt)

[//]: # (```)

## Data

The dataset is not publicly available due to privacy concerns and the requirement for IRB approval. 

Please contact us if you are interested in accessing the dataset.

[//]: # ()
[//]: # (## Usage)

[//]: # ()
[//]: # (To train and evaluate TS2Vec on a dataset, run the following command:)

[//]: # ()
[//]: # (```train & evaluate)

[//]: # (python train.py <dataset_name> <run_name> --loader <loader> --batch-size <batch_size> --repr-dims <repr_dims> --gpu <gpu> --eval)

[//]: # (```)

[//]: # (The detailed descriptions about the arguments are as following:)

[//]: # (| Parameter name | Description of parameter |)

[//]: # (| --- | --- |)

[//]: # (| dataset_name | The dataset name |)

[//]: # (| run_name | The folder name used to save model, output and evaluation metrics. This can be set to any word |)

[//]: # (| loader | The data loader used to load the experimental data. This can be set to `UCR`, `UEA`, `forecast_csv`, `forecast_csv_univar`, `anomaly`, or `anomaly_coldstart` |)

[//]: # (| batch_size | The batch size &#40;defaults to 8&#41; |)

[//]: # (| repr_dims | The representation dimensions &#40;defaults to 320&#41; |)

[//]: # (| gpu | The gpu no. used for training and inference &#40;defaults to 0&#41; |)

[//]: # (| eval | Whether to perform evaluation after training |)

[//]: # ()
[//]: # (&#40;For descriptions of more arguments, run `python train.py -h`.&#41;)

[//]: # ()
[//]: # (After training and evaluation, the trained encoder, output and evaluation metrics can be found in `training/DatasetName__RunName_Date_Time/`. )

[//]: # ()
[//]: # (**Scripts:** The scripts for reproduction are provided in `scripts/` folder.)

[//]: # ()
[//]: # ()
[//]: # (## Code Example)

[//]: # ()
[//]: # (```python)

[//]: # (from contrastive_model import ContrastiveModel)

[//]: # (import datautils)

[//]: # ()
[//]: # (# Load the ECG200 dataset from UCR archive)

[//]: # (train_data, train_labels, test_data, test_labels = datautils.load_UCR&#40;'ECG200'&#41;)

[//]: # (# &#40;Both train_data and test_data have a shape of n_instances x n_timestamps x n_features&#41;)

[//]: # ()
[//]: # (# Train a TS2Vec model)

[//]: # (model = ContrastiveModel&#40;)

[//]: # (    input_dims=1,)

[//]: # (    device=0,)

[//]: # (    output_dims=320)

[//]: # (&#41;)

[//]: # (loss_log = model.fit&#40;)

[//]: # (    train_data,)

[//]: # (    verbose=True)

[//]: # (&#41;)

[//]: # ()
[//]: # (# Compute timestamp-level representations for test set)

[//]: # (test_repr = model.encode&#40;test_data&#41;  # n_instances x n_timestamps x output_dims)

[//]: # ()
[//]: # (# Compute instance-level representations for test set)

[//]: # (test_repr = model.encode&#40;test_data, encoding_window='full_series'&#41;  # n_instances x output_dims)

[//]: # ()
[//]: # (# Sliding inference for test set)

[//]: # (test_repr = model.encode&#40;)

[//]: # (    test_data,)

[//]: # (    causal=True,)

[//]: # (    sliding_length=1,)

[//]: # (    sliding_padding=50)

[//]: # (&#41;  # n_instances x n_timestamps x output_dims)

[//]: # (# &#40;The timestamp t's representation vector is computed using the observations located in [t-50, t]&#41;)

[//]: # (```)
