# Feature Engineering and Predictive Modeling on Stock Market Return
This repo is dedicated to showcase the data engineering and predictive modeling process of stock market return.

## Folder structure
```
576_PROJECT/
├── data_engineer/              # This folder that executes data engineering
│   │                           # and store feature names selected by the algorithms
│   ├── final_process.py
│   ├── find_intersection.py
│   ├── pca_selection.csv
│   ├── selection_pca.py
│   └── shap_selection.csv
│
├── datasets/                   # This folder stores raw, intermediate and final datasets
│   ├── test.csv
│   ├── train_2000.csv
│   ├── train_final.csv
│   ├── train_intersect.csv
│   └── train.csv
│
├── model_trainers/             # This folder contains all model trainers
│   ├── baselines/
│   └── proposed/
│       ├── mlp.py
│       └── patchtst.py
│
├── saved_models/               # This folder contains saved model weights
│
├── README.md
├── requirements.txt
└── test.py                     # Testing script for predictive models
```

## Folder explanation
Under the directory `data_engineer/` are the data engineering scripts. Those scripts are meant for outputting the feature names selected by different data engineering algorithms. The final dataset is stored under directory `datasets/`

Under the directory `datasets/` are the raw, intermediate and final datasets for trainig and testing. `train_final.csv` is the final dataset proper for training and testing.

Under the directory `model_trainers/` are all model trainers. The sub-directory `baselines/` contains all baseline models for baseline comparison, and the sub-directory `proposed/` contains the model trainers for our proposed method.

## Script execution
### Preparation
To properly execute the scripts, please navigate to the root directory and install all necessary dependencies using the commend:
```
pip install -r requirements.txt
```

*To ensure all scripts are executed and all files are stored properly, please always execute any of the following scripts directly under the root directory.*

### Data engineering
The data engineering process goes like this: first, we execute the data engineering scripts to store the feature names selected by the algorithms; second, we run the intersection script to collect the feature interaction set from all data selection algorithms; finally, we output the dataset with only the selected feature columns and some data augmentation.

We have supported two data engineering methods: selection by PCA and SHAP. To execute the entire process, please do the following:
```
python data_engineer/selection_pca.py
python data_engineer/TODO_shap.py
python find_intersection.py
python final_process.py
```
You will then find the final dataset proper for training named `train_final.csv` under `datasets/`.

### Model training
#### Baselines
For each baseline models under `model_trainers/baselines`, simply execute the corresponding scripts. Each script will train and test the dataset. Considering the short training time on GPU, we are not storing the model weights for trained baseline models.

Here is an example to properly execute a baseline model:
```
python model_trainers/baselines/TODO_lstm.py --lr --...
```
It is worth noting that since we experimented exhaustively on various model configurations, we have not set default values for many hyperparameters, so those values should be explicitly stated in the command line.

#### Proposed method
We have proposed an ensemble method that combines two powerful time-series prediction models to predict stock market return. To elaborate, we first separately train two models on the training data: PatchTST and TFT, then we freeze the parameters of those models and fit a bottleneck MLP for the final inference.

To properly run the propsed method, please refer here:
```
python model_trainers/proposed/patchtst.py --TODO...
python model_trainers/proposed/TODO_tft.py --TODO...
python model_trainers/proposed/mlp.py --TODO...
```

Then, to test the ensemble method' performance, execute:
```
python test.py --TODO...
```

#### Ablation and other integration
We also conducted some ablation and alternative intergration of models. These are mostly on testing the performance of PatchTST or TFT let alone plus MLP. To execute such ablation, simply do that:
```
python model_trainers/proposed/patchtst.py --TODO...
python model_trainers/proposed/mlp.py --TODO...
```

Then test it:
```
python test.py --TODO...
```