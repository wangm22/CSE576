# Feature Engineering and Predictive Modeling on Stock Market Return
This repo is dedicated to showcase the data engineering and predictive modeling process of stock market return.

## Folder structure
```
576_PROJECT/
├── data_engineer/              # This folder that executes data engineering
│   │                           # and store feature names selected by the algorithms
│   ├── final_process.py
│   ├── find_intersection.py
│   ├── selection_pca.py
│   ├── shap_process.py
│   └── ...
│
├── datasets/                   # This folder stores raw, intermediate and final datasets
│   └── ...
│
├── model_trainers/             # This folder contains all model trainers
│   ├── baselines/
│   │   ├── baseline_xgb.py
│   │   ├── run_lstm.py
│   │   ├── run_transformer.py
│   │   └── ...
│   │
│   └── proposed/
│       ├── mlp.py
│       ├── patchtst.py
│       └── tft.py
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
To properly execute the scripts, please ensure that you are on `python` of version `3.11`. Then, please navigate to the root directory and install all necessary dependencies using the commend:
```
pip install -r requirements.txt
```

### Data engineering
The data engineering process goes like this: first, we execute the data engineering scripts to store the feature names selected by the algorithms; second, we run the intersection script to collect the feature interaction set from all data selection algorithms; finally, we output the dataset with only the selected feature columns and some data augmentation.

We have supported two data engineering methods: selection by PCA and SHAP. To execute the entire process, please do the following:
```
python data_engineer/selection_pca.py
cd data_engineer
python shap_process.py
```

Then:
```
python data_engineer/find_intersection.py
python data_engineer/final_process.py
```
You will then find the final dataset proper for training named `train_final.csv` under `datasets/`.

### Model training
#### Baselines
For each baseline models under `model_trainers/baselines`, simply execute the corresponding scripts. Each script will train and test the dataset. Considering the short training time on GPU, we are not storing the model weights for trained baseline models.

To run LSTM:
```
cd model_trainers/baselines
python run_lstm.py
```

To run Transformer:
```

```

To run XGBoost:
```
python model_trainers/baselines/baseline_xgb.py
```

#### Proposed method
We have proposed an ensemble method that combines two powerful time-series prediction models to predict stock market return. To elaborate, we first separately train two models on the training data: PatchTST and TFT, then we freeze the parameters of those models and fit a bottleneck MLP for the final inference.

To properly run the propsed method, please refer here:
```
python model_trainers/proposed/patchtst.py
python model_trainers/proposed/tft.py
python model_trainers/proposed/mlp.py
```

Then, to test the ensemble method's performance, execute:
```
python test.py
```

#### Ablation and other integration
We also conducted some ablation and alternative intergration of models. These are mostly on testing the performance of PatchTST or TFT let alone plus MLP. To execute such ablation, simply do that:
```
python model_trainers/proposed/patchtst.py --tft False
python model_trainers/proposed/mlp.py --tft False
```

Then test it:
```
python test.py --tft False
```

Similarly, setting `--patchtst False` will run the experiment on TFT + MLP.