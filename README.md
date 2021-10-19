# Deep Neural Network for linear regression problems

https://www.kaggle.com/shivachandel/kc-house-data
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data

Note: the second dataset is really small and ML ensemble method should give better results than DL

- collect data into pandas
- split data into categorical/numerical sets
- eliminate low features/columns and deal with missing values
- remove outliers and normalize (numerical) data
- run a random search model using keras tuners
- run a mixed inputs model
- save and load models

mode 0 : run a simple model
mode 1 : run a random search
mode 2 : run a mixed-input model

output figures:
"_features_before_cleaning.png"
"_features_after_cleaning.png"
"_correlation_before_cleaning.png"
"_correlation_after_cleaning.png"
"_optimization.png"
"_pred_vs_true.png"

## Install and configure project

### Install dependencies

```bash
pip install -r requirements.txt
```

### Usage

main file: dnn_linear_regression_problem.py 

```bash
./dnn_linear_regression_problem.py 
```


