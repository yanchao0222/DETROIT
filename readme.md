# Deep Imputor of Missing Values in Temporal Lab Tests

*******************************************************
**IEEE ICHI Data Analytics Challenge on Missing data Imputation (DACMI)**
*******************************************************
>***Team members***:

>**Chao Yan**,  EECS, Vanderbilt University

>**Cheng Gao**, DBMI, Vanderbilt University Medical Center

>**Xinmeng Zhang**, EECS, Vanderbilt University

>***Advised by***: 

>**You Chen**, DBMI,  Vanderbilt University Medical Center

>**Bradley Malin**,  DBMI,  Vanderbilt University Medical Center

<br />
<br />

This repository contains the code in **PyTorch** (0.4.1) for our method of imputing missing values in temporal lab tests of patients.

## Framework
Achieving the missing value imputation by our imputor consists of three main parts: **1) prefill**, **2) train a deep regressor**, and **3) apply**.


### Prefill
A wide variaty of techniques have been developed to address the imputation problem of missing values. We tested multiple existing methods to prefill the missing data, which will be leveraged for model training. The tentative methods included filling with mean, k-nearest neighbors (KNN), multiple imputations with chained equations (MICE), and matrix completion. By comparing the imputed values with ground truth values, we observed that filling with the local mean had better imputing performance on labs PCL, PLCO2, MCV, PLT, WBC, RDW, PBUN and PCRE. In contrast, the labs PK, PNA, HCT, HGB, and PGLU can be better imputed using matrix completion (softImpute), which is based on 

>***Mazumder R, Hastie T, Tibshirani R. Spectral regularization algorithms for learning large incomplete matrices. Journal of machine learning research. 2010;11(Aug):2287-322.*** 

The comparison results can be found in our short paper. 

By running `./train_and_evaluate_with_training_data/prefill.py`, all the missings in folder `./train_with_missing/`  can be prefilled and then saved into folder `./train_and_evaluate_with_training_data/filled_groundtruth/mix_impute`. This is done and unnecessary to rerun. If needed, please make sure that the package 'fancyimpute' is successfully installed.


### Train a deep nerual regressor
We engineered a fully connected neural netowrk architecture to predict lab test values. For each lab, we trained a distinct model to predict values only for this lab. Thus, there are 13 models being trained. We divide the training data into two parts: (90% for training vs 10% for validation). The file `./train_and_evaluate_with_training_data/regressor_train.py` takes both `./train_and_evaluate_with_training_data/train_groundtruth/` and the prefilled results as input (using 90% data), and outputs the trained models, as well as the imputation performance in the validation set.

Please refer to our paper for the details of the model training and the performance results.

For each lab, we run model learning for 20 times, and then selected the one showing best imputation performance in the validation set. The finally learned 13 models are saved in `./train_and_evaluate_with_training_data/selected_model/`.


### Apply
We first applied the prefill protocal in the test dataset, which can be achieved by running `./fill_test_data/prefill.py` (with input in folder `./fill_test_data/challenge_test_data/`). The prefilled results are saved in folder `./fill_test_data/test_mix_impute`.

Then we applied the selected models (stored in folder `./fill_test_data/selected_model/`) out of the training phase to all the missing postions of new patients.
