import os
import io
import sys
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from xgboost import Booster

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, matthews_corrcoef, accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from scipy.special import expit  # sigmoid 函数
from sklearn.ensemble import RandomForestClassifier
import argparse

REDUCES = {
    'CHOOSE_FEATURES':
        ['xgboost'],
    "COMPRESS_TEATURES":
        ['PCA'],
}

RANDOM_STATE = 42
TEST_RADIO = 0.2
params = {
    'objective': 'binary:logistic',
    'scale_pos_weight': 12,
    'use_label_encoder': False,
    'eval_metric': 'logloss',

    'n_estimators': 143,
    'learning_rate': 0.1,
    'random_state': RANDOM_STATE
}

MODEL_NAME = 'self_seq'
MODEL_DIM = {
    'self_seq': [1280, 255],
    'self_strc': [960, 27]
}

origin_dim = MODEL_DIM[MODEL_NAME][0]
print('origin_dim ', origin_dim)

reduces_clf = 'CHOOSE_FEATURES'
assert reduces_clf in REDUCES.keys()

method = REDUCES[reduces_clf][-1]

reduced_dim = MODEL_DIM[MODEL_NAME][1]

print(f'reduce dim {reduced_dim}')

seq_ft_dir = f'data/{MODEL_NAME}/reduce_dim_{reduced_dim}/'
str_ft_dir = f'data/self_strc/reduce_dim_27/'

seq_ft_path = os.path.join(seq_ft_dir, reduces_clf, method + '.xlsx')
str_ft_path = os.path.join(str_ft_dir, 'COMPRESS_TEATURES', 'PCA.xlsx')

print(f'seq_ft_path {seq_ft_path}')
print(f'str_ft_path {str_ft_path}')

RESULTS_DIR = os.path.join(seq_ft_dir, reduces_clf, method, f'test_{TEST_RADIO}_okk')

print('result dir ', RESULTS_DIR)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

seq_data = pd.read_excel(seq_ft_path)
str_data = pd.read_excel(str_ft_path)
print('seq data.shape ', seq_data.shape)
print('str_data.shape ', str_data.shape)


merged_data = pd.merge(seq_data, str_data, left_on=0, right_on=0, suffixes=('_seq', '_str'))
assert merged_data.iloc[:,1+reduced_dim].equals(merged_data.iloc[:,-1])
merged_data = merged_data.drop(merged_data.columns[1 + reduced_dim], axis=1)

merged_data.to_excel(os.path.join(RESULTS_DIR, 'merged_data.xlsx'), index=False, header=False)

NAMES = merged_data.iloc[:, 0]
LABELS = merged_data.iloc[:, -1]
labels = LABELS.to_numpy()
features = merged_data.iloc[:, 1:-1].to_numpy()

print('save to ', RESULTS_DIR + "/evaluation_results.txt")


total_sample_nums = features.shape[0]
indices = np.arange(total_sample_nums)
print('total samples : ', len(indices))
X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(
    features, labels, indices,
    test_size=TEST_RADIO,
    random_state=RANDOM_STATE,
    stratify=labels
)
print(f'train dataset {1-TEST_RADIO} {X_train.shape}; test dataset{TEST_RADIO} {X_test.shape}')
test_features = X_test
test_labels = Y_test

xgb_clf = xgb.XGBClassifier(
    **params,
)

print('test_feature data.shape ', test_features.shape)
print('test_label .shape ', test_labels.shape, test_labels)

xgb_clf.fit(X_train, Y_train)

# ==============test=========================================================
print('=======================================test===================================')
y_proba = xgb_clf.predict_proba(X_test)[:, 1]
y_pred = xgb_clf.predict(X_test)

test_ids = [NAMES[i] for i in idx_test]
true_labels = Y_test

df_result = pd.DataFrame({
    'ID': test_ids,
    'TrueLabel': true_labels,
    'PredictedScore': y_proba
})

# save
df_result.to_csv(os.path.join(RESULTS_DIR, "test_predictions.csv"), index=False)
print('save to ', RESULTS_DIR + "/test_predictions.csv")

output = io.StringIO()
sys.stdout = output

print('Accuracy: ', accuracy_score(Y_test, y_pred))
print("F1-score:",
      f1_score(Y_test, y_pred))
print("Recall:", recall_score(Y_test, y_pred))
print("Precision:", precision_score(Y_test, y_pred))
print("AUC-ROC:", roc_auc_score(Y_test, y_proba))
print("AUC-PR:", average_precision_score(Y_test, y_proba))
print("MCC:", matthews_corrcoef(Y_test, y_pred))

sys.stdout = sys.__stdout__

print(output.getvalue())


with open(os.path.join(RESULTS_DIR, "evaluation_results.txt"), "w") as file:
    file.write(output.getvalue())

print('save to ', RESULTS_DIR + "/evaluation_results.txt")

xgb_clf.save_model(os.path.join(RESULTS_DIR, 'xgb_model.json'))
print(f"save xbg to {os.path.join(RESULTS_DIR, 'xgb_model.json')}")
