
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
import ast
parser = argparse.ArgumentParser()
parser.add_argument('--seq_model_name', type=str, default='self_seq', help="represent the embeded sequence data")
parser.add_argument('--feature_type', type=str, default='seq_strc', help="seq_str: final features contain both sequence and structure; seq: final features only contain sequence; strc: final features only contain structure")
parser.add_argument('--reduced_dim_strc', type=int, default=28, help="reduced dim for structure data")
parser.add_argument('--reduced_dim_seq', type=ast.literal_eval, default=255, help="reduced dim for sequence data")
parser.add_argument('--result_dir_suffix', type=str, default='', help="if change the data or code, please use a new dir")
parser.add_argument('--gpu', type=str, default='0', help="gpu id")
parser.add_argument('--seed', type=int, default=42, help="random seed")
parser.add_argument('--test_radio', type=float, default=0.2, help="test radio in the total dataset")
parser.add_argument('--info_radio', type=float, default=-1, help="test radio in the total dataset")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

MODEL_NAME = args.seq_model_name
RANDOM_STATE = args.seed
TEST_RADIO = args.test_radio
reduced_dim_seq = args.reduced_dim_seq
reduced_dim_strc = args.reduced_dim_strc
result_dir_suffix = args.result_dir_suffix
feature_type = args.feature_type
info_radio = args.info_radio
## train and test from same dataset
REDUCES = {
    'CHOOSE_FEATURES':
        ['xgboost'],
    "COMPRESS_TEATURES":
        ['PCA'],
}

# RANDOM_STATE = 42

# params = {
#     'objective': 'binary:logistic',
#     'scale_pos_weight': 12,
#     'use_label_encoder': False,
#     'eval_metric': 'logloss',
#
#     'n_estimators': 143,
#     'learning_rate': 0.1,
#     'random_state': RANDOM_STATE
# }

params = {
    'objective': 'binary:logistic',
    'scale_pos_weight': 12,

    'n_estimators': 127,
    'learning_rate': 0.2,

    'subsample': 0.8,
    'colsample_bytree': 0.8,

    # 'reg_alpha': 0.4,

    'random_state': RANDOM_STATE
}

MODEL_DIM = {
    'self_seq': 1280,
    'AMPLIFY_350M': 960,
    'prot_t5_xl_uniref50': 1024,
}

origin_dim = MODEL_DIM[MODEL_NAME]
print('origin_dim ', origin_dim)

reduces_clf = 'CHOOSE_FEATURES'
assert reduces_clf in REDUCES.keys()

method = REDUCES[reduces_clf][-1]


print(f'seq reduce dim {reduced_dim_seq}')

seq_ft_dir = f'data/{MODEL_NAME}/reduce_dim_{reduced_dim_seq}/'  # 255
str_ft_dir = f'data/self_strc/reduce_dim_{reduced_dim_strc}/'
if info_radio >= 0:
    seq_ft_dir = f'data/{MODEL_NAME}/reduce_dim_{info_radio:.2f}_{reduced_dim_seq}/'
seq_ft_path = os.path.join(seq_ft_dir, reduces_clf, method + '.xlsx')  # xgboost
strc_ft_path = os.path.join(str_ft_dir, 'COMPRESS_TEATURES', 'PCA.xlsx')  # pca

print(f'seq_ft_path {seq_ft_path}')
print(f'strc_ft_path {strc_ft_path}')

RESULTS_DIR = os.path.join(seq_ft_dir, reduces_clf, method, f'test_{TEST_RADIO}{result_dir_suffix}')

print('result dir ', RESULTS_DIR)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

## load data
seq_data = pd.read_excel(seq_ft_path)
strc_data = pd.read_excel(strc_ft_path)
print('seq data.shape ', seq_data.shape)
print('strc_data.shape ', strc_data.shape)


# strc_data最后一列一定有，但是可能是-1
name_labels = seq_data.iloc[:, [0, -1]]
assert len(name_labels.columns) == 2
print('drop the last column of strc ', set(strc_data.iloc[:, -1]))
strc_data = strc_data.drop(strc_data.columns[-1], axis=1)
## update the right label
strc_data = pd.merge(strc_data, name_labels, left_on=0, right_on=0, suffixes=('_seq', '_str'))
print('name label, ', name_labels)
print('after merge ', strc_data.iloc[:, -1])
## 重命名所有列
strc_data.columns = [i for i in range(len(strc_data.columns))]
assert len(set(strc_data.iloc[:, -1])) == 2, f'strc_data label should be 0 or 1, but now labels are {set(strc_data.iloc[:, -1])}'

###############strc+seq######################
if 'seq_strc' == feature_type:
    merged_data = pd.merge(seq_data, strc_data, left_on=0, right_on=0, suffixes=('_seq', '_str'))
    assert merged_data.iloc[:,1+reduced_dim_seq].equals(merged_data.iloc[:,-1]), f'seq_strc feature should be the same, but now they are {merged_data.iloc[:,1+reduced_dim_seq]} and {merged_data.iloc[:,-1]}'
    print(f'merged_data[1 + reduced_dim_seq] {set(merged_data.iloc[:, 1 + reduced_dim_seq])}')
    merged_data = merged_data.drop(merged_data.columns[1 + reduced_dim_seq], axis=1)
    # print('merged_data2.shape ', merged_data.shape)
    merged_data.to_excel(os.path.join(RESULTS_DIR, 'merged_data.xlsx'), index=False, header=False)

##############only strc######################
elif 'strc' == feature_type:
    merged_data = strc_data
    merged_data.to_excel(os.path.join(RESULTS_DIR, 'merged_data_only_strc.xlsx'), index=False, header=False)
#############only seq#######################
elif 'seq' == feature_type:
    merged_data = seq_data
    merged_data.to_excel(os.path.join(RESULTS_DIR, 'merged_data_only_seq.xlsx'), index=False, header=False)
#####################
print('merged_data.shape ', merged_data.shape)

NAMES = merged_data.iloc[:, 0]
LABELS = merged_data.iloc[:, -1]
labels = LABELS.to_numpy()
features = merged_data.iloc[:, 1:-1].to_numpy()

assert len(set(labels)) == 2, f'label should be 0 or 1, but now labels are {set(labels)}'

total_sample_nums = features.shape[0]
indices = np.arange(total_sample_nums)
print('total samples : ', len(indices))

#### split 0.8 training set; 0.2 test set###################
X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(
    features, labels, indices,
    test_size=TEST_RADIO,
    random_state=RANDOM_STATE,
    stratify=labels
)

test_features = X_test
test_labels = Y_test

print(f'train dataset {1-TEST_RADIO} {X_train.shape}; test dataset{TEST_RADIO} {X_test.shape}')
print('test_feature data.shape ', test_features.shape)
print('test_label .shape ', test_labels.shape, test_labels)

# train model
xgb_clf = xgb.XGBClassifier(
    **params,
)

xgb_clf.fit(X_train, Y_train)

xgb_clf.save_model(os.path.join(RESULTS_DIR, 'xgb_model.json'))
print(f"save xbg to {os.path.join(RESULTS_DIR, 'xgb_model.json')}")

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

split_type = ['Bacteria', 'Viruses', 'Eukaryota']

# save
# df_result.to_csv(os.path.join(RESULTS_DIR, "test_predictions.csv"), index=False)
# print('save to ', RESULTS_DIR + "/test_predictions.csv")

acc = accuracy_score(Y_test, y_pred)
precision = precision_score(Y_test, y_pred)
recall = recall_score(Y_test, y_pred)
f1_score = f1_score(Y_test, y_pred)
roc_auc = roc_auc_score(Y_test, y_proba)
pr_auc = average_precision_score(Y_test, y_proba)
mcc = matthews_corrcoef(Y_test, y_pred)

eval_scores = {
    "Accuracy": acc,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1_score,
    "ROC AUC": roc_auc,
    "PR AUC": pr_auc,
    "MCC": mcc
}

print(eval_scores)


with open(os.path.join(RESULTS_DIR, "evaluation_results.txt"), "w") as file:
    for key, value in eval_scores.items():
        file.write(f"{key}: {value}\n")

print('save to ', RESULTS_DIR + "/evaluation_results.txt")

