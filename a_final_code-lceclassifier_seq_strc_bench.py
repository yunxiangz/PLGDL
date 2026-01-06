import os
import io
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, \
    average_precision_score, matthews_corrcoef, accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import argparse
import ast

parser = argparse.ArgumentParser()
parser.add_argument('--seq_model_name', type=str, default='self_seq', help="represent the embeded sequence data")
parser.add_argument('--feature_type', type=str, default='seq_strc',
                    help="seq_str: final features contain both sequence and structure; seq: final features only contain sequence; strc: final features only contain structure")
parser.add_argument('--test_bench', type=str, default='bpags', help="test dataset name: bpags or monkeypos")
parser.add_argument('--is_filtered_data', type=ast.literal_eval, default=True,
                    help="whether the train data need to filter duo to test data")
parser.add_argument('--reduced_dim_strc', type=int, default=28, help="reduced dim for structure data")
parser.add_argument('--reduced_dim_seq', type=int, default=255, help="reduced dim for sequence data")
parser.add_argument('--result_dir_suffix', type=str, default='',
                    help="if change the data or code, please use a new dir")
parser.add_argument('--gpu', type=str, default='0', help="gpu id")
parser.add_argument('--seed', type=int, default=42, help="random seed")

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

MODEL_NAME = args.seq_model_name
RANDOM_STATE = args.seed
reduced_dim_seq = args.reduced_dim_seq
reduced_dim_strc = args.reduced_dim_strc
result_dir_suffix = args.result_dir_suffix
feature_type = args.feature_type
test_bench = args.test_bench
is_filtered_data = args.is_filtered_data

REDUCES = {
    'CHOOSE_FEATURES':
        ['xgboost'],
    "COMPRESS_TEATURES":
        ['PCA'],

}

if 'bpags' == test_bench:
    params = {
        'objective': 'binary:logistic',
        'scale_pos_weight': 16,

        'learning_rate': 0.08,  # 0.05 0.08 0.1
        'n_estimators': 24,

        'max_depth': 5,
        'min_child_weight': 2,
        'gamma': 0.3,

        'subsample': 0.7,
        'colsample_bytree': 0.7,

        'random_state': RANDOM_STATE
    }

else:
    params = {
        'objective': 'binary:logistic',
        'scale_pos_weight': 12,
        'use_label_encoder': False,
        'learning_rate': 0.08,
        'max_depth': 4,
        'random_state': 42,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'min_child_weight': 1.5,
        'gamma': 0.05,
        'n_estimators': 139
    }

MODEL_DIM = {
    'self_seq': 1280,
    'self_strc': 960
}

origin_dim = MODEL_DIM[MODEL_NAME]
print('origin_dim ', origin_dim)

reduces_clf = 'CHOOSE_FEATURES'

assert reduces_clf in REDUCES.keys()

method = REDUCES[reduces_clf][-1]

print(f'reduce dim {reduced_dim_seq}')

### training data#####
seq_ft_dir = f'data/{MODEL_NAME}/{test_bench}/reduce_dim_{reduced_dim_seq}'
str_ft_dir = f'data/self_strc/{test_bench}/reduce_dim_{reduced_dim_strc}/'
if is_filtered_data:
    seq_ft_dir = f'data/{MODEL_NAME}/filter_for_{test_bench}/reduce_dim_{reduced_dim_seq}'
    str_ft_dir = f'data/self_strc/filter_for_{test_bench}/reduce_dim_{reduced_dim_strc}/'

seq_ft_path = os.path.join(seq_ft_dir, reduces_clf, method + '.xlsx')
strc_ft_path = os.path.join(str_ft_dir, 'COMPRESS_TEATURES', 'PCA.xlsx')

print(f'seq_ft_path {seq_ft_path}')
print(f'strc_ft_path {strc_ft_path}')

RESULTS_DIR = os.path.join(seq_ft_dir, reduces_clf, method, f'test{result_dir_suffix}')

print('result dir ', RESULTS_DIR)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

## load data
seq_data = pd.read_excel(seq_ft_path)
strc_data = pd.read_excel(strc_ft_path)
print('seq data.shape ', seq_data.shape)
print('strc_data.shape ', strc_data.shape)

name_labels = seq_data.iloc[:, [0, -1]]
assert len(name_labels.columns) == 2

print('drop the last column of strc ', set(strc_data.iloc[:, -1]))
strc_data = strc_data.drop(strc_data.columns[-1], axis=1)

## update the right label
strc_data = pd.merge(strc_data, name_labels, left_on=0, right_on=0, suffixes=('_seq', '_str'))
print('name label, ', name_labels)
print('after merge ', strc_data.iloc[:, -1])

strc_data.columns = [i for i in range(len(strc_data.columns))]
assert len(set(strc_data.iloc[:,
               -1])) == 2, f'strc_data label should be 0 or 1, but now labels are {set(strc_data.iloc[:, -1])}'

###############strc+seq######################
if 'seq_strc' == feature_type:
    merged_data = pd.merge(seq_data, strc_data, left_on=0, right_on=0, suffixes=('_seq', '_str'))
    assert merged_data.iloc[:, 1 + reduced_dim_seq].equals(merged_data.iloc[:,
                                                           -1]), f'seq_strc feature should be the same, but now they are {merged_data.iloc[:, 1 + reduced_dim_seq]} and {merged_data.iloc[:, -1]}'
    # print(f'merged_data[1 + reduced_dim_seq] {merged_data[1 + reduced_dim_seq]}')
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

X_train, Y_train = features, labels

xgb_clf = xgb.XGBClassifier(
    **params,
)

xgb_clf.fit(X_train, Y_train, verbose=False)

xgb_clf.save_model(os.path.join(RESULTS_DIR, 'xgb_model.json'))
print(f"save xbg to {os.path.join(RESULTS_DIR, 'xgb_model.json')}")

print('f===========================test=====================================')
### test
seq_ft_dir = f'data_test/{MODEL_NAME}/{test_bench}/reduce_dim_{reduced_dim_seq}/'
str_ft_dir = f'data_test/self_strc/{test_bench}/reduce_dim_{reduced_dim_strc}'
if is_filtered_data:
    seq_ft_dir = f'data_test/{MODEL_NAME}/{test_bench}/filter_for_{test_bench}/reduce_dim_{reduced_dim_seq}/'
    str_ft_dir = f'data_test/self_strc/{test_bench}/filter_for_{test_bench}/reduce_dim_{reduced_dim_strc}'

seq_ft_path = os.path.join(seq_ft_dir, reduces_clf, method + '.xlsx')
strc_ft_path = os.path.join(str_ft_dir, 'COMPRESS_TEATURES', 'PCA.xlsx')

print(f'seq_ft_path {seq_ft_path}')
print(f'strc_ft_path {strc_ft_path}')

RESULTS_DIR = os.path.join(seq_ft_dir, reduces_clf, method, f'test{result_dir_suffix}')

print('result dir ', RESULTS_DIR)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

## load data
seq_data = pd.read_excel(seq_ft_path)
strc_data = pd.read_excel(strc_ft_path)
print('seq data.shape ', seq_data.shape)
print('strc_data.shape ', strc_data.shape)

if 'monkeypos' == test_bench:
    strc_data.iloc[:, 0] = strc_data.iloc[:, 0].astype(str) + ".1"
    seq_data.iloc[:, 0] = seq_data.iloc[:, 0].astype(str)

name_labels = seq_data.iloc[:, [0, -1]]
assert len(name_labels.columns) == 2
print('drop the last column of strc ', set(strc_data.iloc[:, -1]))
strc_data = strc_data.drop(strc_data.columns[-1], axis=1)
print(f'name_labels {name_labels.shape}, strc_data {strc_data.shape}')
## update the right label
strc_data = pd.merge(strc_data, name_labels, left_on=0, right_on=0, suffixes=('_seq', '_str'))
print('name label, ', name_labels)
print('after merge ', strc_data.iloc[:, -1])

strc_data.columns = [i for i in range(len(strc_data.columns))]
# assert len(set(strc_data.iloc[:, -1])) == 2, f'strc_data label should be 0 or 1, but now labels are {set(strc_data.iloc[:, -1])}'

###############strc+seq######################
if 'seq_strc' == feature_type:
    merged_data = pd.merge(seq_data, strc_data, left_on=0, right_on=0, suffixes=('_seq', '_str'))
    assert merged_data.iloc[:, 1 + reduced_dim_seq].equals(merged_data.iloc[:,
                                                           -1]), f'seq_strc feature should be the same, but now they are {merged_data.iloc[:, 1 + reduced_dim_seq]} and {merged_data.iloc[:, -1]}'
    print(f'merged_data[1 + reduced_dim_seq] {merged_data.iloc[:, 1 + reduced_dim_seq]}')
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

print('plan to save to ', RESULTS_DIR + "/evaluation_results.txt")

X_test = features
Y_test = labels
print(f'X_test {X_test.shape};; Y_test {Y_test.shape}')

model_dir = f'data/{MODEL_NAME}/{test_bench}/reduce_dim_{reduced_dim_seq}/'
if is_filtered_data:
    model_dir = f'data/{MODEL_NAME}/filter_for_{test_bench}/reduce_dim_{reduced_dim_seq}/'

model_path = os.path.join(model_dir, reduces_clf, method, f'test{result_dir_suffix}', 'xgb_model.json')

print(f'xgb model path {model_path}')

xgb_clf = xgb.XGBClassifier()
xgb_clf.load_model(model_path)
if 'bpags' == test_bench:
    y_proba = xgb_clf.predict_proba(X_test)[:, 1]
    y_pred = xgb_clf.predict(X_test)

    print('plan to save to ', RESULTS_DIR + "/evaluation_results.txt")

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

    print('finish save to ', RESULTS_DIR + "/evaluation_results.txt")

elif 'monkeypos' == test_bench:

    y_proba = xgb_clf.predict_proba(X_test)[:, 1]

    sorted_indices = np.argsort(y_proba)[::-1]  # 按照概率降序排列

    sorted_names = [NAMES[i] for i in sorted_indices]

    sorted_probs = y_proba[sorted_indices]

    with open(os.path.join(RESULTS_DIR, "evaluation_results.txt"), "w") as f:
        f.write("Index\tName\tProbability\n")  # 写入表头
        for i in range(len(sorted_indices)):
            f.write(f"{sorted_indices[i]}\t{sorted_names[i]}\t{sorted_probs[i]:.4f}\n")
    print(f"Saved to {os.path.join(RESULTS_DIR, 'evaluation_results.txt')}")

    top_num = 10
    top_indices = np.argsort(y_proba)[-top_num:][::-1].tolist()

    print(f"{top_num}:  {top_indices}")
    print("probs：", y_proba[top_indices])
    name_ids = [NAMES[i] for i in top_indices]
    print("ids：", name_ids)
