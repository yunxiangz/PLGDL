import os
import io
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score, \
    average_precision_score, matthews_corrcoef, accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier




## Total kinds of methods
REDUCES = {
    'CHOOSE_FEATURES':
        ['xgboost'],
    "COMPRESS_TEATURES":
        ['PCA'],

}

RANDOM_STATE = 42

TEST_RADIO = 0

filter_for_bpags = True

## IF train, choose tesbench='
# test_bench = ''
# If test, choose testbench='bpags'
test_bench = 'bpags'

## best
params = {
    'objective': 'binary:logistic',
    'scale_pos_weight': 16,
    'eval_metric': 'logloss',
    'use_label_encoder': False,

    'learning_rate': 0.08,
    'n_estimators': 60,

    'max_depth': 4,
    'min_child_weight': 3,
    'gamma': 0.3,

    'subsample': 0.8,
    'colsample_bytree': 0.8,

    'reg_alpha': 0.7,
    'reg_lambda': 1.0,

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

if test_bench == '':
    seq_ft_dir = f'data/{MODEL_NAME}/reduce_dim_{reduced_dim}/'
    str_ft_dir = f'data/self_strc/reduce_dim_27/'

    if filter_for_bpags:
        seq_ft_dir = f'data/{MODEL_NAME}/filter_for_bpags/reduce_dim_{reduced_dim}'
        str_ft_dir = f'data/self_strc/filter_for_bpags/reduce_dim_27/'


else:
    seq_ft_dir = f'data_test/{MODEL_NAME}/{test_bench}/reduce_dim_{reduced_dim}/'
    str_ft_dir = f'data_test/self_strc/{test_bench}/reduce_dim_27'
    if filter_for_bpags:
        seq_ft_dir = f'data_test/{MODEL_NAME}/{test_bench}/filter_for_bpags/reduce_dim_{reduced_dim}/'
        str_ft_dir = f'data_test/self_strc/{test_bench}/filter_for_bpags/reduce_dim_27'

seq_ft_path = os.path.join(seq_ft_dir, reduces_clf, method + '.xlsx')
str_ft_path = os.path.join(str_ft_dir, 'COMPRESS_TEATURES', 'PCA.xlsx')

print(f'seq_ft_path {seq_ft_path}')
print(f'str_ft_path {str_ft_path}')

RESULTS_DIR = os.path.join(seq_ft_dir, reduces_clf, method, f'test_{TEST_RADIO}')

print('result dir ', RESULTS_DIR)
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

## load data
seq_data = pd.read_excel(seq_ft_path)
str_data = pd.read_excel(str_ft_path)
print('seq data.shape ', seq_data.shape)
print('str_data.shape ', str_data.shape)

## merge data
merged_data = pd.merge(seq_data, str_data, left_on=0, right_on=0, suffixes=('_seq', '_str'))
print(f'merge shape {merged_data.shape}')
if test_bench == '':
    assert merged_data.iloc[:, 1 + reduced_dim].equals(merged_data.iloc[:, -1])
print(f'merged_data.columns {merged_data.columns}')

if test_bench == '':
    print(f' merged_data.columns[1 + reduced_dim] {merged_data.columns[1 + reduced_dim]}')
    print(
        f' merged_data.columns[{merged_data.columns[1 + reduced_dim]}] {merged_data.iloc[:, 1 + reduced_dim]}')
    merged_data = merged_data.drop(merged_data.columns[1 + reduced_dim], axis=1)
elif test_bench == 'bpags':
    print(f' merged_data.columns[-1] {merged_data.columns[-1]}')
    print(
        f' merged_data.columns[{merged_data.columns[-1]}]  {merged_data.iloc[:, -1]}')
    merged_data = merged_data.drop(merged_data.columns[-1], axis=1)
    cols = list(merged_data.columns)
    col_to_move = cols[1 + reduced_dim]
    new_cols = cols[:1 + reduced_dim] + cols[2 + reduced_dim:] + [col_to_move]
    merged_data = merged_data[new_cols]

else:
    raise Exception
merged_data.to_excel(os.path.join(RESULTS_DIR, 'merged_data.xlsx'), index=False, header=False)
print(f"merge data saves to {os.path.join(RESULTS_DIR, 'merged_data.xlsx')}")

NAMES = merged_data.iloc[:, 0]
LABELS = merged_data.iloc[:, -1]
labels = LABELS.to_numpy()
features = merged_data.iloc[:, 1:-1].to_numpy()

print('plan to save to ', RESULTS_DIR + "/evaluation_results.txt")

if test_bench == '':
    test_seq_ft_dir = f'data_test/{MODEL_NAME}/bpags/filter_for_bpags/reduce_dim_{reduced_dim}/'
    test_str_ft_dir = f'data_test/self_strc/bpags/filter_for_bpags/reduce_dim_27'
    test_seq_ft_path = os.path.join(test_seq_ft_dir, reduces_clf, method + '.xlsx')
    test_str_ft_path = os.path.join(test_str_ft_dir, 'COMPRESS_TEATURES', 'PCA.xlsx')

    test_seq_data = pd.read_excel(test_seq_ft_path)
    test_str_data = pd.read_excel(test_str_ft_path)

    test_merged_data = pd.merge(test_seq_data, test_str_data, left_on=0, right_on=0, suffixes=('_seq', '_str'))

    test_merged_data = test_merged_data.drop(test_merged_data.columns[-1], axis=1)
    cols = list(test_merged_data.columns)
    col_to_move = cols[1 + reduced_dim]
    print('move col ', col_to_move)
    new_cols = cols[:1 + reduced_dim] + cols[2 + reduced_dim:] + [col_to_move]
    test_merged_data = test_merged_data[new_cols]

    test_labels = test_merged_data.iloc[:, -1].to_numpy()  # 所有行，最后一列
    test_features = test_merged_data.iloc[:, 1:-1].to_numpy()  # 所有行，除了第一列和最后一列

    X_train, Y_train = features, labels

    xgb_clf = xgb.XGBClassifier(
        **params
    )

    print('test_feature data.shape ', test_features.shape)
    print('test_label .shape ', test_labels.shape, test_labels)

    xgb_clf.fit(X_train, Y_train, verbose=False)

else:
    X_test = features
    Y_test = labels
    print(f'X_test {X_test.shape};; Y_test {Y_test.shape}')

    model_dir = f'data/{MODEL_NAME}/reduce_dim_{reduced_dim}/'
    if filter_for_bpags:
        model_dir = f'data/{MODEL_NAME}/filter_for_bpags/reduce_dim_{reduced_dim}/'

    model_path = os.path.join(model_dir, reduces_clf, method, f'test_{TEST_RADIO}', 'xgb_model.json')

    print(f'xgb model path {model_path}')

    xgb_clf = xgb.XGBClassifier()
    xgb_clf.load_model(model_path)

    if test_bench == 'bpags':
        y_proba = xgb_clf.predict_proba(X_test)[:, 1]
        y_pred = xgb_clf.predict(X_test)

        print('plan to save to ', RESULTS_DIR + "/evaluation_results.txt")

        output = io.StringIO()
        sys.stdout = output

        print('Accuracy: ', accuracy_score(Y_test, y_pred))
        print("F1-score:",
              f1_score(Y_test, y_pred))  # Precision 和 Recall 的平衡，适合不均衡分类任务 2⋅Precision⋅Recall/(Precision+Recall)
        print("Recall:", recall_score(Y_test, y_pred))  # 预测正确的数据中，正样本预测正确的占比 TP/(TP+FN)
        print("Precision:", precision_score(Y_test, y_pred))  # 对所有正样本的预测中，预测正确的比例 TP/(TP+FP)
        print("AUC-ROC:", roc_auc_score(Y_test, y_proba))
        print("AUC-PR:", average_precision_score(Y_test, y_proba))
        print("MCC:", matthews_corrcoef(Y_test, y_pred))


        sys.stdout = sys.__stdout__

        print(output.getvalue())

        with open(os.path.join(RESULTS_DIR, "evaluation_results.txt"), "w") as file:
            file.write(output.getvalue())
        print('finish save to ', RESULTS_DIR + "/evaluation_results.txt")


if test_bench == '':
    print(f'best iteration ', xgb_clf.best_iteration)
    xgb_clf.save_model(os.path.join(RESULTS_DIR, 'xgb_model.json'))
    print(f"save xbg to {os.path.join(RESULTS_DIR, 'xgb_model.json')}")
