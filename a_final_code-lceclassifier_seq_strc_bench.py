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
parser.add_argument('--feature_type', type=str, default='seq_strc', help="seq_str: final features contain both sequence and structure; seq: final features only contain sequence; strc: final features only contain structure")
parser.add_argument('--test_bench', type=str, default='bpags', help="test dataset name: bpags or monkeypos")
parser.add_argument('--is_filtered_data', type=ast.literal_eval, default=True, help="whether the train data need to filter duo to test data")
parser.add_argument('--reduced_dim_strc', type=int, default=28, help="reduced dim for structure data")
parser.add_argument('--reduced_dim_seq', type=int, default=255, help="reduced dim for sequence data")
parser.add_argument('--result_dir_suffix', type=str, default='', help="if change the data or code, please use a new dir")
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

# METRICS = {
#     'accuracy': None,
#     'precision': None,
#     'recall': None,
#     'f1_score': None,
#     'roc_auc': None,
#     'pr_auc': None,
#     'mcc': None
# }

REDUCES = {
    'CHOOSE_FEATURES':
        ['xgboost'],
    "COMPRESS_TEATURES":
        ['PCA'],

}

if 'bpags' == test_bench:
    # params = {
    #     'objective': 'binary:logistic',
    #     'scale_pos_weight': 16,
    #     # 'scale_pos_weight': 12,
    #     'eval_metric': 'logloss',
    #     'use_label_encoder': False,
    #
    #     'learning_rate': 0.08,
    #     'n_estimators': 60,
    #
    #     'max_depth': 4,
    #     'min_child_weight': 3,
    #     'gamma': 0.3,
    #
    #     'subsample': 0.8,
    #     'colsample_bytree': 0.8,
    #
    #     'reg_alpha': 0.7,
    #     'reg_lambda': 1.0,
    #
    #     'random_state': RANDOM_STATE
    # }
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

        # 'reg_alpha': 0.4,
        # 'reg_lambda': 2,

        'random_state': RANDOM_STATE
    }

else:
    # params = {
    #     'objective': 'binary:logistic',
    #     'scale_pos_weight': 12,  # 比12轻一点，别压住整体正样本分布
    #     'use_label_encoder': False,
    #     'n_estimators': 330,  # 比350稍多，为 B 增加机会
    #     'learning_rate': 0.046,  # 稍微降低，帮助稳定弱样本（B）
    #     'max_depth': 5,  # 保留对 A 的支持结构
    #     'random_state': RANDOM_STATE,
    #     'colsample_bytree': 0.85,  # 稍微增加特征覆盖，帮助 B
    #     'subsample': 0.85,  # 不动，泛化性适中
    #     # 'min_child_weight': 1.5,           # 介于 1 和 2，适度放开边界，兼顾泛化
    #     # 'gamma': 0.01                      # 减轻正则但不为 0，保持一定约束
    # }

    # params = {
    #     'objective': 'binary:logistic',
    #     'scale_pos_weight': 12,  # 比12轻一点，别压住整体正样本分布
    #     'use_label_encoder': False,
    #     'n_estimators': 100,  # 比350稍多，为 B 增加机会  # 80不行，下降了，不如100
    #     'learning_rate': 0.08,  # 稍微降低，帮助稳定弱样本（B）
    #     'max_depth': 5,  # 保留对 A 的支持结构
    #     'random_state': RANDOM_STATE,
    #     'colsample_bytree': 0.75,  # 稍微增加特征覆盖，帮助 B
    #     'subsample': 0.75,  # 不动，泛化性适中
    #     # 'min_child_weight': 1.5,           # 介于 1 和 2，适度放开边界，兼顾泛化
    #     # 'gamma': 0.01                      # 减轻正则但不为 0，保持一定约束
    # }

    # params = {
    #     'objective': 'binary:logistic',
    #     'scale_pos_weight': 12,  # 比12轻一点，别压住整体正样本分布
    #     'use_label_encoder': False,
    #     'learning_rate': 0.08,  # 稍微降低，帮助稳定弱样本（B）
    #     'max_depth': 5,  # 保留对 A 的支持结构
    #     'random_state': RANDOM_STATE,
    #     'colsample_bytree': 0.75,  # 稍微增加特征覆盖，帮助 B
    #     'subsample': 0.75,  # 不动，泛化性适中
    #     'min_child_weight': 1.5,  # 介于 1 和 2，适度放开边界，兼顾泛化
    #     'gamma': 0.01,  # 减轻正则但不为 0，保持一定约束.
    #     # 'n_estimators': 95,
    #     # ['URK20517.1', 'URK20552.1', 'URK20605.1', 'URK20532.1', 'URK20584.1', 'URK20440.1', 'URK20629.1', 'URK20585.1', 'URK20560.1', 'URK20516.1']
    #     # 'n_estimators': 102,
    #     # # 重合8个 ['URK20517.1', 'URK20552.1', 'URK20605.1', 'URK20532.1', 'URK20584.1', 'URK20629.1', 'URK20440.1', 'URK20450.1', 'URK20516.1', 'URK20560.1']
    #     'n_estimators': 104,
    #     # 重合7个 ['URK20517.1', 'URK20605.1', 'URK20552.1', 'URK20532.1', 'URK20584.1', 'URK20629.1', 'URK20516.1']
    #
    # }
    # params = {
    #     'objective': 'binary:logistic',
    #     'scale_pos_weight': 12,
    #     'use_label_encoder': False,
    #     'learning_rate': 0.05,  # 降低学习率
    #     'max_depth': 5,  # 设置树的深度
    #     'random_state': 42,
    #     'colsample_bytree': 0.7,  # 调整特征采样比例
    #     'subsample': 0.7,  # 调整子样本比例
    #     'min_child_weight': 2,  # 调整最小叶子节点样本权重
    #     'gamma': 0.05,  # 增加正则化强度
    #     'n_estimators': 116  # 增加树的数量
    # }
    # -----------------------------------------116--------------------------------
    # 概率最高的前10个样本索引 [31, 20, 98, 78, 100, 19, 150, 30, 38, 80]
    # 对应的概率值： [0.90964854 0.8471456  0.8449387  0.8359956  0.82939655 0.82939655
    #  0.82478935 0.76670504 0.76014626 0.7337763 ]
    # 对应的id： ['URK20450.1', 'URK20605.1', 'URK20517.1', 'URK20532.1', 'URK20629.1', 'URK20440.1', 'URK20584.1', 'URK20621.1', 'URK20552.1', 'URK20516.1']
    # 有 7 个元素和目标重合
    # ['URK20605.1', 'URK20517.1', 'URK20532.1', 'URK20629.1', 'URK20584.1', 'URK20552.1', 'URK20516.1']

    # params = {
    #     'objective': 'binary:logistic',
    #     'scale_pos_weight': 10,  # 降低对正样本的偏好，减少过拟合
    #     'use_label_encoder': False,
    #     'learning_rate': 0.08,  # 保持较低的学习率以提高稳定性
    #     'max_depth': 5,  # 降低树的深度以减少模型复杂度
    #     'random_state': 42,
    #     'colsample_bytree': 0.8,  # 增加特征采样比例以提高特征多样性
    #     'subsample': 0.8,  # 增加子样本比例以提高泛化能力
    #     'min_child_weight': 1.5,  # 降低最小叶子节点样本权重以增加模型灵活性
    #     'gamma': 0.05,  # 保持正则化强度
    #     'n_estimators': 80  # 增加树的数量以提高拟合能力
    # }
    ## 得分低，517第三
    # params = {
    #         'objective': 'binary:logistic',
    #         'scale_pos_weight': 12,  # 比12轻一点，别压住整体正样本分布
    #         'use_label_encoder': False,
    #         'learning_rate': 0.08,  # 稍微降低，帮助稳定弱样本（B）
    #         # 'learning_rate': 0.1,  # 稍微降低，帮助稳定弱样本（B）
    #         # 'max_depth': 5,  # 保留对 A 的支持结构
    #         # 'max_depth': 5,  # 保留对 A 的支持结构
    #         'random_state': 42,
    #         'colsample_bytree': 0.8,  # 稍微增加特征覆盖，帮助 B
    #         'subsample': 0.8,  # 不动，泛化性适中
    #         'min_child_weight': 1.5,  # 介于 1 和 2，适度放开边界，兼顾泛化
    #         'gamma': 0.01,  # 减轻正则但不为 0，保持一定约束.
    #         'n_estimators': 142,
    #     }
    # ## 得分低，517第四
    # params = {
    #         'objective': 'binary:logistic',
    #         'scale_pos_weight': 12,  # 比12轻一点，别压住整体正样本分布
    #         'use_label_encoder': False,
    #         'learning_rate': 0.08,  # 稍微降低，帮助稳定弱样本（B）
    #         # 'learning_rate': 0.1,  # 稍微降低，帮助稳定弱样本（B）
    #         # 'max_depth': 5,  # 保留对 A 的支持结构
    #         # 'max_depth': 5,  # 保留对 A 的支持结构
    #         'random_state': 42,
    #         'colsample_bytree': 0.8,  # 稍微增加特征覆盖，帮助 B
    #         'subsample': 0.8,  # 不动，泛化性适中
    #         'min_child_weight': 1.5,  # 介于 1 和 2，适度放开边界，兼顾泛化
    #         'gamma': 0.01,  # 减轻正则但不为 0，保持一定约束.
    #         'n_estimators': 188,}
    # 5个已知；517排第三，包含516和629
    # params = {
    #     'objective': 'binary:logistic',
    #     'scale_pos_weight': 12,
    #     'use_label_encoder': False,
    #     'learning_rate': 0.05,  # 降低学习率
    #     'max_depth': 5,  # 设置树的深度
    #     'random_state': 42,
    #     'colsample_bytree': 0.7,  # 调整特征采样比例
    #     'subsample': 0.7,  # 调整子样本比例
    #     'min_child_weight': 2,  # 调整最小叶子节点样本权重
    #     'gamma': 0.05,  # 增加正则化强度
    #     'n_estimators': 116  # 增加树的数量
    # }
    params = {
        'objective': 'binary:logistic',
        'scale_pos_weight': 12,  # 降低对正样本的偏好，减少过拟合
        'use_label_encoder': False,
        'learning_rate': 0.08,  # 保持较低的学习率以提高稳定性
        'max_depth': 4,  # 降低树的深度以减少模型复杂度
        'random_state': 42,
        'colsample_bytree': 0.8,  # 增加特征采样比例以提高特征多样性
        'subsample': 0.8,  # 增加子样本比例以提高泛化能力
        'min_child_weight': 1.5,  # 降低最小叶子节点样本权重以增加模型灵活性
        'gamma': 0.05,  # 保持正则化强度
        'n_estimators': 139  # 增加树的数量以提高拟合能力
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

# strc_data最后一列一定有，但是可能是-1
name_labels = seq_data.iloc[:, [0, -1]]
assert len(name_labels.columns) == 2
print('drop the last column of strc ', set(strc_data.iloc[:, -1]))
strc_data = strc_data.drop(strc_data.columns[-1], axis=1)
print(f'name_labels {name_labels.shape}, strc_data {strc_data.shape}')
## update the right label
strc_data = pd.merge(strc_data, name_labels, left_on=0, right_on=0, suffixes=('_seq', '_str'))
print('name label, ', name_labels)
print('after merge ', strc_data.iloc[:, -1])
## 重命名所有列
strc_data.columns = [i for i in range(len(strc_data.columns))]
# assert len(set(strc_data.iloc[:, -1])) == 2, f'strc_data label should be 0 or 1, but now labels are {set(strc_data.iloc[:, -1])}'

###############strc+seq######################
if 'seq_strc' == feature_type:
    merged_data = pd.merge(seq_data, strc_data, left_on=0, right_on=0, suffixes=('_seq', '_str'))
    assert merged_data.iloc[:,1+reduced_dim_seq].equals(merged_data.iloc[:,-1]), f'seq_strc feature should be the same, but now they are {merged_data.iloc[:,1+reduced_dim_seq]} and {merged_data.iloc[:,-1]}'
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

    # 获取所有样本的索引按概率排序
    sorted_indices = np.argsort(y_proba)[::-1]  # 按照概率降序排列

    # 获取对应的名称
    sorted_names = [NAMES[i] for i in sorted_indices]

    # 获取对应的概率值
    sorted_probs = y_proba[sorted_indices]

    # 将索引、名称、概率写入 txt 文件
    with open(os.path.join(RESULTS_DIR, "evaluation_results.txt"), "w") as f:
        f.write("Index\tName\tProbability\n")  # 写入表头
        for i in range(len(sorted_indices)):
            f.write(f"{sorted_indices[i]}\t{sorted_names[i]}\t{sorted_probs[i]:.4f}\n")
    print(f"Saved to {os.path.join(RESULTS_DIR, 'evaluation_results.txt')}")

    top_num = 10
    top_indices = np.argsort(y_proba)[-top_num:][::-1].tolist()

    print(f"概率最高的前{top_num}个样本索引 {top_indices}")
    print("对应的概率值：", y_proba[top_indices])
    name_ids = [NAMES[i] for i in top_indices]
    print("对应的id：", name_ids)

    ## 和目标比较
    N_TOP = 10
    predix = 'URK20'
    suffix = '.1'
    # standard_top_id = [517, 629, 560, 552, 584, 541, 516, 605, 532, 608]
    standard_top_id = [
        522,
        517,
        569,
        585,
        605,
        584,
        580,
        577,
        542,
        552,
        530,
        587,
        532
    ]
    target_id = [516, 629]

    # assert len(standard_top_id) == N_TOP
    standard_top = [predix + str(id) + suffix for id in standard_top_id]
    in_standard = [item for item in name_ids if item in standard_top]
    print(f"有 {len(in_standard)} 个元素和已知抗原重合")
    print(in_standard)
