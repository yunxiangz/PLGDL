"""
=============================
LCEClassifier on Iris dataset
=============================

An example of :class:`lce.LCEClassifier`
"""
import math
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils import Bunch
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
import pandas as pd

import csv
import numpy as np

import xgboost as xgb
def read_type(types):
    validList = []
    sheet = pd.read_excel("../data/antigen.xlsx", "pos")
    for row in sheet.index.values:
        type = sheet.iloc[row, 3]
        if type in types and sheet.iloc[row, 0] not in validList:
            validList.append(sheet.iloc[row, 0])
    sheet = pd.read_excel("../data/antigen.xlsx", "neg")
    for row in sheet.index.values:
        type = sheet.iloc[row, 3]
        if type in types and sheet.iloc[row, 0] not in validList:
            validList.append(sheet.iloc[row, 0])
    return validList

def read_our(types,filename,filename1,numour,numesm):
    sheet = pd.read_excel(filename, 'res')
    validList=sheet.values
    attribute_names = sheet.columns.values[2:numour+2]
    sheet = pd.read_excel(filename1, 'res')
    validList1=sheet.values
    esm = {}
    for item in validList1:
        esm[item[1]] = item[2:numesm+2]
    valid_names = read_type(types)
    valid_pdbs=[]
    for item in validList:
        if item[1] in valid_names and item[1] in esm.keys():
            additional = esm[item[1]]
            item = item.tolist()
            for inner in additional:
                item.insert(len(item)-1, inner)
            item = np.array(item)
            valid_pdbs.append(item)
    valid_pdbs = np.array(valid_pdbs)
    attribute_names = np.hstack((attribute_names,numour+sheet.columns.values[2:numesm+2]))

    da=valid_pdbs[0:,2:numour+numesm+2]
    data=[]
    for i in da:
        temp=[]
        for j in i:
            temp.append(float(j))

        data.append(temp)
    target=[]
    for i in valid_pdbs[0:,numour+numesm+2]: #除去第一行的第九列
        target.append(int(i))
    target_names=['0','1']
    real_data = Bunch(data=data, target=target, feature_names= attribute_names, target_names = target_names)
    return real_data

def runmodel(types,filename,filename1,bar,seed,numour,numesm):
    data = read_our(types,filename,filename1,numour,numesm)
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, random_state=seed)

    clf = xgb.XGBClassifier(min_child_weight=1.1,random_state = 0, n_jobs = 8)
    clf.fit(X_train, y_train)

    # Make prediction and generate classification report
    y_pred,scores = clf.predict(X_test)
    print(classification_report(y_test, y_pred))


def writetocsv(row):
    with open("../result/esm_ours.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)

def getspecific(seed=0,i=9,numour=180,numesm=180):
    filename = '../structure/finetune_ourdata_train_{}_{}.xlsx'.format(i * 5, numour)
    filename1 = '../sequence/main_bpags_train_{}.xlsx'.format(numesm)
    print('#########################Bacteria#{}#########################'.format(i))
    types = ['Bacteria']
    ba=runmodel(types, filename, filename1, 0.5, seed, numour, numesm)
    print('#########################Viruses#{}#########################'.format(i))
    types = ['Viruses']
    va=runmodel(types, filename, filename1, 0.5, seed, numour, numesm)

    print('#########################Eukaryota#{}#########################'.format(i))
    types = ['Eukaryota']
    ea=runmodel(types, filename, filename1, 0.50, seed, numour, numesm)
    print('#########################All#{}#########################'.format(i))
    types = ['Bacteria', 'Eukaryota', 'Viruses']
    aa=runmodel(types, filename, filename1, 0.5, seed, numour, numesm)
    return [ba,va,ea,aa]

def computestatics(TN,FP,FN,TP):
    P = TP + FN
    N = FP + TN
    acc = (TP+TN)/(P+N)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    f1= (2*precision*recall)/(precision+recall)
    mcc = (TP*TN-FP*FN)/math.sqrt((TP+FN)*(TP+FP)*(TN+FP)*(TN+FN))
    print("Accuracy: {}".format(acc))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("F1 Score: {}".format(f1))
    print("MCC {}".format(mcc))

getspecific(seed=0, i=9,numour=180,numesm=180)

#computestatics(139, 14, 25, 95)
#computestatics(58, 20, 6, 51)
#computestatics(29, 5, 6, 41)
computestatics(222, 35, 43, 189)