import math

import pandas as pd

import csv
import numpy as np

import xgboost as xgb
from numpy import array
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV


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


def read_homology():
    validList = []
    sheet = pd.read_excel("..//data/bpags_homology.xlsx", "res")
    for row in sheet.index.values:
        name = sheet.iloc[row, 0]
        name = name.replace('.1','')
        names = name.split('|')
        if len(names)>1:
            validList.append(names[1])
        else:
            validList.append(names[0])
    return validList


def read_our(types,del_homology,filename,filename1,trainset,numour,numesm):
    sheet = pd.read_excel(filename, 'res')
    validList=sheet.values
    names = validList[:,1]

    sheet = pd.read_excel(filename1, 'res')
    validList1=sheet.values
    esm = {}
    esmY = {}
    for item in validList1:
        esm[item[1]] = item[2:numesm+2]
        esmY[item[1]] = item[numesm+2]
    valid_pdbs=[]
    for item in validList:
        if item[1] in esm.keys():
            additional = esm[item[1]]
            item = item.tolist()
            for inner in additional:
                item.insert(len(item)-1, inner)
            item.insert(len(item)-1, esmY[item[1]])
            item = np.array(item)
            valid_pdbs.append(item)
    valid_pdbs = np.array(valid_pdbs)

    da=valid_pdbs[0:,2:numour+numesm+2]
    y_valid = valid_pdbs[:, numour+numesm+2]

    data=[]
    for i in da:
        temp=[]
        for j in i:
            temp.append(float(j))
        data.append(temp)
    data = np.array(data).tolist()
    y_valid = np.array(y_valid).astype(int).tolist()
    names = np.array(names).tolist()
    if del_homology:
        homologylist = read_homology()
        for idx,name in reversed(list(enumerate(names))):
            if name in homologylist:
                del data[idx]
                del y_valid[idx]
                del names[idx]
    if trainset:
        valid_names = read_type(types)
        for idx, name in reversed(list(enumerate(names))):
            if name not in valid_names:
                del data[idx]
                del y_valid[idx]
                del names[idx]
    return data, y_valid, np.array(names)

def runmodel(types,del_homology,filename,filename1,trainfilename,trainfilename1,numour,numems):
    X_test, y_test, name_test = read_our(types,False,filename,filename1,False,numour,numems)
    X_train, y_train, name_test1 = read_our(types, del_homology,trainfilename, trainfilename1,True,numour,numems)

    # Train Classifier with default parameters
    clf = xgb.XGBClassifier(min_child_weight=1.1,random_state=0, n_jobs=8)
    clf.fit(X_train, y_train)
    # Make prediction and generate classification report
    y_pred,scores = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

def writetocsv(row):
    with open("../result/esm_ours_bpags.csv", "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row)


def getspecific():
    for i in range(9,10):
        numour = 180
        numesm = 180
        trainfilename = '../structure/finetune_ourdata_train_{}_{}.xlsx'.format(i*5,numour)
        trainfilename1 = '../sequence/main_bpags_train_{}.xlsx'.format(numesm)

        filename='../structure/finetune_bpags_test_{}_{}.xlsx'.format(i*5,numour)
        filename1='../sequence/main_bpags_test_{}.xlsx'.format(numesm)
        print('#########################All###{}#######################'.format(i*5))
        types = ['Bacteria','Eukaryota','Viruses']
        del_homology = True
        runmodel(types,del_homology,filename,filename1,trainfilename,trainfilename1,numour,numesm)

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

getspecific()
computestatics(91, 23, 24, 102)
