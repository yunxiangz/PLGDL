import os
os.environ['CUDA-VISIBLE-DEVICES']='0'
import numpy as np
import pandas as pd
import torch
print(torch.__version__)
from sklearn.decomposition import PCA
from torchdrug import layers
from torchdrug.layers import geometry

from MultipleBinaryCLassificationZyx import MultipleBinaryClassificationZyx
from data import loaddata
from torchdrug import core
from engineZyx import EngineZyx

[dataset,train_set, valid_set, test_set, pdbfiles] = loaddata("../protein-datasets/")


names = []
category = []
for i in range(len(pdbfiles)):
   tmp = pdbfiles[i].split('/')[0]
   if tmp != 'test':
       continue
   names.append(pdbfiles[i].split('-')[1])
   category.append(1)

for i in range(len(pdbfiles)):
   tmp = pdbfiles[i].split('/')[0]
   if tmp != 'valid':
       continue
   names.append(pdbfiles[i].split('-')[1])
   category.append(0)

names1 = (np.array(names)).reshape((-1,1))
category1 = (np.array(category)).reshape((-1,1))


graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                    edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                 geometry.KNNEdge(k=10, min_distance=5),
                                                                 geometry.SequentialEdge(max_distance=2)],
                                                    edge_feature="gearnet")

def runmodel(epoch):
    gearnet = torch.load("/data/scratch/protein_output/Unsupervised/AlphaFoldDB/MultiviewContrast/2023-05-16-09-12-14/model_epoch_{}.pth".format(epoch))
    task = MultipleBinaryClassificationZyx(gearnet, graph_construction_model=graph_construction_model, num_mlp_layer=3,
                                              task=[_ for _ in range(len(dataset.tasks))], criterion="bce", metric=["auprc@micro", "f1_max"])
    optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)
    solver = EngineZyx(task, train_set, valid_set, test_set, optimizer,
                         gpus=[0], batch_size=4)
    task.model.load_state_dict(torch.load("/data/scratch/protein_output/Unsupervised/AlphaFoldDB/MultiviewContrast/2023-05-16-09-12-14/model_parameter_epoch_{}.pth".format(epoch)),strict=True)

    num_pca_components = 40
    pca = PCA(num_pca_components)
    res1 = solver.evaluate("test")
    res1 = res1['graph_feature'].cpu().numpy()

    res2 = solver.evaluate("valid")
    res2 = res2['graph_feature'].cpu().numpy()

    resall = np.vstack((res1, res2))
    resall = pca.fit_transform(resall)
    result = []
    for i in range(len(resall)):
        tmp = np.hstack((names[i],resall[i],category[i]))
        result.append(tmp)

    k1_spss = pca.components_.T
    weight = (np.dot(k1_spss, pca.explained_variance_ratio_)) / np.sum(pca.explained_variance_ratio_)
    weighted_weight = weight / np.sum(weight)
    max_location = sorted(enumerate(weighted_weight), key=lambda y: y[1], reverse=True)

    f = open('./finetune_ourdata_idxes_{}.txt'.format(epoch), 'w')
    idxes = []
    for i in range(num_pca_components):
        idx = max_location[i]
        idxes.append(idx[0])
        f.write(str(idx[0]) + '\n')
    f.close()

    data = pd.DataFrame(result)
    writer = pd.ExcelWriter('./finetune_ourdata_train_{}.xlsx'.format(epoch))
    data.to_excel(writer, 'res'.format(epoch),float_format='%.3f')
    writer.save()
    writer.close()


def runmodeltemp(epoch):
    gearnet = torch.load("/data/scratch/protein_output/Unsupervised/AlphaFoldDB/MultiviewContrast/2023-05-16-09-12-14/model_epoch_{}.pth".format(epoch))
    task = MultipleBinaryClassificationZyx(gearnet, graph_construction_model=graph_construction_model, num_mlp_layer=3,
                                              task=[_ for _ in range(len(dataset.tasks))], criterion="bce", metric=["auprc@micro", "f1_max"])
    optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)
    solver = EngineZyx(task, train_set, valid_set, test_set, optimizer,
                         gpus=[0], batch_size=4)
    task.model.load_state_dict(torch.load("/data/scratch/protein_output/Unsupervised/AlphaFoldDB/MultiviewContrast/2023-05-16-09-12-14/model_parameter_epoch_{}.pth".format(epoch)),strict=True)

    res1 = solver.evaluate("test")
    res1 = res1['graph_feature'].cpu().numpy()

    res2 = solver.evaluate("valid")
    res2 = res2['graph_feature'].cpu().numpy()

    resall = np.vstack((res1, res2))
    result = []
    for i in range(len(resall)):
        tmp = np.hstack((names[i],resall[i],category[i]))
        result.append(tmp)

    data = pd.DataFrame(result)
    writer = pd.ExcelWriter('./finetune_ourdata_train_1280.xlsx')
    data.to_excel(writer, 'res'.format(epoch),float_format='%.3f')
    writer.save()
    writer.close()

for i in range(9,11):
    runmodel(i*5)