import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from torchdrug import models, tasks
from torchdrug import layers
from torchdrug.layers import geometry

from MultipleBinaryCLassificationOur import MultipleBinaryClassificationOur
from data import loaddata
from torchdrug import core
from engineOur import EngineOur

[dataset,train_set, valid_set, test_set, pdbfiles] = loaddata("../protein-datasets/")

names = []
category = []
for i in range(len(pdbfiles)):
   tmp = pdbfiles[i].split('/')[0]
   if tmp != 'train':
       continue
   names.append(pdbfiles[i].split('-')[1])
   category.append(-1)


names = (np.array(names)).reshape((-1,1))
category = (np.array(category)).reshape((-1,1))


graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()],
                                                    edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                 geometry.KNNEdge(k=10, min_distance=5),
                                                                 geometry.SequentialEdge(max_distance=2)],
                                                    edge_feature="gearnet")

def runmodel(epoch):
    gearnet = torch.load("/data/scratch/protein_output/Unsupervised/AlphaFoldDB/MultiviewContrast/2023-05-16-09-12-14/model_epoch_{}.pth".format(epoch))
    task = MultipleBinaryClassificationOur(gearnet, graph_construction_model=graph_construction_model, num_mlp_layer=3,
                                              task=[_ for _ in range(len(dataset.tasks))], criterion="bce", metric=["auprc@micro", "f1_max"])
    optimizer = torch.optim.Adam(task.parameters(), lr=1e-4)
    solver = EngineOur(task, train_set, valid_set, test_set, optimizer,
                         gpus=[0], batch_size=4)
    task.model.load_state_dict(torch.load("/data/scratch/protein_output/Unsupervised/AlphaFoldDB/MultiviewContrast/2023-05-16-09-12-14/model_parameter_epoch_{}.pth".format(epoch)),strict=True)

    res3 = solver.evaluate("train")
    res3 = res3['graph_feature'].cpu().numpy()

    for index in range(170,176,5):
        numour = index
        f = open('./finetune_ourdata_idxes_{}_{}.txt'.format(epoch,numour), 'r')
        idxes = f.readlines()
        for i in range(len(idxes)):
            idxes[i] = int(idxes[i].replace('\n', ''))
        tmp = res3[:, idxes]

        surface = np.hstack((names, tmp, category))
        data = pd.DataFrame(surface)
        writer = pd.ExcelWriter('./finetune_bpags_test_{}_{}.xlsx'.format(epoch,numour))
        data.to_excel(writer, 'res'.format(epoch),float_format='%.3f')
        writer.save()
        writer.close()


for i in range(1,11):
    runmodel(i*5)
    print(i*5)