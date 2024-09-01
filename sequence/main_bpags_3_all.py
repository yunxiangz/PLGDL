'''
To run the model :
conda activate GENTRL
go to download, sequence-main
python scripts/extract.py esm2_t33_650M_UR50D ../../Documents/GearNet/sequence/BPAgs246.fasta ../../Documents/GearNet/sequence/BPAgs/ --repr_layers 0 32 33 --include mean per_tok

'''

import torch
import numpy as np
import pandas as pd

import esm
from sklearn.decomposition import PCA
FASTA_PATH = "/data/attributes/pos.fasta" # Path to P62593.fasta
EMB_PATH = "/data/attributes/pos/" # Path to directory of embeddings for P62593.fasta
EMB_LAYER = 33
ys = []
Xs = []
cate = []


types = ['Bacteria','Eukaryota','Viruses']
def read_type(types):
    validList = []
    sheet = pd.read_excel("/data/attributes/antigen.xlsx", "pos")
    for row in sheet.index.values:
        type = sheet.iloc[row, 3]
        if type in types and sheet.iloc[row, 0] not in validList:
            validList.append(sheet.iloc[row, 0])
    sheet = pd.read_excel("/data/attributes/antigen.xlsx", "neg")
    for row in sheet.index.values:
        type = sheet.iloc[row, 3]
        if type in types and sheet.iloc[row, 0] not in validList:
            validList.append(sheet.iloc[row, 0])
    return validList
validList = read_type(types)


for header, _seq in esm.data.read_fasta(FASTA_PATH):
    scaled_effect = header.split('|')[-1]
    if scaled_effect not in validList:
        continue
    ys.append(scaled_effect)
    fn = f'{EMB_PATH}/{str(scaled_effect)}.pt'
    embs = torch.load(fn)
    Xs.append(embs['mean_representations'][EMB_LAYER])
    cate.append(1)
FASTA_PATH = "/data/attributes/neg.fasta" # Path to P62593.fasta
EMB_PATH = "/data/attributes/neg/" # Path to directory of embeddings for P62593.fasta
for header, _seq in esm.data.read_fasta(FASTA_PATH):
    scaled_effect = header.split('|')[-1]
    if scaled_effect not in validList:
        continue
    ys.append(scaled_effect)
    fn = f'{EMB_PATH}/{str(scaled_effect)}.pt'
    embs = torch.load(fn)
    Xs.append(embs['mean_representations'][EMB_LAYER])
    cate.append(0)

Xs = torch.stack(Xs, dim=0).numpy()

ys = np.array(ys).reshape((-1, 1))
cate = np.array(cate).reshape((-1, 1))

for i in range(1,20):
    num_pca_components = 20*i
    pca = PCA(num_pca_components)
    Xs_pca = pca.fit_transform(Xs)

    k1_spss = pca.components_.T
    weight = (np.dot(k1_spss,pca.explained_variance_ratio_))/np.sum(pca.explained_variance_ratio_)
    weighted_weight = weight/np.sum(weight)
    max_location = sorted(enumerate(weighted_weight),key=lambda y:y[1],reverse=True)

    f = open('main_bpags_idxes_{}.txt'.format(num_pca_components),'w')
    idxes = []
    for i in range(num_pca_components):
        idx = max_location[i]
        idxes.append(idx[0])
        f.write(str(idx[0])+'\n')
    f.close()

    tmp = Xs[:,idxes]
    surface = np.hstack((ys,tmp,cate))
    data = pd.DataFrame(surface)
    writer = pd.ExcelWriter('./main_bpags_train_{}.xlsx'.format(num_pca_components))
    data.to_excel(writer, 'res',float_format='%.3f')
    writer.save()
    writer.close()
