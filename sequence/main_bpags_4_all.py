#https://github.com/facebookresearch/esm/blob/main/examples/sup_variant_prediction.ipynb

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

data_gd = pd.read_excel('/data/attributes/bpags_gt.xlsx', sheet_name='Sheet1')
FASTA_PATH = "/data/attributes/BPAgs246.fasta" # Path to P62593.fasta
EMB_PATH = "/data/attributes/BPAgs/" # Path to directory of embeddings for P62593.fasta
EMB_LAYER = 33
ys = []
Xs = []
cate = []
for header, _seq in esm.data.read_fasta(FASTA_PATH):
    scaled_effect = header.split('|')[-1]
    ys.append(scaled_effect)
    fn = f'{EMB_PATH}/{str(scaled_effect)}.pt'
    embs = torch.load(fn)
    Xs.append(embs['mean_representations'][EMB_LAYER])
    label = -1
    for row_gt in data_gd.index.values:
        id_gt = data_gd.iloc[row_gt, 3]
        gt = data_gd.iloc[row_gt, 0]
        if scaled_effect == id_gt:
            label = gt
            break
    cate.append(label)

Xs = torch.stack(Xs, dim=0).numpy()

ys = np.array(ys).reshape((-1,1))
cate = np.array(cate).reshape((-1,1))

for idx in range(1,20):
    numesm = 20*idx
    f = open('main_bpags_idxes_{}.txt'.format(numesm),'r')
    idxes = f.readlines()
    for i in range(len(idxes)):
        idxes[i] = int(idxes[i].replace('\n',''))
    tmp = Xs[:,idxes]

    surface = np.hstack((ys,tmp,cate))
    data = pd.DataFrame(surface)
    writer = pd.ExcelWriter('./main_bpags_test_{}.xlsx'.format(numesm))
    data.to_excel(writer, 'res',float_format='%.3f')
    writer.save()
    writer.close()
