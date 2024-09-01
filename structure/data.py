import torch

from enzymeCommissionOurVal import EnzymeCommissionOurVal

print(torch.__version__)
from torchdrug import transforms, tasks

from enzymeCommissionOur import EnzymeCommissionOur

from torchdrug import data
from torchdrug import layers
from torchdrug.layers import geometry
import time
def loaddata(path):
    import json
    truncate_transform = transforms.TruncateProtein(max_length=350, random=False)
    protein_view_transform = transforms.ProteinView(view="residue")
    transform = transforms.Compose([truncate_transform, protein_view_transform])

    start_time = time.time()
    dataset = EnzymeCommissionOur(path, transform=transform, atom_feature=None,
                                  bond_feature=None)

    end_time = time.time()
    print("Duration of first instantiation: ", end_time - start_time)

    train_set, valid_set, test_set = dataset.split()

    return [dataset,train_set, valid_set, test_set,dataset.pdb_files]

def loaddataVal():
    import json
    truncate_transform = transforms.TruncateProtein(max_length=350, random=False)
    protein_view_transform = transforms.ProteinView(view="residue")
    transform = transforms.Compose([truncate_transform, protein_view_transform])

    start_time = time.time()
    dataset = EnzymeCommissionOurVal("../owndata/", transform=transform, atom_feature=None,
                                  bond_feature=None)

    end_time = time.time()
    print("Duration of first instantiation: ", end_time - start_time)

    train_set, valid_set, test_set = dataset.split()

    return [dataset,train_set, valid_set, test_set,dataset.pdb_files]