import torch

print(torch.__version__)
from torchdrug import transforms, tasks

from enzymeCommission import enzymeCommission

import time
def loaddata(path):
    truncate_transform = transforms.TruncateProtein(max_length=350, random=False)
    protein_view_transform = transforms.ProteinView(view="residue")
    transform = transforms.Compose([truncate_transform, protein_view_transform])

    start_time = time.time()
    dataset = enzymeCommission(path, transform=transform, atom_feature=None,
                                  bond_feature=None)

    end_time = time.time()
    print("Duration of first instantiation: ", end_time - start_time)

    train_set, valid_set, test_set = dataset.split()

    return [dataset,train_set, valid_set, test_set, dataset.pdb_files]
