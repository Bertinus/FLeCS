from torch.utils.data import Dataset
import torch
import random


class Paul15Dataset(Dataset):
    def __init__(self, adata, shortest_paths, path_length, option="late"):

        assert option in ["early", "late"]

        self.path_length = path_length
        self.adata = adata

        # Remove paths that are too short and shorten paths that are too long
        if option == "late":
            self.shortest_paths = [shortest_p[-self.path_length:] for shortest_p in shortest_paths.values()
                                   if len(shortest_p) >= self.path_length]
        else:
            self.shortest_paths = [shortest_p[:self.path_length] for shortest_p in shortest_paths.values()
                                   if len(shortest_p) >= self.path_length]

    def __len__(self):
        return len(self.shortest_paths)

    def __getitem__(self, idx):
        return torch.Tensor([list(self.adata.X[cell]) for cell in self.shortest_paths[idx]])


class SciplexDataset(Dataset):
    def __init__(self, adata, perts_with_fp):

        self.adata = adata
        self.perts_with_fp = perts_with_fp

        self.pert_paths = []
        for pert in perts_with_fp:
            self.pert_paths.extend([(pert, v) for v in adata.uns[pert + "_shortest_paths"].values()])

        random.Random(0).shuffle(self.pert_paths)

    def __len__(self):
        return len(self.pert_paths)

    def __getitem__(self, idx):
        pert = self.pert_paths[idx][0]
        return self.perts_with_fp[pert], torch.Tensor([list(self.adata.X[cell]) for cell in self.pert_paths[idx][1]])
