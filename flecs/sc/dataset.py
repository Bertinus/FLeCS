from torch.utils.data import Dataset
import torch


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
