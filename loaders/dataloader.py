import os
from torch.utils.data import DataLoader
from loaders.loadfigs import SeqFolder
from loaders.sampler import episode_sampler, random_sampler

# dataloader used for meta-train
def meta_train_dataloader(dataset_path, way, shots):
    dataset = SeqFolder(dataset_path)
    sampler = episode_sampler(dataset=dataset,
                             way=way, shots=shots)
    loader = DataLoader(dataset=dataset,
                        batch_sampler=sampler,
                        num_workers=4,
                        pin_memory=True)
    return loader

# dataloader used for meta-val and meta-test
def meta_test_dataloader(dataset_path, way, shots, trial=1000):
    dataset = SeqFolder(dataset_path)
    sampler = random_sampler(dataset=dataset,
                             way=way, shots=shots,
                             trial=trial)
    loader = DataLoader(dataset=dataset,
                        batch_sampler=sampler,
                        num_workers=4,
                        pin_memory=True)
    return loader