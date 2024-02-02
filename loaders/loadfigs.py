import os
import pickle
import numpy as np
from torch.utils.data import Dataset

FIG_EXTENSIONS = ['.npy', '.npz', '.pkl']

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def find_classes(directory):
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def make_dataset(directory, class_to_idx, extensions):
    instances = []
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_index)
                    instances.append(item)
    return instances

def load_seq(path):
    fig = np.load(path).astype('float32')
    return fig[np.newaxis,:]

class SeqFolder(Dataset):

    def __init__(self,
                 root,
                 loader=load_seq,
                 extensions=None,
                 transform=None,
                 target_transform=None):
        if extensions is None:
            extensions = FIG_EXTENSIONS
        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.classes, self.class_to_idx = find_classes(self.root)
        self.figs = make_dataset(self.root, self.class_to_idx, self.extensions)
        self.targets = [s[1] for s in self.figs]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        path, target = self.figs[index]
        fig = self.loader(path)
        if self.transform is not None:
            fig = self.transform(fig)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return fig, target

    def __len__(self):
        return len(self.figs)