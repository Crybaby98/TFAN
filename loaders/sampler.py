import numpy as np
from copy import deepcopy
from torch.utils.data import Sampler

# sampler used for meta-train
class episode_sampler(Sampler):

    def __init__(self, dataset, way, shots):
        super().__init__(dataset)

        self.way = way
        self.shots = shots

        class2id = {}
        for i, (fig_path, class_id) in enumerate(dataset.figs):
            if class_id not in class2id:
                class2id[class_id] = []
            class2id[class_id].append(i)

        self.class2id = class2id

    def __iter__(self):

        class2id_copy = deepcopy(self.class2id)
        for class_id in class2id_copy:
            np.random.shuffle(class2id_copy[class_id])

        while len(class2id_copy) >= self.way:
            fig_id_list = []
            class_id_list = list(class2id_copy.keys())
            batch_class_id = np.random.choice(class_id_list, size=self.way, replace=False)

            for shot in self.shots:
                for class_id in batch_class_id:
                    for _ in range(shot):
                        fig_id_list.append(class2id_copy[class_id].pop())

            for class_id in batch_class_id:
                if len(class2id_copy[class_id]) < sum(self.shots):
                    class2id_copy.pop(class_id)

            yield fig_id_list

# sampler used for meta-val and meta-test
class random_sampler(Sampler):

    def __init__(self, dataset, way, shots, trial):
        super().__init__(dataset)

        self.way = way
        self.shot = shots[0]
        self.query_shot = shots[1]
        self.trial = trial

        class2id = {}
        for i, (fig_path, class_id) in enumerate(dataset.figs):
            if class_id not in class2id:
                class2id[class_id] = []
            class2id[class_id].append(i)

        self.class2id = class2id

    def __iter__(self):

        way = self.way
        shot = self.shot
        query_shot = self.query_shot
        trial = self.trial
        class2id_copy = deepcopy(self.class2id)
        class_id_list = list(class2id_copy.keys())

        for i in range(trial):
            id_list = []
            np.random.shuffle(class_id_list)
            picked_class = class_id_list[:way]

            for cat in picked_class:
                np.random.shuffle(class2id_copy[cat])

            for cat in picked_class:
                id_list.extend(class2id_copy[cat][:shot])

            for cat in picked_class:
                id_list.extend(class2id_copy[cat][shot:(shot + query_shot)])

            yield id_list
