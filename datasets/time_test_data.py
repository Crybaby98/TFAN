import numpy as np
import os
import json
from tqdm import tqdm
from collections import Counter

with open('datasets/split_with_val.json', 'r') as json_file:
    websites = json.load(json_file)
test_websites = websites['test']

name = ['3d','10d','2w','4w','6w']
day = ['3','10','14','28','42']

for i in range(len(name)):
    
    dataset_name = 'datasets/tor_time_test' + name[i] + '_200w_100tr'
    target_dir = os.path.join('datasets/AWF900/test', day[i])

    os.system('rm -rf ' + dataset_name)
    os.system('rm -rf ' + target_dir)

    print('===================================================')
    print('tor_time_test'+ name[i] + '_200w_100tr.npz')
    print('===================================================')
    
    print('Dataset unzipping......')

    os.makedirs(dataset_name)
    content = np.load(dataset_name + '.npz', allow_pickle=True)
    fingerprints = content['data']
    labels = content['labels']
    websites = Counter(labels).keys()
    for cnt,web in enumerate(websites):
        n = [index for index,label in enumerate(labels) if label == web]
        np.save(file=web + '.npy', arr=fingerprints[n])
        os.system('mv ' + web + '.npy' + ' ' + dataset_name + '/')

    print('Dataset unzip ready!')

    print('Dataset building......')
    
    os.makedirs(target_dir)
    for web in tqdm(test_websites):
        class_path = os.path.join(target_dir, web)
        os.makedirs(class_path)
        origin_file = os.path.join(dataset_name, web)
        load = np.load(origin_file+'.npy')
        for index,data in enumerate(load):
            index = str(index+1).zfill(3)
            filename = web + '_' + index + '.npy'
            np.save(filename,data)
            os.system('mv ' + filename + ' ' + class_path + '/')

    print('Dataset build ready!\n')
    
    os.system('rm -rf ' + dataset_name)
