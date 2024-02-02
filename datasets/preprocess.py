import os
import json
import numpy as np
from tqdm import tqdm
from collections import Counter

dataset_name = 'datasets/tor_900w_2500tr'
target_dir = 'datasets/AWF900'

# split = ['train','val','test']
# split_file = 'datasets/split_with_val.json'

split = ['train','test']
split_file = 'datasets/split_no_val.json'

os.system('rm -rf ' + dataset_name)
os.system('rm -rf ' + target_dir)

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

with open(split_file, 'r') as json_file:
    websites = json.load(json_file)

print('Dataset splitting......')

for s in split:
    print('--- split ' + s + ' ---')
    save_path = os.path.join(target_dir, s)
    os.makedirs(save_path)
    cnt = 100 if s == 'test' else 1000
    for web in tqdm(websites[s]):
        class_path = os.path.join(save_path, web)
        os.makedirs(class_path)
        origin_file = os.path.join(dataset_name, web)
        load = np.load(origin_file+'.npy')[100:100+cnt]
        for index,data in enumerate(load):
            index = str(index+1).zfill(3)
            filename = web + '_' + index + '.npy'
            np.save(filename,data)
            os.system('mv ' + filename + ' ' + class_path + '/')

print('Dataset split ready!')

os.system('rm -rf ' + dataset_name)

os.rename('datasets/AWF900/test','datasets/AWF900/0')
os.makedirs('datasets/AWF900/test')
os.system('mv ' + 'datasets/AWF900/0' + ' ' + 'datasets/AWF900/test/')
