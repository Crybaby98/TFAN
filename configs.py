import argparse
import logging
import torch
import random
import numpy as np

def get_parser():
   
    parser = argparse.ArgumentParser(description="TFAN for Few-Shot WF attack")
        
    parser.add_argument('--dataset', type=str, default='AWF900')
    parser.add_argument('--dataset_dir', type=str, default='./datasets/AWF900')
    parser.add_argument("--way", help="number of websites for classification", type=int, default=100)
    parser.add_argument("--shot", help="number of support instances per class", type=int, default=5)
    parser.add_argument("--query", help="number of query instances per class", type=int, default=15)

    parser.add_argument("--no_val", help="directly use the test set for validation", action="store_true")
    parser.add_argument("--val_epoch", help="number of epochs before eval on val", type=int, default=5)
    parser.add_argument("--val_trial", help="number of episodes during validation", type=int, default=1000)
    parser.add_argument("--time_gap", help="days of delay in collecting test data", type=int, default=0)
    
    parser.add_argument("--gpu", help="gpu device", type=int, default=0)
    parser.add_argument("--seed", help="random seed", type=int, default=42)

    parser.add_argument("--lr", help="initial learning rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", help="weight decay for optimizer", type=float, default=2e-3)
    
    parser.add_argument("--gamma", help="learning rate cut scalar", type=float, default=0.1)
    parser.add_argument("--stage", help="number of lr stages", type=int, default=4)
    parser.add_argument("--stage_size", help="number of epochs before lr is cut by gamma", type=int, default=15)
    
    return parser.parse_args()

def get_logger(filename):
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s", datefmt='%m/%d %I:%M:%S')

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def get_info(args):
    
    return f'{args.dataset}_' \
           f'way-{args.way}_' \
           f'shot-{args.shot}_' \
           f'query-{args.query}'
           
def get_augmentation(inp,way,shot,aug_num,max_move):   
    inp = inp.numpy()      
    spt = inp[:way*shot]
    qry = inp[way*shot:]
    
    new_spt = np.empty((0,1,5000), float)
    for index in range(way):  
        ori_fp_per_web = spt[index*shot:(index+1)*shot]
        support_set_per_web = np.empty((0,1,5000), float)  
        for ori_fp in ori_fp_per_web:
            ori_fp = ori_fp[np.newaxis,:]
            support_set_per_web = np.append(support_set_per_web,ori_fp, axis=0)        
            for _ in range(aug_num):
                roll_step = random.randint(-1 * max_move, max_move + 1)
                syn_fp = np.roll(ori_fp, roll_step, axis=2)
                support_set_per_web = np.append(support_set_per_web,syn_fp, axis=0)                             
        new_spt = np.append(new_spt,support_set_per_web, axis=0)
        
    total = np.append(new_spt,qry,axis=0)
    total = torch.from_numpy(total).float()   
    return total

def get_score(acc_list):
    mean = np.mean(acc_list)
    interval = 1.96 * np.sqrt(np.var(acc_list) / len(acc_list))
    return mean, interval
