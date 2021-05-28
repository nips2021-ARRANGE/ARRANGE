import os
import json
import argparse
import time
import numpy as np
import copy

import torch
import torch.optim as optim
from config import settings
from data.LoadData import data_loader
from data.LoadData import val_loader
from models import *
from utils import Log
from utils import Restore

decay_epoch = [1000]

def get_arguments():
    parser = argparse.ArgumentParser(description='Incremental')
    parser.add_argument("--sesses", type=int,default='8',help='0 is base train, incremental from 1,2,3,...,8')
    parser.add_argument("--start_sess", type=int,default='1')
    parser.add_argument("--max_epoch", type=int,default='1000')
    parser.add_argument("--batch_size", type=int,default='128')
    parser.add_argument("--dataset", type=str,default='CIFAR100')
    parser.add_argument("--arch", type=str,default='ExpNetQ_v5_3') #quickcnn_v4 ExpNetQ_v1_1
    parser.add_argument("--lr", type=float,default=0.002)#0.01
    parser.add_argument("--r", type=float,default=16)#0.01
    parser.add_argument("--gamma", type=float,default=2.0)#0.01
    #parser.add_argument("--decay_epoch", nargs='+', type=int, default=[40,80])

    return parser.parse_args()


def test(args, network):
    TP = 0.0
    All = 0.0
    val_data = val_loader(args)
    for i, data in enumerate(val_data):
        img, label = data
        img, label = img.cuda(), label.cuda()       
        out, output = network(img, sess=args.sess, Mode='test')
        #out, output = network(img, args.sess)
        _, pred = torch.max(output, dim=1)
        TP += torch.eq(pred, label).sum().float().item()
        All += torch.eq(label, label).sum().float().item()
    
    acc = float(TP)/All
    
    return acc

def test_continue(args, network):
    val_data = val_loader(args)
    acc_list = []
    for sess in range(args.sess+1):
        TP = 0.0
        All = 0.0
        val_data.dataset.Update_Session(sess)
        for i, data in enumerate(val_data):
            img, label = data
            img, label = img.cuda(), label.cuda()       
            out, output = network(img, args.sess, Mode='test')
            _, pred = torch.max(output, dim=1)
            
            
            TP += torch.eq(pred, label).sum().float().item()
            All += torch.eq(label, label).sum().float().item()
    
        acc = float(TP)/All
        acc_list.append(acc)
           
    return acc_list

def acc_list2string(acc_list):
    acc_str=''
    for idx,item in enumerate(acc_list):
        acc_str +='Sess%d: %.3f, '%(idx, item)
    
    return acc_str

def Trans_ACC(args, acc_list):
    ACC = 0
    ACC_A = 0
    ACC_M=0
    num = 0
    for idx, acc in enumerate(acc_list):
        ACC+=acc*settings.CIFAR100_SessLen[idx]
        num+=settings.CIFAR100_SessLen[idx]
        if idx ==args.sess:
            ACC_A+=acc
        else:
            ACC_M+=acc*settings.CIFAR100_SessLen[idx]
    ACC=ACC/num
    ACC_M=ACC_M/(num-settings.CIFAR100_SessLen[idx])
    return ACC, ACC_A, ACC_M


def train(args):
    ACC_list = []
    lr = args.lr
    network = eval(args.arch).OneModel(args, fix_layers=3, fix_fc=False, fw=False) #fc:fc1  fw:sess-1 fc
    network.cuda()
    print(network)
    if args.start_sess>0:
        Restore.load(args, network, filename='Sess0.pth.tar')
        args.sess = args.start_sess-1
        ACC = test(args, network)
        ACC_list.append(ACC)
        print('Sess: %d'%args.sess, 'acc_val: %f'%ACC)
    for sess in range(1, args.sesses+1):
        Restore.load(args, network, filename='Sess'+str(sess)+'.pth.tar')
        #print(network.Alpha)
        #import pdb;pdb.set_trace()
        network.alpha = network.Alpha[sess]
        #network_Old=copy.deepcopy(network)
        args.sess = sess
        #ACC = 0
        Best_ACC = 0  
        loss_list = []
        begin_time = time.time()
        #print(network.alpha.data[sess])
        ACC_Sess = test_continue(args, network)
        ACC_Sess_str = acc_list2string(ACC_Sess)
        ACC, ACC_A, ACC_M = Trans_ACC(args, ACC_Sess)
        print(ACC, ACC_A, ACC_M)
        ACC_list.append(ACC)
    ACC_list.append(np.mean(ACC_list))
    print('ACC:', ACC_list[-1])
    print('End')


if __name__ == '__main__':
    args = get_arguments()
    print('Running parameters:\n')
    print(json.dumps(vars(args), indent=4, separators=(',', ':')))
    train(args)
