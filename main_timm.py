#from timm_pack.timm import create_model
import timm
import os
import argparse
import time
import json
import datetime
import csv

import torch
from torch.utils.data import RandomSampler, DataLoader
from util import *
from id_dataset import *
from fraud_classification import Classifier

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

device = 'cuda'

def train(args):
    #os.getcwd()
    # Read configuration json file
    config_json = args.config

    with open(config_json) as f:
        cfg = json.load(f)

    # Set random seed
    set_all_seed(cfg["seed"])

    # Set logger
    cur = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    save_dir = f'/media/data2/eunju/ids/' + cur + '/'
    log_dir = save_dir + 'logs/'
    model_dir = save_dir + 'ckpts/'

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # Set CSV log file
    csv_file = open(log_dir + 'vit.csv', 'w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    csv_writer.writerow(['epoch', 'train_loss', 'train_acc', 
                         'test_loss', 'test_acc',
                         'elapsed_time_per_epoch', 'total_elapsed_time'])
    
    # load data
    train_set = IdDataset(cfg, "train", getattr(torchvision_aug(384, 384), f"get_aug{20}")())
    train_sampler = RandomSampler(train_set)
    #train_sampler = ImbalancedDatasetSampler(dataset)
    trainloader = DataLoader(train_set, 
                            batch_size=cfg["train"]["batch_size"],
                            sampler=train_sampler)
    
    test_set = IdDataset(cfg, "valid", getattr(torchvision_test_aug(384, 384), f"get_aug{5}")())
    testloader = DataLoader(test_set, batch_size=cfg["eval"]["batch_size"])

    model = timm.create_model('vit_base_patch16_384', pretrained=True, num_classes=2).to(device) # vit_base_patch16_224

    fraud_classifier = Classifier(cfg, model)
    
    epoch_start = 0
    best_mean = 0.  

    time_start = time.time()

    model.train() # <---- training mode

    for e in range(epoch_start, cfg["train"]["num_epochs"]):
        time_e_start = time.time()
        
        train_acc, train_loss = fraud_classifier.train_one_epoch(trainloader)
        test_acc, test_loss = fraud_classifier.test(testloader)

        time_current = time.time()

        elapse_e = time_current - time_e_start
        elapse_f = time_current - time_start

        msg = f'epoch : {e} ' + \
              f'train_loss: {train_loss:.6f} ' + \
              f'train_acc: {train_acc:.4f} ' + \
              f'test_loss: {test_loss:.6f} ' + \
              f'test_acc: {test_acc:.4f} ' + \
              f'elapsed_time_per_epoch: {float2timeformat(elapse_e)} ' + \
              f'total_elapsed_time: {float2timeformat(elapse_f)}'
        print(msg)
        
        csv_writer.writerow([e, train_loss, train_acc, test_loss, test_acc, elapse_e, elapse_f])

        # Save model
        mean = (train_acc + test_acc) / 2
        if best_mean < mean:
            best_mean = mean
            file_name = f'epoch_{e}_train_acc_{train_acc:.4f}_test_acc_{test_acc:.4f}.pt'
            file_path = model_dir + file_name
            torch.save(fraud_classifier.model.state_dict(), file_path)

if __name__ == '__main__':
    print(f'running script {__file__}')

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', default="./config.json") # './study/vit/config.json'
    #parser.add_argument('-p', '--pretrained', default=None)
    args = parser.parse_args()

    train(args=args)




