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
                         'valid_loss', 'valid_acc',
                         'test1_loss', 'test1_acc',
                         'test2_loss', 'test2_acc',
                         'test3_loss', 'test3_acc',
                         'test4_loss', 'test4_acc',
                         'elapsed_time_per_epoch', 'total_elapsed_time'])
    
    # load data
    train_set = IdDataset(cfg, "train", getattr(torchvision_aug(cfg["data"]["img_H"], cfg["data"]["img_W"]), f"get_aug{20}")())
    train_sampler = RandomSampler(train_set)
    #train_sampler = ImbalancedDatasetSampler(dataset)
    trainloader = DataLoader(train_set, 
                            batch_size=cfg["train"]["batch_size"],
                            sampler=train_sampler)
    
    valid_set = IdDataset(cfg, "valid", getattr(torchvision_test_aug(cfg["data"]["img_H"], cfg["data"]["img_W"]), f"get_aug{5}")())
    validloader = DataLoader(valid_set, batch_size=cfg["eval"]["batch_size"])

    testloader_list = []
    for i in range(1,5):
        test_file = f'test_json' + str(i) 
        with open(cfg["eval"][test_file]) as f:
            files = json.load(f)

        test_set = IdTestset(files, getattr(torchvision_test_aug(cfg["data"]["img_H"], cfg["data"]["img_W"]), f"get_aug{5}")())
        testloader = DataLoader(test_set, batch_size=cfg["eval"]["batch_size"])
        testloader_list.append(testloader)

    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2).to(device) # vit_base_patch16_224

    fraud_classifier = Classifier(cfg, model)
    
    epoch_start = 0
    best_mean = 0.  
    test_acc_list = []
    test_loss_list = []

    time_start = time.time()

    model.train() # <---- training mode

    for e in range(epoch_start, cfg["train"]["num_epochs"]):
        time_e_start = time.time()
        
        train_acc, train_loss = fraud_classifier.train_one_epoch(trainloader)
        valid_acc, valid_loss = fraud_classifier.test(validloader)

        for testloader in testloader_list:
            test_acc, test_loss = fraud_classifier.test(testloader)
            test_acc_list.append(test_acc)
            test_loss_list.append(test_loss)          

        time_current = time.time()

        elapse_e = time_current - time_e_start
        elapse_f = time_current - time_start

        msg = f'epoch : {e} ' + \
              f'train_loss: {train_loss:.8f} ' + \
              f'train_acc: {train_acc*100:.4f} ' + \
              f'valid_loss: {valid_loss:.8f} ' + \
              f'valid_acc: {valid_acc*100:.4f} ' + \
              f'test1_loss: {test_loss[0]:.8f} ' + \
              f'test1_acc: {test_acc[0]*100:.4f} ' + \
              f'test2_loss: {test_loss[1]:.8f} ' + \
              f'test2_acc: {test_acc[1]*100:.4f} ' + \
              f'test3_loss: {test_loss[2]:.8f} ' + \
              f'test3_acc: {test_acc[2]*100:.4f} ' + \
              f'test4_loss: {test_loss[3]:.8f} ' + \
              f'test4_acc: {test_acc[3]*100:.4f} ' + \
              f'elapsed_time_per_epoch: {float2timeformat(elapse_e)} ' + \
              f'total_elapsed_time: {float2timeformat(elapse_f)}'
        print(msg)
        
        csv_writer.writerow([e, train_loss, train_acc, valid_loss, valid_acc, 
                             test_loss[0], test_acc[0],
                             test_loss[1], test_acc[1],
                             test_loss[2], test_acc[2],
                             test_loss[3], test_acc[3],
                             elapse_e, elapse_f])

        # Save model
        mean = (train_acc + valid_acc + test_acc[0] + test_acc[1] + test_acc[2] + test_acc[3]) / 6
        if best_mean < mean:
            best_mean = mean
            file_name = f'e_{e}_tr_acc_{train_acc*100:.2f}_val_acc_{valid_acc*100:.2f}_t1_acc_{test_acc[0]*100:.2f}_t2_acc_{test_acc[1]*100:.2f}_t3_acc_{test_acc[2]*100:.2f}_t4_acc_{test_acc[3]*100:.2f}.pt'
            file_path = model_dir + file_name
            torch.save(fraud_classifier.model.state_dict(), file_path)

if __name__ == '__main__':
    print(f'running script {__file__}')

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', default="./config.json") # './study/vit/config.json'
    #parser.add_argument('-p', '--pretrained', default=None)
    args = parser.parse_args()

    train(args=args)




