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
from fraud_classification import Classifier, get_img_size

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
    save_dir = f'/media/data2/eunju/ids/' + cur + \
                'r50_vit' + args.vModel + \
                '_p' + str(args.patch) + '_sd' + str(args.side) + '_so' + str(args.seqOp) + '/'
    log_dir = save_dir + 'logs/'
    model_dir = save_dir + 'ckpts/'
    
    img_H, img_W = get_img_size(args.seqOp, args.side)
    
    # load data
    train_set = IdDataset(cfg, "train", getattr(torchvision_aug(img_H, img_W), f"get_aug{20}")())
    train_sampler = RandomSampler(train_set)
    #train_sampler = ImbalancedDatasetSampler(dataset)
    trainloader = DataLoader(train_set, 
                            batch_size=cfg["train"]["batch_size"],
                            sampler=train_sampler,
                            num_workers=cfg["train"]["num_workers"], 
                            pin_memory=True)
    
    valid_set = IdDataset(cfg, "valid", getattr(torchvision_test_aug(img_H, img_W), f"get_aug{5}")())
    validloader = DataLoader(valid_set, batch_size=cfg["eval"]["batch_size"], num_workers=cfg["eval"]["num_workers"], pin_memory=True)

    testloader_list = []
    for i in range(1,5):
        test_file = f'test_json' + str(i) 
        with open(cfg["eval"][test_file]) as f:
            files = json.load(f)

        test_set = IdTestset(files, getattr(torchvision_test_aug(img_H, img_W), f"get_aug{5}")())
        testloader = DataLoader(test_set, batch_size=cfg["eval"]["batch_size"], num_workers=cfg["eval"]["num_workers"], pin_memory=True)
        testloader_list.append(testloader)

    fraud_classifier = Classifier(cfg, args)

    os.makedirs(log_dir, exist_ok=True)

    # Set CSV log file
    csv_file = open(log_dir + 'vit.csv', 'w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    csv_writer.writerow(['epoch', 'train_loss', 'train_acc', 
                         'valid_loss', 'valid_acc',
                         'test1_loss', 'test1_acc',
                         'test2_loss', 'test2_acc',
                         'test3_loss', 'test3_acc',
                         'test4_loss', 'test4_acc',
                         'test_mean_acc', 'elapsed_time_per_epoch', 'total_elapsed_time'])
    
    epoch_start = 1
    best_mean = 0.      

    time_start = time.time()

    for e in range(epoch_start, cfg["train"]["num_epochs"] + 1):
        time_e_start = time.time()
        test_acc_list = []
        test_loss_list = []

        train_acc, train_loss = fraud_classifier.train_one_epoch(trainloader)
        valid_acc, valid_loss = fraud_classifier.test(validloader)

        train_acc = train_acc * 100
        valid_acc = valid_acc * 100

        for testloader in testloader_list:
            test_acc, test_loss = fraud_classifier.test(testloader)
            test_acc_list.append(test_acc*100)
            test_loss_list.append(test_loss)          

        time_current = time.time()

        elapse_e = time_current - time_e_start
        elapse_f = time_current - time_start

        mean = (test_acc_list[0] + test_acc_list[1] + test_acc_list[2] + test_acc_list[3]) / 4

        msg = f'epoch : {e} ' + \
              f'train_loss: {train_loss:.8f} ' + f'train_acc: {train_acc:.4f} ' + \
              f'valid_loss: {valid_loss:.8f} ' + f'valid_acc: {valid_acc:.4f} ' + \
              f'test1_loss: {test_loss_list[0]:.8f} ' + f'test1_acc: {test_acc_list[0]:.4f} ' + \
              f'test2_loss: {test_loss_list[1]:.8f} ' + f'test2_acc: {test_acc_list[1]:.4f} ' + \
              f'test3_loss: {test_loss_list[2]:.8f} ' + f'test3_acc: {test_acc_list[2]:.4f} ' + \
              f'test4_loss: {test_loss_list[3]:.8f} ' + f'test4_acc: {test_acc_list[3]:.4f} ' + \
              f'test_mean_acc: {mean:.4f} ' + \
              f'elapsed_time_per_epoch: {float2timeformat(elapse_e)} ' + \
              f'total_elapsed_time: {float2timeformat(elapse_f)}'
        print(msg)

        
        
        csv_writer.writerow([e, train_loss, train_acc, valid_loss, valid_acc, 
                             test_loss_list[0], test_acc_list[0],
                             test_loss_list[1], test_acc_list[1],
                             test_loss_list[2], test_acc_list[2],
                             test_loss_list[3], test_acc_list[3],
                             mean, elapse_e, elapse_f])

        # Save model
        if best_mean < mean:
            os.makedirs(model_dir, exist_ok=True)
            best_mean = mean
            file_name = f'e_{e}_tr_{train_acc:.2f}_val_{valid_acc:.2f}_t1_{test_acc_list[0]:.2f}_t2_{test_acc_list[1]:.2f}_t3_{test_acc_list[2]:.2f}_t4_{test_acc_list[3]:.2f}.pt'
            file_path = model_dir + file_name
            torch.save(fraud_classifier.model.state_dict(), file_path)

if __name__ == '__main__':
    
    print(f'running script {__file__}')

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', default="./config.json") # './study/vit_timm/config.json'
    parser.add_argument('-sq', '--seqOp', type=int, default=2)
    parser.add_argument('-pt', '--patch', type=int, default=16)
    parser.add_argument('-sd', '--side', type=int, default=224)
    parser.add_argument('-vm', '--vModel', type=str, default="base", choices=["base", "large"])
    args = parser.parse_args()

    train(args=args)
    '''
    model = timm.create_model('resnet50', pretrained=True, num_classes=2)
    x = torch.randn(1, 3, 224, 224)
    model(x).shape
    '''
