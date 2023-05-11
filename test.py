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

FINE_TUNED = 0

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = 'cuda'

def test(args):
    #os.getcwd()
    # Read configuration json file
    config_json = args.config

    with open(config_json) as f:
        cfg = json.load(f)

    # Set random seed
    set_all_seed(cfg["seed"])

    model_dir = f'/media/data2/eunju/ids/230426_221224/'
    
    if FINE_TUNED == 1:
        model_path = model_dir + 'ckpts/epoch_37_train_acc_0.9945_test_acc_0.9943.pt'
        model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=2).to(device)
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt)
    else:
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=2).to(device)

    fraud_classifier = Classifier(cfg, model)
    
    for i in range(1,5):
        test_file = f'test_json' + str(i) 
        with open(cfg["eval"][test_file]) as f:
            files = json.load(f)
        testset = IdTestset(files, getattr(torchvision_test_aug(cfg["data"]["img_H"], cfg["data"]["img_W"]), f"get_aug{5}")())
        testloader = DataLoader(testset, batch_size=cfg["eval"]["batch_size"])
        test_acc, test_loss = fraud_classifier.test(testloader)

        msg = f'test_loss: {test_loss:.6f} test_acc: {test_acc:.4f} '
        print(msg)
        with open(model_dir + 'test_result.txt', 'a') as f:
            f.write('='*20)
            f.write(str(i))
            f.write(msg)
            f.write('\n')

'''
    csv_writer.writerow(['test_loss', 'test_acc',
                         'total_elapsed_time'])
    csv_writer.writerow([test_loss, test_acc, elapse_f])
'''        

if __name__ == '__main__':
    print(f'running script {__file__}')

    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', default="./study/vit_timm/config.json") # './study/vit_timm/config.json'
    #parser.add_argument('-p', '--pretrained', default=None)
    args = parser.parse_args()

    test(args=args)


