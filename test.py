#from timm_pack.timm import create_model
import os
import argparse
import json

import torch
from torch.utils.data import RandomSampler, DataLoader
from vit import *
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

    model_dir = f'/media/data2/eunju/ids/ckpts/'
    model_path = model_dir + 'epoch_11_train_acc_0.9007870604375767_test_acc_0.8938828259620908.pt'
    
    # load model, loss, optimizer
    vanilla_transformer = Transformer(
         dim = 1024,
         depth = 6, 
         heads = 8, # 16
         mlp_dim = 2048, 
         dropout = 0.1
         )

    model = ViT(
            image_size = (cfg["data"]["img_H"], cfg["data"]["img_W"]),
            patch_size = 32,
            num_classes = 2,
            dim = 1024,
            transformer = vanilla_transformer,
            emb_dropout = 0.1
            ).to(device)
    
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt)

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

    parser.add_argument('-c', '--config', default="./config.json") # './study/vit_timm/config.json'
    #parser.add_argument('-p', '--pretrained', default=None)
    args = parser.parse_args()

    test(args=args)


