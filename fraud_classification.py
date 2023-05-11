import torch
from torch import nn
import timm
import numpy as np
#from timm_pack.timm import create_model
#from timm.models.vision_transformer import _cfg, VisionTransformer

def get_loss_model():
    return nn.CrossEntropyLoss()

def get_optimizer(cfg, params):
    if cfg["optimizer"]["type"] == "adam":
        optimizer = torch.optim.Adam(params,
                        lr=cfg["optimizer"]["lr"],
                        weight_decay=cfg["optimizer"]["weight_decay"])
    elif cfg["optimizer"]["type"] == "sgd":
        optimizer = torch.optim.SGD(params,
                        lr=cfg["optimizer"]["lr"],
                        weight_decay=cfg["optimizer"]["weight_decay"],
                        momentum=cfg["optimizer"]["momentum"])
    return optimizer

def get_lr_scheduler(cfg, optimizer):
    return torch.optim.lr_scheduler.StepLR(optimizer,
                        step_size=cfg["optimizer"]["step_size"], gamma=cfg["optimizer"]["gamma"])

def get_img_size(seq, side):
    if seq == 0:  # only vit
        img_H = side
        img_W = side
    else:            # only conv or conv + vit
        img_H = 512
        img_W = 800
    return img_H, img_W   

class Resize(nn.Module):
    def __init__(self, *args):
        super(Resize, self).__init__()
        self.shape = args
    
    def forward(self, x):
        W = x.shape[2]
        H = x.shape[3]
        side = self.shape[0]

        rx = np.random.randint(W - side)
        ry = np.random.randint(H - side)

        return x[:,:,rx:rx+side, ry:ry+side] 

def get_concatenator(seq, side, conv_model, vit_model):
    model = None

    '''
        for name, param in conv_model.named_parameters():
            print(name)
    '''
    
    if seq == 0:             # 224 (h) x 224 (w) x 3 (ch)
        print("vit only")
        model = vit_model

    elif seq == 1:           # don't care, 3 (ch)
        print("conv only")
        model = conv_model

    elif seq == 2:           # 256 (h) x 400 (w) x 64 (ch) -> 224 (h) x 224 (w) x 3 (ch)
        print("conv, resize, 1x1 conv, vit")
        model = nn.Sequential(
            conv_model.conv1,
            Resize(side, side),
            nn.Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1), bias=False),
            vit_model)
        
    elif seq == 3:           # 256 (h) x 400 (w) x 64 (ch) -> 224 (h) x 224 (w) x 3 (ch)
        print("conv, bn, act, resize, 1x1 conv, vit")
        model = nn.Sequential(
            conv_model.conv1,
            conv_model.bn1,
            conv_model.act1,
            Resize(side, side),
            nn.Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1), bias=False),
            vit_model)
        
    elif seq == 4:           # 128 (h) x 200 (w) x 64 (ch) -> 224 (h) x 224 (w) x 3 (ch)
        print("conv, bn, act, maxpool, resize, 1x1 conv, vit")
        model = nn.Sequential(
            conv_model.conv1,
            conv_model.bn1,
            conv_model.act1,
            conv_model.maxpool,
            nn.Conv2d(64, 3, kernel_size=(1, 1), stride=(1, 1), bias=False),
            vit_model)
    else:
        assert(False)
        
    return model

class Classifier():
    def __init__(self, cfg, args):
        print(args.seqOp, args.patch, args.side)
        conv_model = timm.create_model('resnet50', pretrained=True)

        vit_name = f'vit_'+ args.vModel + '_patch' + str(args.patch) + '_' + str(args.side)
        vit_model = timm.create_model(vit_name, pretrained=True, num_classes=2) # vit_base_patch16_224
        
        model = get_concatenator(args.seqOp, args.side, conv_model, vit_model)
        assert(model != None)

        self.model = model.to('cuda')
        self.criterion = get_loss_model()
        self.optimizer = get_optimizer(cfg, self.model.parameters())
        self.lr_scheduler = get_lr_scheduler(cfg, self.optimizer)
        
    def train_one_epoch(self, trainloader):      # 현재 cutmix 사용 안 함. 
        self.model.train()
        acc = 0.
        losses = 0.

        for data, label in trainloader:
            data = data.cuda()
            label = label.cuda()

            # train model            
            out = self.model(data)
            loss = self.criterion(out, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            acc += self.calculate_accuracy(out, label) # should divide it by the batch size
            losses += loss.item()

        data_size = len(trainloader.dataset)
        acc /= data_size
        losses /= data_size

        self.lr_scheduler.step()

        return acc, losses
    
    @torch.no_grad()
    def calculate_accuracy(self, out, label):
        acc = 0.
        batch = out.shape[0]
        for i in range(batch):        
            #r = nn.functional.softmax(o, dim=1)[0] # in fact, don't need to do softmax in classification
            r = torch.argmax(out[i])
            g = 1 if r == label[i] else 0
            acc += g
        return acc

    @torch.no_grad()
    def test(self, testloader):
        self.model.eval()
        acc = 0.
        losses = 0.

        for data, label in testloader:
            data = data.cuda()
            label = label.cuda()

            out = self.model(data)
            loss = self.criterion(out, label)

            acc += self.calculate_accuracy(out, label)
            losses += loss.item()

        data_size = len(testloader.dataset)
        acc /= data_size
        losses /= data_size

        self.model.train()
        return acc, losses
