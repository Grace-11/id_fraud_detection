import torch
from torch import nn

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

class Classifier():
    def __init__(self, cfg, model):
        self.model = model
        self.criterion = get_loss_model()
        self.optimizer = get_optimizer(cfg, model.parameters())
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