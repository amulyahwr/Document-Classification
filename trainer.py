from tqdm import tqdm

import torch
from torch.autograd import Variable as Var

import numpy as np

class Trainer(object):
    def __init__(self, args, model, criterion, optimizer):
        super(Trainer, self).__init__()
        self.args       = args
        self.model      = model
        self.criterion  = criterion
        self.optimizer  = optimizer
        self.epoch      = 0

    # helper function for training
    def train(self, batch, labels):
        self.model.train()
        self.optimizer.zero_grad()

        if self.args.cuda:
            labels = labels.cuda()
            batch = batch.cuda()

        labels = labels.long()
        batch = torch.unsqueeze(batch, dim=1).repeat(1,3,1,1).float()

        output = self.model(batch)

        loss = self.criterion(output, labels)
        
        (loss/self.args.batchsize).backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


        return loss

    # helper function for testing
    def test(self, batch, labels):
        with torch.no_grad():
            self.model.eval()

            if self.args.cuda:
                labels = labels.cuda()
                batch = batch.cuda()

            labels = labels.long()
            batch = torch.unsqueeze(batch, dim=1).repeat(1,3,1,1).float()

            output = self.model(batch)

            loss = self.criterion(output, labels)

            scores = torch.softmax(output, dim=1)
            predictions = torch.argmax(scores, dim=1)
            predictions = predictions.cpu().numpy()
            labels = labels.cpu().numpy()

        return loss, predictions, labels
