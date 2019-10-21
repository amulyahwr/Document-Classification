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
    def train_holistic(self, batch_holistic, labels):
        self.model.train()
        self.optimizer.zero_grad()

        if self.args.cuda:
            labels = labels.cuda()
            batch_holistic = batch_holistic.cuda()

        labels = labels.long()
#         print('batch_holistic.shape:',batch_holistic.shape)
        batch_holistic = torch.unsqueeze(batch_holistic, dim=1).repeat(1,3,1,1).float()
#         print('batch_holistic.shape:',batch_holistic.shape)
        output = self.model(batch_holistic)

#         print(labels)
        loss = self.criterion(output, labels)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


        return loss

    # helper function for testing
    def test_holistic(self, batch_holistic, labels):
        with torch.no_grad():
            self.model.eval()

            if self.args.cuda:
                labels = labels.cuda()
                batch_holistic = batch_holistic.cuda()

            labels = labels.long()
            batch_holistic = torch.unsqueeze(batch_holistic, dim=1).repeat(1,3,1,1).float()

            output = self.model(batch_holistic)

            loss = self.criterion(output, labels)

            scores = torch.softmax(output, dim=1)
            predictions = torch.argmax(scores, dim=1)
            predictions = predictions.cpu().numpy()
            labels = labels.cpu().numpy()

        return loss, predictions, labels

    # helper function for training
    def train_rest(self, batch_holistic, batch_header, batch_footer, batch_left_body, batch_right_body, labels):
        self.model.train()
        self.optimizer.zero_grad()

        if self.args.cuda:
            labels = labels.cuda()
            batch_holistic = batch_holistic.cuda()
            batch_header = batch_header.cuda()
            batch_footer = batch_footer.cuda()
            batch_left_body = batch_left_body.cuda()
            batch_right_body = batch_right_body.cuda()


        labels = labels.long()
        batch_holistic = torch.unsqueeze(batch_holistic, dim=1).repeat(1,3,1,1).float()
        batch_header = torch.unsqueeze(batch_header, dim=1).repeat(1,3,1,1).float()
        batch_footer = torch.unsqueeze(batch_footer, dim=1).repeat(1,3,1,1).float()
        batch_left_body = torch.unsqueeze(batch_left_body, dim=1).repeat(1,3,1,1).float()
        batch_right_body = torch.unsqueeze(batch_right_body, dim=1).repeat(1,3,1,1).float()

        output = self.model(batch_holistic, batch_header, batch_footer, batch_left_body, batch_right_body)

        loss = self.criterion(output, labels)

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()


        return loss

    # helper function for testing
    def test_rest(self, batch_holistic, batch_header, batch_footer, batch_left_body, batch_right_body, labels):
        with torch.no_grad():
            self.model.eval()

            if self.args.cuda:
                labels = labels.cuda()
                batch_holistic = batch_holistic.cuda()
                batch_header = batch_header.cuda()
                batch_footer = batch_footer.cuda()
                batch_left_body = batch_left_body.cuda()
                batch_right_body = batch_right_body.cuda()

            abels = labels.long()
            batch_holistic = torch.unsqueeze(batch_holistic, dim=1).repeat(1,3,1,1).float()
            batch_header = torch.unsqueeze(batch_header, dim=1).repeat(1,3,1,1).float()
            batch_footer = torch.unsqueeze(batch_footer, dim=1).repeat(1,3,1,1).float()
            batch_left_body = torch.unsqueeze(batch_left_body, dim=1).repeat(1,3,1,1).float()
            batch_right_body = torch.unsqueeze(batch_right_body, dim=1).repeat(1,3,1,1).float()

            output = self.model(batch_holistic, batch_header, batch_footer, batch_left_body, batch_right_body)

            loss = self.criterion(output, labels)

            scores = torch.softmax(output, dim=1)
            predictions = torch.argmax(scores, dim=1)
            predictions = predictions.cpu().numpy()
            labels = labels.cpu().numpy()

        return loss, predictions, labels
