from __future__ import division
from __future__ import print_function

import os
import random
import logging

import torch.optim as optim
import numpy as np

# NEURAL NETWORK MODULES/LAYERS
import model_vgg16
import model_vgg19
import model_resnet50
import model_densenet121
import model_inceptionv3

from dataset import Dataset
# METRICS CLASS FOR EVALUATION
from metrics import Metrics
# CONFIG PARSER
from config import parse_args
# TRAIN AND TEST HELPER FUNCTIONS
from trainer import Trainer
import glob
from tqdm import tqdm

import torchvision
from torchvision import models
import torch
import torch.nn as nn

# MAIN BLOCK
def main():
    global args
    args = parse_args()

    # argument validation
    args.cuda = args.cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    print(args)
    train_dir = glob.glob(os.path.join(args.data,'train/holistic/*.pt'))
    dev_dir = glob.glob(os.path.join(args.data,'val/holistic/*.pt'))
    test_dir = glob.glob(os.path.join(args.data,'test/holistic/*.pt'))

    train_dataset = Dataset(os.path.join(args.data,'train'), train_dir)
    dev_dataset = Dataset(os.path.join(args.data,'val'), dev_dir)
    test_dataset = Dataset(os.path.join(args.data,'test'), test_dir)

    print('==> Size of train data   : %d ' % len(train_dataset))
    print('==> Size of val data   : %d ' % len(dev_dataset))
    print('==> Size of test data   : %d ' % len(test_dataset))


    # initialize model, criterion/loss_function, optimizer
    if args.pretrained_model == 'vgg16':
        pretrained_vgg16 = models.vgg16(pretrained=True)

        # Freeze training for all layers
        for child in pretrained_vgg16.children():
            for param in child.parameters():
                param.requires_grad = False

        if args.pretrained_holistic == 0:
            model = model_vgg16.DocClassificationHolistic(args, pretrained_vgg16)
        elif args.pretrained_holistic == 1:
            pretrained_orig_vgg16 = model_vgg16.DocClassificationHolistic(args, pretrained_vgg16)
            pretrained_holistic = model_vgg16.DocClassificationHolistic(args, pretrained_orig_vgg16.pretrained_model)
            checkpoint = torch.load('./checkpoints/vgg16.pt')
            pretrained_holistic.load_state_dict(checkpoint['model'])

            model = model_vgg16.DocClassificationRest(args, pretrained_orig_vgg16, pretrained_holistic)

    elif args.pretrained_model == 'vgg19':
        pretrained_vgg19 = models.vgg19(pretrained=True)

        # Freeze training for all layers
        for child in pretrained_vgg19.children():
            for param in child.parameters():
                param.requires_grad = False

        if args.pretrained_holistic == 0:
            model = model_vgg19.DocClassificationHolistic(args, pretrained_vgg19)
        elif args.pretrained_holistic == 1:
            pretrained_orig_vgg19 = model_vgg19.DocClassificationHolistic(args, pretrained_vgg19)
            pretrained_holistic = model_vgg19.DocClassificationHolistic(args, pretrained_orig_vgg19.pretrained_model)
            checkpoint = torch.load('./checkpoints/vgg19.pt')
            pretrained_holistic.load_state_dict(checkpoint['model'])

            model = model_vgg19.DocClassificationRest(args, pretrained_orig_vgg19, pretrained_holistic)

    elif args.pretrained_model == 'resnet50':
        pretrained_resnet50 = models.resnet50(pretrained=True)

        # Freeze training for all layers
        for child in pretrained_resnet50.children():
            for param in child.parameters():
                param.requires_grad = False

        if args.pretrained_holistic == 0:
            model = model_resnet50.DocClassificationHolistic(args, pretrained_resnet50)
        elif args.pretrained_holistic == 1:
            pretrained_orig_resnet50 = model_resnet50.DocClassificationHolistic(args, pretrained_resnet50)
            pretrained_holistic = model_resnet50.DocClassificationHolistic(args, pretrained_orig_resnet50.pretrained_model)
            checkpoint = torch.load('./checkpoints/resnet50.pt')
            pretrained_holistic.load_state_dict(checkpoint['model'])

            model = model_resnet50.DocClassificationRest(args, pretrained_orig_resnet50, pretrained_holistic)

    elif args.pretrained_model == 'densenet121':
        pretrained_densenet121 = models.densenet121(pretrained=True)

        # Freeze training for all layers
        for child in pretrained_densenet121.children():
            for param in child.parameters():
                param.requires_grad = False

        if args.pretrained_holistic == 0:
            model = model_densenet121.DocClassificationHolistic(args, pretrained_densenet121)
        elif args.pretrained_holistic == 1:
            pretrained_orig_densenet121 = model_densenet121.DocClassificationHolistic(args, pretrained_densenet121)
            pretrained_holistic = model_densenet121.DocClassificationHolistic(args, pretrained_orig_densenet121.pretrained_model)
            checkpoint = torch.load('./checkpoints/densenet121.pt')
            pretrained_holistic.load_state_dict(checkpoint['model'])

            model = model_densenet121.DocClassificationRest(args, pretrained_orig_densenet121, pretrained_holistic)

    elif args.pretrained_model == 'inceptionv3':
        pretrained_inceptionv3 = models.inception_v3(pretrained=True)

        # Freeze training for all layers
        for child in pretrained_inceptionv3.children():
            for param in child.parameters():
                param.requires_grad = False

        if args.pretrained_holistic == 0:
            model = model_inceptionv3.DocClassificationHolistic(args, pretrained_inceptionv3)
        elif args.pretrained_holistic == 1:
            pretrained_orig_inceptionv3 = model_inceptionv3.DocClassificationHolistic(args, pretrained_inceptionv3)
            pretrained_holistic = model_inceptionv3.DocClassificationHolistic(args, pretrained_orig_inceptionv3.pretrained_model)
            checkpoint = torch.load('./checkpoints/inceptionv3.pt')
            pretrained_holistic.load_state_dict(checkpoint['model'])

            model = model_inceptionv3.DocClassificationRest(args, pretrained_orig_inceptionv3, pretrained_holistic)

    criterion = nn.CrossEntropyLoss(reduction='sum')

    parameters = filter(lambda p: p.requires_grad, model.parameters())

    if args.cuda:
        model.cuda(), criterion.cuda()

    if args.optim=='adam':
        optimizer   = optim.Adam(parameters, lr=args.lr, weight_decay=args.wd)
    elif args.optim=='adagrad':
        optimizer   = optim.Adagrad(parameters, lr=args.lr, weight_decay=args.wd)
    elif args.optim=='sgd':
        optimizer   = optim.SGD(parameters, lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adadelta':
        optimizer = optim.Adadelta(parameters, lr=args.lr, weight_decay=args.wd)
    metrics = Metrics(args.num_classes)

    # create trainer object for training and testing
    trainer = Trainer(args, model, criterion, optimizer)

    train_idx = list(np.arange(len(train_dataset)))
    dev_idx = list(np.arange(len(dev_dataset)))
    test_idx = list(np.arange(len(test_dataset)))

    best = float('inf')
    columns = ['ExpName','ExpNo', 'Epoch', 'Loss','Accuracy']
    results = []
    early_stop_count = 0

    for epoch in range(args.epochs):

        train_loss = 0.0
        dev_loss = 0.0
        test_loss = 0.0

        train_predictions = []
        train_labels = []

        dev_predictions = []
        dev_labels = []

        test_predictions = []
        test_labels = []

        random.shuffle(train_idx)
        random.shuffle(dev_idx)
        random.shuffle(test_idx)

        batch_train_data = [train_idx[i:i + args.batchsize] for i in range(0, len(train_idx), args.batchsize)]
        batch_dev_data = [dev_idx[i:i + args.batchsize] for i in range(0, len(dev_idx), args.batchsize)]
        batch_test_data = [test_idx[i:i + args.batchsize] for i in range(0, len(test_idx), args.batchsize)]

        for batch in tqdm(batch_train_data, desc='Training batches..'):
            train_batch_holistic, \
            train_batch_header, \
            train_batch_footer, \
            train_batch_left_body, \
            train_batch_right_body, \
            train_batch_labels = train_dataset[batch]

            if args.pretrained_holistic == 0:
                _ = trainer.train_holistic(train_batch_holistic, train_batch_labels)
            elif args.pretrained_holistic == 1:
                _ = trainer.train_rest(train_batch_holistic, \
                                        train_batch_header, \
                                        train_batch_footer, \
                                        train_batch_left_body, \
                                        train_batch_right_body, \
                                        train_batch_labels)


        for batch in tqdm(batch_train_data, desc='Training batches..'):
            train_batch_holistic, \
            train_batch_header, \
            train_batch_footer, \
            train_batch_left_body, \
            train_batch_right_body, \
            train_batch_labels = train_dataset[batch]

            if args.pretrained_holistic == 0:
                train_batch_loss, train_batch_predictions, train_batch_labels = trainer.test_holistic(train_batch_holistic, train_batch_labels)
            elif args.pretrained_holistic == 1:
                train_batch_loss, train_batch_predictions, train_batch_labels = trainer.test_rest(train_batch_holistic, \
                                                                                            train_batch_header, \
                                                                                            train_batch_footer, \
                                                                                            train_batch_left_body, \
                                                                                            train_batch_right_body, \
                                                                                            train_batch_labels)

            train_predictions.append(train_batch_predictions)
            train_labels.append(train_batch_labels)
            train_loss = train_loss + train_batch_loss

        train_accuracy = metrics.accuracy(np.concatenate(train_predictions), np.concatenate(train_labels))

        for batch in tqdm(batch_dev_data, desc='Dev batches..'):
            dev_batch_holistic, \
            dev_batch_header, \
            dev_batch_footer, \
            dev_batch_left_body, \
            dev_batch_right_body, \
            dev_batch_labels = dev_dataset[batch]

            if args.pretrained_holistic == 0:
                dev_batch_loss, dev_batch_predictions, dev_batch_labels = trainer.test_holistic(dev_batch_holistic, dev_batch_labels)
            elif args.pretrained_holistic == 1:
                dev_batch_loss, dev_batch_predictions, dev_batch_labels = trainer.test_rest(dev_batch_holistic, \
                                                                                        dev_batch_header, \
                                                                                        dev_batch_footer, \
                                                                                        dev_batch_left_body, \
                                                                                        dev_batch_right_body, \
                                                                                        dev_batch_labels)


            dev_predictions.append(dev_batch_predictions)
            dev_labels.append(dev_batch_labels)
            dev_loss = dev_loss + dev_batch_loss

        dev_accuracy = metrics.accuracy(np.concatenate(dev_predictions), np.concatenate(dev_labels))

        for batch in tqdm(batch_test_data, desc='Test batches..'):
            test_batch_holistic, \
            test_batch_header, \
            test_batch_footer, \
            test_batch_left_body, \
            test_batch_right_body, \
            test_batch_labels = test_dataset[batch]

            if args.pretrained_holistic == 0:
                test_batch_loss, test_batch_predictions, test_batch_labels = trainer.test_holistic(test_batch_holistic, test_batch_labels)
            elif args.pretrained_holistic == 1:
                test_batch_loss, test_batch_predictions, test_batch_labels = trainer.test_rest(test_batch_holistic, \
                                                                                        test_batch_header, \
                                                                                        test_batch_footer, \
                                                                                        test_batch_left_body, \
                                                                                        test_batch_right_body, \
                                                                                        test_batch_labels)


            test_predictions.append(test_batch_predictions)
            test_labels.append(test_batch_labels)
            test_loss = test_loss + test_batch_loss

        test_accuracy = metrics.accuracy(np.concatenate(test_predictions), np.concatenate(test_labels))

        print('==> Training Epoch: %d, \
                        \nLoss: %f, \
                        \nAccuracy: %f'%(epoch + 1, \
                                            train_loss/(len(batch_train_data) * args.batchsize), \
                                            train_accuracy))
        print('==> Dev Epoch: %d, \
                        \nLoss: %f, \
                        \nAccuracy: %f'%(epoch + 1, \
                                            dev_loss/(len(batch_dev_data) * args.batchsize), \
                                            dev_accuracy))

        print('==> Test Epoch: %d, \
                        \nLoss: %f, \
                        \nAccuracy: %f'%(epoch + 1, \
                                            test_loss/(len(batch_test_data) * args.batchsize), \
                                            test_accuracy))
        #quit()
        results.append((args.expname, \
                        args.expno, \
                        epoch+1, \
                        test_loss/(len(batch_test_data) * args.batchsize), \
                        test_accuracy))

        if best > test_loss:
            best = test_loss
            checkpoint = {'model': trainer.model.state_dict(), 'optim': trainer.optimizer,
                          'loss': test_loss, 'accuracy': test_accuracy,
                          'args': args, 'epoch': epoch }
            print('==> New optimum found, checkpointing everything now...')
            torch.save(checkpoint, '%s.pt' % os.path.join(args.save, args.expname))
            #np.savetxt("test_pred.csv", test_pred.numpy(), delimiter=",")
        else:
            early_stop_count = early_stop_count + 1

            if early_stop_count == 20:
                quit()

if __name__ == "__main__":
    main()
