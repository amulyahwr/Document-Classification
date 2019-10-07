from __future__ import division
from __future__ import print_function

import os
import random
import logging

import torch.optim as optim
import numpy as np

# NEURAL NETWORK MODULES/LAYERS
from model import *
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

# MAIN BLOCK
def main():
    global args
    args = parse_args()
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    # file logger
    fh = logging.FileHandler(os.path.join(args.save, args.expname)+'.log', mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # argument validation
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.sparse and args.wd!=0:
        logger.error('Sparsity and weight decay are incompatible, pick one!')
        exit()
    logger.debug(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    train_dir = glob.glob(os.path.join(args.data,'train/holistic/*.pt'))
    dev_dir = glob.glob(os.path.join(args.data,'val/holistic/*.pt'))
    test_dir = glob.glob(os.path.join(args.data,'test/holistic/*.pt'))

    train_dataset = Dataset(os.path.join(args.data,'train'), train_dir)
    dev_dataset = Dataset(os.path.join(args.data,'val'), dev_dir)
    test_dataset = Dataset(os.path.join(args.data,'test'), test_dir)

    logger.debug('==> Size of train data   : %d ' % len(train_dataset))
    logger.debug('==> Size of val data   : %d ' % len(dev_dataset))
    logger.debug('==> Size of test data   : %d ' % len(test_dataset))


    # initialize model, criterion/loss_function, optimizer
    if args.pretrained_model == 'vgg16':
        pretrained_model = models.vgg16(pretrained=True)

    # Freeze training for all layers
    for param in pretrained_model.features.parameters():
        param.requires_grad = False

    for param in pretrained_model.classifier.parameters():
        param.requires_grad = False

    model = DocClassification(args, pretrained_model)
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
            batch_holistic, _, _, _, _, batch_labels = train_dataset[batch]
            _ = trainer.train(batch_holistic, batch_labels)

        for batch in tqdm(batch_train_data, desc='Training batches..'):
            train_batch_holistic, _, _, _, _, train_batch_labels = train_dataset[batch]
            train_batch_loss, train_batch_predictions, train_batch_labels = trainer.test(train_batch_holistic, train_batch_labels)

            train_predictions.append(train_batch_predictions)
            train_labels.append(train_batch_labels)
            train_loss = train_loss + train_batch_loss

        train_accuracy = metrics.accuracy(np.concatenate(train_predictions), np.concatenate(train_labels))

        for batch in tqdm(batch_dev_data, desc='Dev batches..'):
            dev_batch_holistic, _, _, _, _, dev_batch_labels = dev_dataset[batch]
            dev_batch_loss, dev_batch_predictions, dev_batch_labels = trainer.test(dev_batch_holistic, dev_batch_labels)

            dev_predictions.append(dev_batch_predictions)
            dev_labels.append(dev_batch_labels)
            dev_loss = dev_loss + dev_batch_loss

        dev_accuracy = metrics.accuracy(np.concatenate(dev_predictions), np.concatenate(dev_labels))

        for batch in tqdm(batch_test_data, desc='Test batches..'):
            test_batch_holistic, _, _, _, _, test_batch_labels = test_dataset[batch]
            test_batch_loss, test_batch_predictions, test_batch_labels = trainer.test(test_batch_holistic, test_batch_labels)

            test_predictions.append(test_batch_predictions)
            test_labels.append(test_batch_labels)
            test_loss = test_loss + test_batch_loss

        test_accuracy = metrics.accuracy(np.concatenate(test_predictions), np.concatenate(test_labels))

        logger.info('==> Training Epoch {}, \
                        \nLoss: {}, \
                        \nAccuracy: {}'.format(epoch + 1, \
                                            train_loss/(len(batch_train_data) * args.batchsize), \
                                            train_accuracy))
        logger.info('==> Dev Epoch {}, \
                        \nLoss: {}, \
                        \nAccuracy: {}'.format(epoch + 1, \
                                            dev_loss/(len(batch_dev_data) * args.batchsize), \
                                            dev_accuracy))

        logger.info('==> Test Epoch {}, \
                        \nLoss: {}, \
                        \nAccuracy: {}'.format(epoch + 1, \
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
            logger.debug('==> New optimum found, checkpointing everything now...')
            torch.save(checkpoint, '%s.pt' % os.path.join(args.save, args.expname))
            #np.savetxt("test_pred.csv", test_pred.numpy(), delimiter=",")
        else:
            early_stop_count = early_stop_count + 1

            if early_stop_count == 20:
                quit()

if __name__ == "__main__":
    main()
