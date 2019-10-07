from copy import deepcopy
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import torch
import scipy.stats

class Metrics():
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def accuracy(self, predictions, labels):
        return accuracy_score(labels, predictions)
