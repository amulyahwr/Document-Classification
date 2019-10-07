import torch
import torch.nn as nn


# putting the whole model together
class DocClassification(nn.Module):
    def __init__(self, args, pretrained_model):
        super(DocClassification, self).__init__()
        self.pretrained_model = pretrained_model

        self.num_features = self.pretrained_model.classifier[6].in_features
        self.features = list(self.pretrained_model.classifier.children())[:-1] # Remove last layer
        self.features.extend([nn.Linear(self.num_features, args.num_classes)]) # Add layer with 16 outputs
        self.pretrained_model.classifier = nn.Sequential(*self.features) # Replace the model classifier

    def forward(self, batch):
        output = self.pretrained_model(batch)

        return output
