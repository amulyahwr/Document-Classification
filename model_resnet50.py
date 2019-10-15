import torch
import torch.nn as nn


# putting the whole model together
class DocClassificationHolistic(nn.Module):
    def __init__(self, args, pretrained_model):
        super(DocClassification, self).__init__()
        self.pretrained_model = pretrained_model
        self.num_features = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Linear(self.num_features, args.num_classes)

    def forward(self, batch):
        output = self.pretrained_model(batch)

        return output

class DocClassificationRest(nn.Module):
    def __init__(self, args, pretrained_resnet50, pretrained_holistic):
        super(DocClassificationRest, self).__init__()
        self.pretrained_resnet50 = pretrained_resnet50
        self.pretrained_holistic = pretrained_holistic

        self.dropout = nn.Dropout(p=0.75)

        self.ff1 = nn.Linear(args.num_classes*5, 256)
        self.ff2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 16)

    def forward(self, batch_holistic, batch_header, batch_footer, batch_left_body, batch_right_body):
        output_holistic = torch.softmax(self.pretrained_resnet50(batch_holistic), dim=1)
        output_header = torch.softmax(self.pretrained_holistic(batch_header), dim=1)
        output_footer = torch.softmax(self.pretrained_holistic(batch_footer), dim=1)
        output_left_body = torch.softmax(self.pretrained_holistic(batch_left_body), dim=1)
        output_right_body = torch.softmax(self.pretrained_holistic(batch_right_body), dim=1)

        output_all = torch.cat((output_holistic, output_header, output_footer, output_left_body, output_right_body), dim=1)
        ff1_out = self.dropout(torch.relu(self.ff1(output_all)))
        ff2_out = self.dropout(torch.relu(self.ff2(ff1_out)))
        output = torch.relu(self.output(ff2_out))

        return output
