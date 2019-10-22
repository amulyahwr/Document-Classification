import torch
import torch.nn as nn


# putting the whole model together
class DocClassificationHolistic(nn.Module):
    def __init__(self, args, pretrained_model):
        super(DocClassificationHolistic, self).__init__()
        self.pretrained_model = pretrained_model
        self.num_features = self.pretrained_model.classifier[6].in_features
        self.features = list(self.pretrained_model.classifier.children())[:-1] # Remove last layer
        self.features.extend([nn.Linear(self.num_features, args.num_classes)]) # Add layer with 16 outputs
        self.pretrained_model.classifier = nn.Sequential(*self.features) # Replace the model classifier

    def forward(self, batch):
        output = self.pretrained_model(batch)

        return output

class DocClassificationRest(nn.Module):

    def __init__(self, args, pretrained_vgg16, pretrained_holistic_model):
        super(DocClassificationRest, self).__init__()
        self.pretrained_vgg16 = DocClassificationHolistic(args, pretrained_vgg16)
        self.pretrained_holistic = DocClassificationHolistic(args, pretrained_vgg16)
        self.pretrained_holistic.load_state_dict(pretrained_holistic_model)

        self.dropout = nn.Dropout(p=0.75)

        self.ff1 = nn.Linear(args.num_classes*5, 256)
        self.ff2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 16)

    def forward(self, batch_holistic, batch_header, batch_footer, batch_left_body, batch_right_body):
        output_holistic = self.pretrained_vgg16(batch_holistic)
        output_header = self.pretrained_holistic(batch_header)
        output_footer = self.pretrained_holistic(batch_footer)
        output_left_body = self.pretrained_holistic(batch_left_body)
        output_right_body = self.pretrained_holistic(batch_right_body)

        output_all = torch.cat((output_holistic, output_header, output_footer, output_left_body, output_right_body), dim=-1)
#         print(output_all.shape)
        ff1_out = self.dropout(torch.relu(self.ff1(output_all)))
        ff2_out = self.dropout(torch.relu(self.ff2(ff1_out)))
        output = torch.relu(self.output(ff2_out))

        return output
