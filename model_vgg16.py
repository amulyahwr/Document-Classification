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
        #self.pretrained_vgg16 = pretrained_vgg16
        self.pretrained_holistic = pretrained_holistic_model

#         self.dropout = nn.Dropout(p=0.75)
#         self.ff1 = nn.Linear(args.num_classes*5, 256)
#         self.ff2 = nn.Linear(256, 256)
#         self.output = nn.Linear(256, 16)

    def forward(self, batch_holistic, batch_header, batch_footer, batch_left_body, batch_right_body):
        output_holistic = self.pretrained_holistic(batch_holistic)
#         output_header = torch.softmax(self.pretrained_holistic(batch_header), dim=-1)
#         output_footer = torch.softmax(self.pretrained_holistic(batch_footer), dim=-1)
#         output_left_body = torch.softmax(self.pretrained_holistic(batch_left_body), dim=-1)
#         output_right_body = torch.softmax(self.pretrained_holistic(batch_right_body), dim=-1)

#         output_all = torch.cat((output_holistic, output_header, output_footer, output_left_body, output_right_body), dim=-1)

#         ff1_out = torch.relu(self.ff1(output_all))

#         ff2_out = torch.relu(self.ff2(ff1_out))

#         output = self.output(ff2_out)
        output = output_holistic

        return output
class DocClassificationEnsemble(nn.Module):

    def __init__(self, args, pretrained_holistic_model
                                              , pretrained_header_model
                                              , pretrained_footer_model
                                              , pretrained_left_body_model
                                              , pretrained_right_body_model):
        super(DocClassificationEnsemble, self).__init__()
        self.pretrained_holistic = pretrained_holistic_model
        self.pretrained_header = pretrained_header_model
        self.pretrained_footer = pretrained_footer_model
        self.pretrained_left_body = pretrained_left_body_model
        self.pretrained_right_body = pretrained_right_body_model

        self.dropout = nn.Dropout(p=0.75)
        self.ff1 = nn.Linear(args.num_classes*5, 256)
        self.ff2 = nn.Linear(256, 256)
        self.output = nn.Linear(256, 16)

    def forward(self, batch_holistic, batch_header, batch_footer, batch_left_body, batch_right_body):
        output_holistic = torch.softmax(self.pretrained_holistic(batch_holistic), dim=-1)
        output_header = torch.softmax(self.pretrained_header(batch_header,None,None,None,None), dim=-1)
        output_footer = torch.softmax(self.pretrained_footer(batch_footer,None,None,None,None), dim=-1)
        output_left_body = torch.softmax(self.pretrained_left_body(batch_left_body,None,None,None,None), dim=-1)
        output_right_body = torch.softmax(self.pretrained_right_body(batch_right_body,None,None,None,None), dim=-1)

        output_all = torch.cat((output_holistic, output_header, output_footer, output_left_body, output_right_body), dim=-1)

        ff1_out = torch.relu(self.ff1(output_all))

        ff2_out = torch.relu(self.ff2(ff1_out))

        output = self.output(ff2_out)
#         output = output_right_body

        return output