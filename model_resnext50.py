import torch
import torch.nn as nn
# putting_the_whole_model_together
class DocClassificationHolistic(nn.Module):
    def __init__(self,args,pretrained_model):
        super(DocClassificationHolistic,self).__init__()
        self.pretrained_model=pretrained_model
        self.num_features=self.pretrained_model.fc.in_features
        self.pretrained_model.fc=nn.Linear(self.num_features,args.num_classes)
    def forward(self,batch):
        output=self.pretrained_model(batch)
        return output

class DocClassificationRest(nn.Module):
    def __init__(self,args,pretrained_holistic):
        super(DocClassificationRest,self).__init__()
        self.pretrained_holistic=pretrained_holistic
    def forward(self,batch_holistic,batch_header,batch_footer,batch_left_body,batch_right_body):
        # output_holistic=self.pretrained_resnet50(batch_holistic)
        # output_header=self.pretrained_holistic(batch_header)
        output_footer=self.pretrained_holistic(batch_footer)
        # output_left_body=self.pretrained_holistic(batch_left_body)
        # output_right_body=self.pretrained_holistic(batch_right_body)
        # output_all=torch.cat((output_holistic,output_header,output_footer,output_left_body,output_right_body),dim=1)
        # ff1_out=self.dropout(torch.relu(self.ff1(output_all)))
        # ff2_out=self.dropout(torch.relu(self.ff2(ff1_out)))
        # output=torch.relu(self.output(ff2_out))
        return output_footer
    
class DocClassification3DAttention(nn.Module):
    def __init__(self,args,pretrained_model):
        super(DocClassification3DAttention,self).__init__()
        self.pretrained_model=pretrained_model
       
        self.bottleneck172 = list(self.pretrained_model.children())[:-2]
        
        self.pretrained_model = nn.Sequential(*self.bottleneck172)
        self.in_channels = 1
        self.depth = 2048
        self.m = 1
        
        self.V = nn.Conv3d(self.in_channels, self.m, (self.depth, 3, 3), stride=(self.depth, 1, 1), padding=(0,1,1))
        self.fc = nn.Linear(self.m*self.depth, 256)
        self.out = nn.Linear(256, 16)
        
        #self.pretrained_model.fc=nn.Linear(self.num_features,args.num_classes)
    def forward(self,batch):

        U = self.pretrained_model(batch)
        U = torch.unsqueeze(U, dim=1)
        
        V = self.V(U)
      
        attn_map = torch.exp(V)/torch.sum(torch.exp(V))
       
        zn = torch.sum(torch.sum(attn_map * U, dim=-1), dim=-1)
        zn = torch.squeeze(zn, dim=1)
        fc = torch.relu(self.fc(zn))
        out = torch.softmax(self.out(fc), dim=-1)
        
        return out, attn_map