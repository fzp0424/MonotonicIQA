from pyexpat import model
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
from BCNN import BCNN
import bc
import argparse


class fc_abs(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(fc_abs, self).__init__() 
        self.fc = nn.Linear(in_channels, out_channels,bias =True)

    def forward(self, x):
        kernel_abs = abs(self.fc.weight)
        #print(kernel_abs)
        out_log = F.linear(input=x, weight=kernel_abs)
        return out_log

def init_linear(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class BaseCNN(nn.Module):
    def __init__(self, config):
        """Declare all needed layers."""
        nn.Module.__init__(self)

        self.config = config
        outdim = 1

        if self.config.backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=True)
        elif self.config.backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=True)
        elif self.config.backbone == 'resnet50':
            self.backbone = bc.resnet50(pretrained=True)

        if config.representation == 'BCNN':
            # assert ((self.config.backbone == 'resnet18') | (self.config.backbone == 'resnet34')), "The backbone network must be resnet18 or resnet34"
            self.representation = BCNN()
            self.fc = nn.Sequential(nn.Linear(512 * 512, 512),
                                    nn.ReLU(),
                                    nn.Linear(512,64),
                                    nn.ReLU(),
                                    nn.Linear(64,outdim))    
            self.fc.apply(init_linear)       
        else:
            self.fc = nn.Sequential(nn.Linear(512 * 6 * 6, 512),
                        nn.ReLU(),
                        nn.Linear(512,64),
                        nn.ReLU(),
                        nn.Linear(64,outdim))    
            self.fc.apply(init_linear)       

        self.nlm1 = nn.Sequential(fc_abs(1, 10), nn.ELU(), fc_abs(10, 20), nn.ELU(), fc_abs(20, 20), nn.ELU(), fc_abs(20, 10),nn.ELU(), fc_abs(10, 1)) 
        self.nlm2 = nn.Sequential(fc_abs(1, 10), nn.ELU(), fc_abs(10, 20), nn.ELU(), fc_abs(20, 20), nn.ELU(), fc_abs(20, 10),nn.ELU(), fc_abs(10, 1)) 
        self.nlm3 = nn.Sequential(fc_abs(1, 10), nn.ELU(), fc_abs(10, 20), nn.ELU(), fc_abs(20, 20), nn.ELU(), fc_abs(20, 10),nn.ELU(), fc_abs(10, 1)) 
        self.nlm4 = nn.Sequential(fc_abs(1, 10), nn.ELU(), fc_abs(10, 20), nn.ELU(), fc_abs(20, 20), nn.ELU(), fc_abs(20, 10),nn.ELU(), fc_abs(10, 1)) 
        self.nlm5 = nn.Sequential(fc_abs(1, 10), nn.ELU(), fc_abs(10, 20), nn.ELU(), fc_abs(20, 20), nn.ELU(), fc_abs(20, 10),nn.ELU(), fc_abs(10, 1)) 
        self.nlm6 = nn.Sequential(fc_abs(1, 10), nn.ELU(), fc_abs(10, 20), nn.ELU(), fc_abs(20, 20), nn.ELU(), fc_abs(20, 10),nn.ELU(), fc_abs(10, 1)) 


    def forward(self, x, tag, istest=False):
        
       if not istest :
            [b,d,c,h,w] = x.shape#batch_size num_tag channels height width 
            y = []
            x = x.view(-1,c,h,w)
            DNN_x = []

            with torch.autograd.set_detect_anomaly(True):

                x = self.backbone.conv1(x)
                x = self.backbone.bn1(x)
                x = self.backbone.relu(x)
                x = self.backbone.maxpool(x)
                x = self.backbone.layer1(x)
                x = self.backbone.layer2(x)
                x = self.backbone.layer3(x)
                x = self.backbone.layer4(x)


                if self.config.representation == 'BCNN':
                    x = self.representation(x)
                else:
                    x = self.backbone.avgpool(x)
                    x = torch.flatten(x, start_dim=1)


                x = self.fc(x)
                x = x.view(b,d)
                DNN_x = x

                x = torch.split(x, 1, dim=1)# dim=1 namely num_tag, 6 species in total

                y = []
                for sap_id in range(len(x)):
                   x_data = x[sap_id] 
                   y_out = getattr(self, 'nlm'+str(sap_id+1), 'None')(x_data).squeeze(1)
                   y.append(y_out)   
                y = torch.stack(y,dim=0) #add dim0 to y 
                y = torch.transpose(y,1,0) 

                return y,DNN_x
       else:
            
            with torch.autograd.set_detect_anomaly(True):
                b,c,h,w = x.shape
                DNN_x = []
                x = self.backbone.conv1(x)
                x = self.backbone.bn1(x)
                x = self.backbone.relu(x)
                x = self.backbone.maxpool(x)
                x = self.backbone.layer1(x)
                x = self.backbone.layer2(x)
                x = self.backbone.layer3(x)
                x = self.backbone.layer4(x)


                if self.config.representation == 'BCNN':
                    x = self.representation(x)

                else:
                    x = self.backbone.avgpool(x)
                    x = torch.flatten(x, start_dim=1)


                x = self.fc(x)
                DNN_x = x
                y = torch.zeros_like(x)

                for i in range(b):
                    tag_ = tag[i].data.item()
                    y[i] = getattr(self, 'nlm'+str(tag_), 'None')(x).squeeze()      
                
                return y,DNN_x



def test():

    config = config()
    net = BaseCNN(config)
    net.cuda()
 
    x1 = torch.randn(2,6,3,384,384)
    x1 = Variable(x1.cuda())

    x2 = [i for i in range(6)]
    x2 = (torch.tensor(x2)).unsqueeze(0)
    x2 = x2.repeat(2,1)
    x2 = Variable(x2.cuda())
    
    # x21 = torch.tensor([1,1]).cuda()
    # x22 = torch.tensor([3,3]).cuda()
    # x2 = []
    # x2.append(x21)
    # x2.append(x22)
   
    y1= net.forward(x1,x2)


    print(y1,len(y1),x2.shape,y1.shape,y1[0].shape)

   
if __name__== '__main__':
    test()   
