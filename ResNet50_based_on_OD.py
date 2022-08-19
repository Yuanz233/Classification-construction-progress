import torch
from torch import nn
from torchvision import models

class ResNet50_basedOD(nn.Module):
    def __init__(self,num_class=1000,info_len=6):
        super(ResNet50_basedOD,self).__init__()
        self.num_class = num_class
        self.resnet50 = models.resnet50()
        self.features = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.fc = nn.Linear(info_len+2048,self.num_class)

    def load_weight(self,root):
        state_dict = torch.load(root)
        self.resnet50.load_state_dict(state_dict)
    

    def forward(self,x1,x2):
        output = self.features(x1)
        output = torch.flatten(output,1)
        output = torch.cat((output,x2),dim=1)
        output = self.fc(output)
        return output

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_x1 = torch.ones(1,3,224,224).to(device)
    x2 = torch.ones(1,6).to(device)
    print(x2)
    resnet50_net = ResNet50_basedOD(num_class=10).to(device)
    output_test = resnet50_net(input_x1,x2)
    print(resnet50_net)
    print(resnet50_net.fc)
    print(output_test.shape)






