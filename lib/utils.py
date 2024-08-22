import torch
import torch.nn as nn
import torchvision
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    #print(net)
    print('Total number of parameters: %d' % num_params)

def print_network2(net):
    sum =0
    for name,param in net.named_parameters():
        mul=1
        for size_ in param.shape:
            mul*=size_
        sum+=mul
        print('%14s : %s' %(name,param.shape))
    print('Total params:', sum)

class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class color_constency_loss(nn.Module):
    def __init__(self,):
        super(color_constency_loss, self).__init__()

    def forward(self, enhances):  
        plane_avg = enhances.mean((2, 3))  
        col_loss = torch.mean((plane_avg[:, 0] - plane_avg[:, 1]) ** 2
                              + (plane_avg[:, 1] - plane_avg[:, 2]) ** 2
                              + (plane_avg[:, 2] - plane_avg[:, 0]) ** 2)
        return col_loss

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=False):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        model = torchvision.models.vgg16(pretrained=False)  
        pre = torch.load(r'./weight/vgg16.pth')
        model.load_state_dict(pre)
        blocks.append(model.features[:4].eval())
        blocks.append(model.features[4:9].eval())
        blocks.append(model.features[9:16].eval())
        blocks.append(model.features[16:23].eval())
        
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
   
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1)) 
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input-self.mean) / self.std
        target = (target-self.mean) / self.std
    
        if self.resize:
            input  = self.transform(input, mode='bilinear', size=(128, 128), align_corners=False) 
            target = self.transform(target, mode='bilinear', size=(128, 128), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += torch.nn.functional.l1_loss(x, y)
        return loss

