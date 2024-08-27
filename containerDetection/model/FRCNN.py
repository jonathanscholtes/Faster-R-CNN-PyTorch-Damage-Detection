import torchvision
import torch
from torchvision.ops import RoIPool
from torchvision import transforms, models, datasets
#from torchvision.models.vgg import model_urls

class FRCNN(torch.nn.Module):
    def __init__(self,dropout=.4):
        super().__init__()
       # model_urls['vgg16_bn'] = model_urls['vgg16_bn'].replace('https://', 'http://')
        rawnet = torchvision.models.vgg16_bn( weights=torchvision.models.VGG16_BN_Weights.DEFAULT)
        for param in rawnet.features.parameters():
            param.requires_grad = True
        self.seq = torch.nn.Sequential(*list(rawnet.features.children())[:-1])
        self.roipool = RoIPool(7, spatial_scale=14/244)
        feature_dim = 512*7*7
        self.cls_score = torch.nn.Linear(feature_dim, 2)
        #self.theta_score = torch.nn.Linear(feature_dim, 1)
        self.theta_score = torch.nn.Sequential(
              torch.nn.Linear(feature_dim, 512),
              torch.nn.ReLU(),
              torch.nn.Linear(512, 1),
              torch.nn.Tanh(),
            )
        self.dropout_layer = torch.nn.Dropout(dropout) 
        self.bbox = torch.nn.Sequential(
              torch.nn.Linear(feature_dim, 512),
              torch.nn.ReLU(),
              torch.nn.Linear(512, 4),
              torch.nn.Tanh(),
            )
        self.cel = torch.nn.CrossEntropyLoss()
        self.sl1 = torch.nn.L1Loss()
        self.thetaloss = torch.nn.L1Loss()
    def forward(self, input, rois, ridx):
        res = input
        res = self.seq(res)
        rois = torch.cat([ridx.unsqueeze(-1), rois*244], dim=-1)
        res = self.roipool(res, rois)
        feat = res.view(len(res), -1)
        cls_score = self.cls_score(feat)        
        theta_score = self.theta_score(feat)
        bbox = self.bbox(feat) # .view(-1, len(label2target), 4)
        return cls_score,theta_score, bbox
    def calc_loss(self, probs, pred_theta, _deltas, labels,theta, deltas):
        detection_loss = self.cel(probs, labels)
        
        ixs, = torch.where(labels != 0)
        _deltas = _deltas[ixs]
        deltas = deltas[ixs]
        pred_theta = pred_theta[ixs]
        theta = theta[ixs]
        self.lmb = 10.0
        if len(ixs) > 0:
            regression_loss = self.sl1(_deltas, deltas)
            theta_loss = self.thetaloss(pred_theta, theta)
            return    detection_loss + self.lmb * regression_loss+theta_loss, detection_loss.detach(), regression_loss.detach(),theta_loss.detach()
        else:
            regression_loss = 0
            theta_loss = 0
            return  detection_loss + self.lmb * regression_loss+theta_loss, detection_loss.detach(), regression_loss,theta_loss
        
        #self.lmb *theta_loss+