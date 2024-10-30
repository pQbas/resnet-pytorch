import torch
import torch.nn as nn
from resnet.model import ResNet

def test_model():
   
   model = ResNet(categories=10)

   input  = torch.rand([4, 3, 224, 224])

   output = model(input)

   assert output.shape == torch.Size([4,10]) 

   probs  = nn.Softmax(dim=1)(output)

   assert probs.shape == torch.Size([4,10]) 
