import torch
import torch.nn as nn
from resnet.model import ResNetBlock

def test_model():
   
   model = ResNetBlock(c_in=10, c_out=10, k_size=3)
   input  = torch.rand([4, 10, 224, 224])
   output = model(input)

   assert output.shape == torch.Size([4, 10, 224, 224]) 
