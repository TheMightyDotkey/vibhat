"""from __future__ import print_function
import torch
x = torch.rand(5,3)
print(x) """

import torch
from torch.autograd import Variable

Variable(torch.randn(2,2),requires_grad=True)
