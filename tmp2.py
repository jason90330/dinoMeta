# import torch
# import torch.nn as nn

import copy
# from collections import OrderedDict

# model = nn.Sequential( OrderedDict( [ ('fc0', nn.Linear(3,1)) ] ) )
# #model.fc0.weight = nn.Parameter( torch.randn(3,1) + 3 )
# print(model.fc0.weight)
# w = model.fc0.weight
# for i in range(5):
#     w_new = w - 2*(w)
# print()
# print(w_new.is_leaf)
# #model.fc0.weight = nn.Parameter( w_new )
# setattr(model,'fc0.weight', w_new )
# print(model.fc0.weight.is_leaf)
# print(model.fc0.weight)
# model_copy = copy.deepcopy(model)
import torch
q = torch.nn.Parameter(torch.Tensor(3,3))
print(q)

p = q[0,:]
print(p)

param_copy = copy.deepcopy(p)