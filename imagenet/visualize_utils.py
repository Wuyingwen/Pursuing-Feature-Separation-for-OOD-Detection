from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from scipy.spatial.distance import pdist, cdist, squareform
import sklearn.covariance
        
### get forward feature by Hook
# -------------------- 第一步：定义接收feature的函数 ---------------------- #
# 这里定义了一个类，类有一个接收feature的函数hook_fun。定义类是为了方便提取多个中间层。
class HookTool: 
    def __init__(self):
        self.fea = None 

    def hook_fun(self, module, fea_in, fea_out):
        self.fea = fea_out
# ---------- 第二步：注册hook，告诉模型我将在哪些层提取feature,比如提取'fc'后的feature，即output -------- #
def get_feas_by_hook(model, extract_module=['fc']):
    fea_hooks = []
    for n, m in model.named_modules():
        # print('name:', n)
        # # if isinstance(m, extract_module):
        # print(extract_module)
        # if n == 'avg_pool':
        #     print('True')
        if n in extract_module:
            cur_hook = HookTool()
            m.register_forward_hook(cur_hook.hook_fun)
            fea_hooks.append(cur_hook)
            
    return fea_hooks
