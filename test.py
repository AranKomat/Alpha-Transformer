import numpy as np
import glob
import os
from lstm import Model

from multiprocessing import connection
from threading import Thread
from time import time

from config import Config
import torch
from torch.autograd import Variable as V
config = Config()
model = Model(config)
def convert_raw(x, config):
    if config.use_cuda:
        return V(x.cuda(), volatile=True)
    else:
        return V(x, volatile=True)

def convert(x, config):
    return convert_raw(torch.from_numpy(x.astype(np.float32)), config)

action, lengths, cache_tmp, sizes = [], [], [], []
x = np.random.randn(config.depth//config.unit_depth,
                               config.hidden_dim, 2)
t = time()
for i in range(20):
    cache = []
    for i in range(4840):
        cache.append(x)
    cache = np.stack(cache) if len(cache_tmp) > 1 else np.expand_dims(cache[0], 0)
    cache = tuple([tuple([convert(cache[:,i,:,j], config) for j in range(cache.shape[-1])]) for i in range(cache.shape[1])])
    action = [[a] for a in range(4840)]
    action = torch.LongTensor(action)
print(time()-t)
t = time()
for i in range(20):
    _, _, d_ary, _ = model(V(action, volatile=True), cache)
    value_ary = d_ary.data.cpu().numpy()
print(time()-t)



