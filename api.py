import numpy as np
import glob
import os

from multiprocessing import connection
from threading import Thread
from time import time

from config import Config
import torch
from torch.autograd import Variable as V
import torch.multiprocessing as mp
from torch.multiprocessing import Pipe
import torch.nn.utils.rnn as rnn

mp = mp.get_context('forkserver')

'''try:
    set_start_method('spawn')
except RuntimeError:
    pass'''

class CacheDicts:
    def __init__(self, process_num):
        self.dicts = [{} for _ in range(process_num)]

    def add(self, keys, values, id):
        for key, value in zip(keys[id], values[id]):
            if key not in self.dicts[id]:
                self.dicts[id][key] = value

    def update(self, keys, id):
        dict = self.dicts[id]
        new_dict = {}
        for key in keys[id]:
            new_dict[key] = dict[key]
        self.dicts[id] = new_dict

class APIClient:
    def __init__(self, conn):
        self.connection = conn

    def predict(self, x):
        self.connection.send(x)
        return self.connection.recv()

class APIServer:
    def __init__(self, config: Config):
        """
        :param config:
        """
        self.config = config
        self.connections = []
        self.model = None
        self.latest_file = None
        self.cache_dicts = CacheDicts(config.multi_process_num)

    def get_api_client(self):
        me, you = Pipe()
        self.connections.append(me)
        return APIClient(you)

    def start_serve(self):
        self.load_model()
        self.prediction_worker = Thread(target=self.prediction_worker, name="prediction_worker")
        self.prediction_worker.daemon = True
        self.prediction_worker.start()

    def load_model(self):
        from lstm import Model, initialize
        self.model = Model(self.config)
        if self.config.use_cuda:
            self.model = self.model.cuda()
        list_of_files = glob.glob(self.config.model_path + '/*')
        latest_file = None
        if list_of_files:
            latest_file = max(list_of_files, key=os.path.getctime)
        model_ckpt = latest_file if not list_of_files else None
        print(model_ckpt)
        if model_ckpt is not None:
            self.model.load_state_dict(torch.load(model_ckpt)['state_dict'])
        else:
            self.model.apply(initialize)

    # update model whenever a ckpt file is created
    def update_model(self):

        try:
            list_of_files = glob.glob(self.config.model_path + '/*')
            if list_of_files:
                latest_file = max(list_of_files, key=os.path.getctime)
                if latest_file != self.latest_file:
                    # TODO: if restore interferes with other processes, modify this
                    try:
                        self.model.load_state_dict(torch.load(latest_file)['state_dict'])
                        print(latest_file)
                        print('loaded')
                        self.latest_file = latest_file
                    except:
                        return
                for file in list_of_files:
                    if int(file.split('.')[-2].split('-')[-1]) % self.config.saving_period != 0\
                            and file != latest_file:
                        try:
                            os.remove(file)
                        except OSError:
                            pass
        except:
            pass

    def prediction_worker(self):
        with torch.no_grad():
            while True:
                self.update_model()
                ready_conns = connection.wait(self.connections, timeout=0.00001)
                if not ready_conns:
                    continue
                action, lengths, cache_tmp, cur_cs, sizes, ids, ids2, rootss = [], [], [], [], [], [], [], []
                #TODO: think about the boundary cases
                for conn in ready_conns:
                    a, c, cur_c, id, roots = conn.recv()
                    self.cache_dicts.update(cur_c, id)
                    action += a
                    cache_tmp += c
                    ids += [id]
                    ids2 += [id]*len(a)
                    sizes.append(len(a))
                    rootss += roots
                # cache = (bs, depth, hid_dim)
                original_action = action
                original_cache = cache_tmp
                cache = []
                none_id = []
                for i in range(len(cache_tmp)):
                    if cache_tmp[i] is not None:
                        cache.append(self.cache_dicts.dicts[ids2[i]][cache_tmp[i]])
                        none_id.append(True)
                    else:
                        if self.config.arch == 'Transformer':
                            cache.append(torch.zeros([1, self.config.hidden_dim, self.config.depth, 2], device=self.config.device))
                        elif self.config.arch == 'sru':
                            cache.append(torch.zeros([1, self.config.hidden_dim, self.config.depth], device=self.config.device))
                        none_id.append(False)
                lengths = [len(c) if n else 0 for c, n in zip(cache, none_id)]
                action = [[a] for a in action]
                #TODO: timing signal to be implemented
                action = torch.tensor(action, device=self.config.device)
                if self.config.arch == 'Transformer':
                    #TODO: implement individual cache method
                    # padding for decoding unnecessary
                    # cache is inverted in length dimension
                    #cache = [(i, c) for i, c in enumerate(cache)]
                    #this line doesn't work for non-individual method
                    cache = sorted(enumerate(cache), key=lambda x: len(x[1]), reverse=True)
                    order, cache = list(map(list, zip(*cache)))
                    cache = [c.unbind(-1) for c in cache.unbind(-1)]
                    new_tensor = [[[None]*len(order) for i in range(2)] for j in range(self.config.depth)]
                    for i in range(len(order)):
                        for j in range(self.config.depth):
                            for k in range(2):
                                new_tensor[j][k][i] = cache[i][j][k]
                    cache = [[flip(rnn.pad_sequence(c, batch_first=True), 1) for c in c_depth] for c_depth in cache]
                    bs, max_length, _ = list(cache[0][0].size())
                    mask = action.new_tensor([[1]*(max_length-lengths[i])+[0]*(lengths[i]) for i in range(bs)]).byte()
                elif self.config.arch == 'sru':
                    # cache = [[for depth] for batch]
                    cache = list(map(list, zip(*cache)))
                    cache = [torch.stack(c) for c in cache]
                    mask = None
                policy, value, cache = self.model(action, cache, mask)
                if self.config.arch == 'Transformer':
                    cache = torch.stack([torch.stack(c,-1) for c in cache],-1)
                    cache = [(o, c) for o, c in zip(order, flip(cache, 1))]
                    cache = sorted(enumerate(cache), key=lambda x: x[0])
                    cache = [c[:lengths[i]+1] for i, c in enumerate(torch.unbind(cache,0))]
                elif self.config.arch == 'sru':
                    cache = [c.unbind() for c in cache]
                    cache = list(map(list, zip(*cache)))
                policy = policy.data.cpu().numpy()
                value = value.data.cpu().numpy()
                def elm_add(x,y):
                    new_tuple = []
                    for i, j in zip(x,y):
                        new_tuple += [tuple(list(i)+[j])]
                    return new_tuple
                idx = 0
                for conn, s, id in zip(ready_conns, sizes, ids):
                    conn.send((policy[idx:idx + s], value[idx:idx + s]))
                    self.cache_dicts.add(elm_add(original_cache[idx:idx + s], original_action[idx:idx + s]),
                                         cache[idx:idx + s], id)
                    idx += s

def convert(x, config):
    return torch.tensor(x.astype(np.float32), device=config.device)

def multi_meshgrid(*args):
    """
    Creates a meshgrid from possibly many
    elements (instead of only 2).
    Returns a nd tensor with as many dimensions
    as there are arguments
    """
    args = list(args)
    template = [1 for _ in args]
    for i in range(len(args)):
        n = args[i].shape[0]
        template_copy = template.copy()
        template_copy[i] = n
        args[i] = args[i].view(*template_copy)
        # there will be some broadcast magic going on
    return tuple(args)

def flip(tensor, dims):
    if not isinstance(dims, (tuple, list)):
        dims = [dims]
    indices = [torch.arange(tensor.shape[dim] - 1, -1, -1,
        dtype=torch.int64) for dim in dims]
    multi_indices = multi_meshgrid(*indices)
    final_indices = [slice(i) for i in tensor.shape]
    for i, dim in enumerate(dims):
        final_indices[dim] = multi_indices[i]
    flipped = tensor[final_indices]
    # need to permute the final dimensions
    # if dims is not consecutive, but I'm lazy
    # now :-)
    return flipped