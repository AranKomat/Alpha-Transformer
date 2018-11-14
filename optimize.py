import numpy as np
import random
from time import time, sleep
import h5py
import torch
import torch.nn as nn
import torch.optim as optimizer
import glob
import os
#from scipy.stats import rankdata
from lstm import Model, initialize
from Optim import ScheduledOptim


# import _pickle as cPickle


# np.set_printoptions(threshold=np.nan)


def start(config):
    model = Model(config)
    model = model.to(config.device)

    #optim = optimizer.SGD(model.parameters(), lr=2e-4, momentum=0.9, weight_decay=config.c)
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=200, gamma=0.1)  # 20M iters
    optim = ScheduledOptim(
        optimizer.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr,
            betas=(0.9, 0.98), eps=1e-09),
        config.hidden_dim, 2000)

    list_of_files = glob.glob(config.model_path + '/*')
    latest_file = None
    if list_of_files:
        latest_file = max(list_of_files, key=os.path.getctime)
    model_ckpt = latest_file
    # model_ckpt = config.model_path + '/model-454.pth'
    print(model_ckpt)
    if model_ckpt:
        checkpoint = torch.load(model_ckpt)
        model.load_state_dict(checkpoint['state_dict'])
        optim.optimizer.load_state_dict(checkpoint['optimizer'])
        start_iter = model_ckpt.split('-')[-1].split('.')[0]
        start_iter = int(start_iter)
    else:
        model.apply(initialize)
        start_iter = 0

    count = 0

    for iter in range(start_iter, config.total_iterations):
        print('iteration: %s' % iter)
        #if (iter + 1) % 100000 == 0:
        #    lr_scheduler.step()
        start_time = time()
        optim.update_learning_rate(iter)


        # reads the randomly sampled (s,pi,z)'s from the buffer
        # ~ 0.1s
        # TODO: if error, set a lock
        # translate, _ = cPickle.load(open('save/vocab_cotra.pkl', 'rb'))

        with h5py.File("buffer", "r") as f:
            cur_row = int(f['/cur_row'][0])
            s_buffer = f['/s']
            pi_buffer = f['/pi']
            z_buffer = f['/z']
            s_tmp = []
            pi_tmp = []
            z_tmp = []
            df = cur_row - count

            '''x = np.bincount(s_buffer[:,1].astype(int)) / 500000
            for i in range(len(x)):
                if x[i] > 0.01:
                    print(i, x[i], translate[i])
            break'''

            if count == 0:
                count = cur_row
                t_inf = time()
            if count != 0 and df >= 1000:
                print('time required for 32 self-play games: ', 32 * (time() - t_inf) / df)
                t_inf = time()
                count = cur_row
            if cur_row >= config.buffer_size:
                r = np.sort(
                    np.random.choice(list(range(0, config.buffer_size)), (config.batch_size // 2), replace=False))
            else:
                r = np.sort(
                    np.random.choice(list(range(0, cur_row)), (config.batch_size // 2), replace=False))
            tmp = []
            # randomly sample rows 8 times for a dramatic speedup.
            num_segments = 8
            for i in range(num_segments):
                tmp.append(
                    r[(config.batch_size // 2) // num_segments * i:(config.batch_size // 2) // num_segments * (i + 1)])
            for i in range(num_segments):
                s_tmp.append(s_buffer[tmp[i], :config.max_length])
                pi_tmp.append(pi_buffer[tmp[i], :config.max_length, ...])
                z_tmp.append(z_buffer[tmp[i], ...])
        s = np.concatenate(s_tmp, 0)
        pi = np.concatenate(pi_tmp, 0)
        z = np.concatenate(z_tmp, 0)
        # print('io time: ',time() - start_time)

        # decompresses sampled pi's
        # takes about 0.005s
        new_pi = np.zeros(((config.batch_size // 2), config.max_length, config.vocab_size))
        for i in range((config.batch_size // 2)):
            for j in range(config.max_length):
                if pi[i, j, 0] == -1:  # meaning the terminal state; pi=0
                    new_pi[i, j, :] = 0
                elif pi[i, j, 0] == -2 or sum(pi[i, j, :]) == 0:  # meaning the padding; place -1 padding
                    new_pi[i, j, :] = -1
                else:
                    # Beware that np.bincount's bin is [0,1,...min_length-1]
                    new_pi[i, j, :] = np.bincount(pi[i, j, :].astype(int),
                                                  minlength=config.vocab_size) / config.simulation_num_per_move
        pi = new_pi

        # creating a mask for loss function and preparing a minibatch
        def generate_mask(array):
            new_array = np.zeros_like(array)
            for i in range(len(array)):
                for j in range(len(array[i])):
                    if j == len(array[i]) - 1:
                        new_array[i, :] = 1
                    elif array[i, j] == config.period_token:
                        new_array[i, :j + 1] = 1
                        break
                    elif array[i, j] == config.blank_token:
                        new_array[i, :j] = 1
                        break
            return new_array

        def pi_mask(array):
            array = array[:, 1:]
            array = np.pad(array, ((0, 0), (0, 1)), 'constant')
            return generate_mask(array)


        # pi_tmp isn't modified here, since the mask will be modified appropriately

        _, pi_mask = pi_mask(s)
        z_mask = generate_mask(s)
        z_batch = np.concatenate(
            [np.ones([(config.batch_size // 2), config.max_length]) * (-1),
             np.ones([(config.batch_size // 2), config.max_length])])
        def convert(x):
            return torch.tensor(x.astype(np.float32), device=config.device)

        t2 = time()

        # gradient update
        model.train()

        cache = []
        for i in range(config.depth // config.unit_depth):
            cache += [torch.zeros(config.batch_size, config.hidden_dim,device=config.device),
                       torch.zeros(config.batch_size, config.hidden_dim,device=config.device)]
        s_batch = convert(np.array(s)).long()
        policy, v, cache = model(s_batch, tuple(cache))

        def loss_policy(y_true, y_pred):
            return torch.sum(-y_true * torch.log(y_pred + 1.0e-8), 2)

        def loss_value(y_true, y_pred):
            return (y_true - y_pred) ** 2

        pi_mask = convert(pi_mask)
        z_mask = convert(z_mask)
        z = convert(z)
        pi = convert(pi)
        loss = torch.mean(torch.sum(loss_policy(pi, policy) * pi_mask +
                                    loss_value(z, v) * z_mask
                                    , 1) / torch.sum(z_mask, 1))
        loss.backward()
        gn = nn.utils.clip_grad_norm(model.parameters(), config.clip)
        print(gn)
        optim.step()
        optim.zero_grad()
        print("grad update: %s seconds" % (time() - t2))
        print("iteration: %s seconds" % (time() - start_time))

        checkpoint = {'state_dict': model.state_dict(),
                      'optimizer': optim.optimizer.state_dict()}
        sleep(config.training_sleep_time)
        torch.save(checkpoint, config.model_path + '/model' + '-' + str(iter + 1) + '.pth')