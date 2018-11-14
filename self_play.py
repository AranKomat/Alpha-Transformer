from concurrent.futures import ProcessPoolExecutor
import random
import nltk
from time import time, sleep
import _pickle as cPickle

from env import TextEnv
from player import Agent
from api import APIServer

import numpy as np
import h5py
from filelock import FileLock

lock = FileLock("my_lock")


class Utils:
    def __init__(self, config):        self.config = config
        self.translate, _ = cPickle.load(open('save/vocab_cotra.pkl', 'rb'))

    # TODO: do something to protect this writing
    def translate_store(self, iter, sentence):
        with open(self.config.sample_file, "a")as f:
            f.write('iter: %d\n' % iter)
            f.write(
                ' '.join([self.translate[item] if item != self.config.start_token else '' for item in sentence]) + '\n')
            f.write('\n')

    def read_game(self):
        while True:
            try:
                with lock:
                    with h5py.File("buffer", "r") as f:
                        cur_row = f['/cur_row'][0]
                        cur_row_tmp = cur_row % self.config.buffer_size
                        s = f['/s'][cur_row_tmp, :self.config.max_length-1]
                    with h5py.File("buffer", "a") as f:
                        f['/cur_row'][0] = cur_row + 1
                    return s, cur_row
            except KeyboardInterrupt:
                print("Caught KeyboardInterrupt")
                break
            except:
                sleep(0.00001)


    def save_game(self, s, pi, iter, cur_row):
        pi += [[-1] * self.config.simulation_num_per_move]
        pi = np.array(pi)  # [bs, length, sims] for mult, [bs, sims] for single

        #TODO: padding for s is 0, which is problematic for non-progressive mode
        # writes (s,pi,z)'s to the buffer
        # t1 = time()
        while True:
            try:
                with lock:
                    with h5py.File("buffer", "a") as f:
                        s_buffer = f['/s']
                        pi_buffer = f['/pi']
                        # z_buffer = f['/z']
                        cur_row_tmp = cur_row % self.config.buffer_size
                        s_buffer[cur_row_tmp, self.config.max_length-1] = s
                        pi_buffer[cur_row_tmp, self.config.max_length-2:self.config.max_length,...] = pi
                        # z_buffer[cur_row_tmo, ...] = z
                    if (iter + 1) % 100 == 0:
                        with h5py.File("buffer", "r") as f:
                            s = f['/s'][cur_row_tmp, :self.config.max_length]
                            s = [int(elm) for elm in s]
                break
            except KeyboardInterrupt:
                print("Caught KeyboardInterrupt")
                break
            except:
                sleep(0.00001)
        # print("iteration: %s seconds" % (time() - start_time))

        if (iter + 1) % 100 == 0:
            self.translate_store(iter, s)  # print #(self.config.batch_size//2) of generated sentences

def start(config, process_ind):
    api_server = APIServer(config)
    utils = Utils(config)
    process_num = config.multi_process_num
    api_server.start_serve()
    with ProcessPoolExecutor(max_workers=process_num) as executor:
        try:
            futures = []
            for i in range(process_num):
                play_worker = SelfPlayWorker(config, api=api_server.get_api_client(),
                                             utils=utils)
                futures.append(executor.submit(play_worker.start))
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            executor.terminate()
            executor.join()

class SelfPlayWorker:
    def __init__(self, config, api, utils):
        """
        :param config:
        :param TextEnv|None env:
        :param APIServer|None api:
        """
        self.config = config
        self.env = None
        self.agent = None
        self.api = api
        self.false_positive_count_of_resign = 0
        self.resign_test_game_count = 0
        self.utils = utils

    def start(self):
        game_idx = 0
        while True:
            try:
                # play game
                t = time()
                period = False
                s, self.cur_row = self.utils.read_game()
                s_tmp = []
                for i, elm in enumerate(s):
                    if elm != self.config.period_token and elm != self.config.blank_token:
                        s_tmp.append(int(elm))
                    else:
                        period = True
                        break
                if not period:
                    self.ini_state = s_tmp
                    self.start_game()
                    s = self.env.string
                    pi = [row[1] for row in self.agent.moves]
                    v = [row[2] for row in self.agent.moves]
                    length = len(s)
                    self.utils.save_game(s[-1], pi, game_idx, self.cur_row)
                    print(game_idx, " done: %.3f" % (time() - t), length, v)
                    game_idx += 1
            except KeyboardInterrupt:
                print("Caught KeyboardInterrupt")
                break

    def start_game(self):

        # enable_resign = self.config.disable_resignation_rate <= random()
        self.agent = Agent(self.config, api=self.api)
        self.env = TextEnv(self.config)
        state = self.env.state()
        # game loop
        while not self.env.done:
            action = self.agent.action(state)
            if action is None:
                break
            self.env.add(action)
            state = self.env.state()
        #policy_compressed = [-1] * self.config.simulation_num_per_move
        #self.agent.moves.append([self.env.string, policy_compressed])
        #self.agent.finish_game(-1)
        # self.finish_game(resign_enabled=enable_resign)