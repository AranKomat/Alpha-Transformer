from _asyncio import Future
from asyncio.queues import Queue
from collections import defaultdict, namedtuple
import asyncio
import numpy as np
from collections import defaultdict
from config import Config
from env import TextEnv
from random import random

QueueItem = namedtuple("QueueItem", "state future")

class Agent:
    def __init__(self, config: Config, enable_resign=True, api=None, process_id=0):
        """
        :param config: Config
        :param api: APIServer
        """
        self.config = config
        self.enable_resign = enable_resign
        self.api = api
        # key=(own, enemy, action)
        self.var_n = defaultdict(lambda: np.zeros((self.config.vocab_size,)))
        self.var_w = defaultdict(lambda: np.zeros((self.config.vocab_size,)))
        self.var_p = defaultdict(lambda: np.zeros((self.config.vocab_size,)))
        self.var_v = {}
        self.cache = []

        self.expanded = set()
        self.now_expanding = set()
        self.prediction_queue = Queue(self.config.prediction_queue_size)
        self.sem = asyncio.Semaphore(self.config.parallel_search_num)

        self.moves = []
        self.loop = asyncio.get_event_loop()
        self.running_simulation_num = 0
        #self.net_num_moves = 0
        self.process_id = process_id

    def var_q(self, key):
        return self.var_w[key] / (self.var_n[key] + 1e-5)

    def action(self, root_state):
        self.stat_update(root_state)
        self.search_moves(root_state)
        policy, policy_compressed = self.calc_policy(root_state)
        action = int(np.random.choice(self.config.vocab_size, p=policy))
        if len(root_state) >= self.config.min_resign_turn:
            if self.config.resign_threshold is not None and self.enable_resign and\
                        np.max(self.var_q(root_state) - (self.var_n[root_state] == 0)*10) <= self.config.resign_threshold:
                return None  # means resign

        self.moves.append([list(root_state), policy_compressed, max(list(self.var_v.values()))])
        return action

    def stat_update(self, cur_state):
        var_v = {}
        var_p = defaultdict(lambda: np.zeros((self.config.vocab_size,)))
        cache_tmp = []
        for state in self.var_v:
            if state[:len(cur_state)] == cur_state:
                var_p[state] = self.var_p[state]
                var_v[state] = self.var_v[state]
            if state[:len(cur_state)-1] == cur_state[:-1]:
                cache_tmp.append(state)
        self.var_n = defaultdict(lambda: np.zeros((self.config.vocab_size,)))
        self.var_w = defaultdict(lambda: np.zeros((self.config.vocab_size,)))
        self.var_p = var_p
        self.cache = cache_tmp
        self.var_v = var_v
        self.expanded = set()

    def search_moves(self, root_state):
        loop = self.loop
        self.running_simulation_num = 0

        coroutine_list = []
        for it in range(self.config.simulation_num_per_move+1):
            cor = self.start_search_my_move(root_state)
            coroutine_list.append(cor)

        coroutine_list.append(self.prediction_worker())
        loop.run_until_complete(asyncio.gather(*coroutine_list))

    async def start_search_my_move(self, root_state):
        self.running_simulation_num += 1
        with await self.sem:  # reduce parallel search number
            env = TextEnv(self.config).update(root_state)
            leaf_v = await self.search_my_move(env, is_root_node=True)
            self.running_simulation_num -= 1
            return leaf_v

    async def search_my_move(self, env: TextEnv, is_root_node=False):

        state = env.state()
        if self.config.switch:
            while state in self.now_expanding:
                await asyncio.sleep(self.config.wait_for_expanding_sleep_sec)

            # is leaf?
            if state not in self.expanded:  # reach leaf node
                if state in self.var_v:
                    self.expanded.add(state)
                    leaf_v = self.var_v[state]
                else:
                    leaf_v = await self.expand_and_evaluate(state)
                return leaf_v
            elif env.done:
                return self.var_v[state]

            action_t = self.select_action_q_and_u(state, is_root_node)
            virtual_loss = self.config.virtual_loss
            virtual_loss_for_w = virtual_loss
            env.add(action_t)
            self.var_n[state][action_t] += virtual_loss
            self.var_w[state][action_t] -= virtual_loss_for_w
            leaf_v = await self.search_my_move(env)  # next move

            # on returning search path
            # update: N, W
            self.var_n[state][action_t] += - virtual_loss + 1
            self.var_w[state][action_t] += virtual_loss_for_w + leaf_v
            return self.var_v[state]
        else:
            action_t = self.select_action_q_and_u(state, is_root_node)
            self.var_n[state][action_t] += 1
            self.var_w[state][action_t] += self.var_v[state+(action_t,)]
            return self.var_v[state+(action_t,)]

    async def expand_and_evaluate(self, state):
        self.now_expanding.add(state)
        future = await self.predict(state)  # type: Future
        await future
        self.var_p[state], self.var_v[state] = future.result()
        self.cache.append(state)
        self.expanded.add(state)
        self.now_expanding.remove(state)
        return self.var_v[state]

    async def prediction_worker(self):
        """For better performance, queueing prediction requests and predict together in this worker.
        """
        q = self.prediction_queue
        margin = 10  # avoid finishing before other searches starting.
        while self.running_simulation_num > 0 or margin > 0:
            if q.empty():
                if margin > 0:
                    margin -= 1
                await asyncio.sleep(self.config.prediction_worker_sleep_sec)
                continue
            item_list = [q.get_nowait() for _ in range(q.qsize())]  # type: list[QueueItem]
            data = []
            for x in item_list:
                x = x.state
                c = None if len(x) == 1 else x[:-1]
                data.append((x[-1], c))
            policy_ary, value_ary = self.api.predict(list(zip(*data))+[self.cache, self.process_id])
            for p, v, item in zip(policy_ary, value_ary, item_list):
                item.future.set_result((p, float(v)))

    async def predict(self, x):
        future = self.loop.create_future()
        item = QueueItem(x, future)
        await self.prediction_queue.put(item)
        return future

    def finish_game(self, z):
        """
        :param z: win=1, lose=-1
        :return:stat_update
        """
        for move in self.moves:  # add this game winner result to all past moves.
            move += [z]

    def calc_policy(self, state):
        '''
        :return: policy: the probablity distribution to be sampled from
        :return: policy_compressed: the numpy array (len=simulation_num_per_move) of
                 visited actions with multiplicity; its np.bincount maps to the 'policy' variable
        '''
        # we take #(simulation_num_per_move) of visits of self.var_n[state] and remove the rest
        remainder = np.sum(self.var_n[state]) - self.config.simulation_num_per_move
        assert remainder >= 0
        if remainder > 0: # seldom occurs
            var_n_state = np.copy(self.var_n[state])
            for i in range(int(remainder)):
                prob_var_n = var_n_state/np.sum(var_n_state)
                r = int(np.random.choice(self.config.vocab_size, p=prob_var_n))
                var_n_state[r] -= 1
        else:
            var_n_state = self.var_n[state]

        policy = var_n_state / self.config.simulation_num_per_move
        policy_compressed = np.repeat(np.arange(var_n_state.size), var_n_state.astype(int))
        return policy, policy_compressed

    def select_action_q_and_u(self, state, is_root_node):
        # noinspection PyUnresolvedReferences
        xx_ = np.sqrt(np.sum(self.var_n[state]))  # SQRT of sum(N(s, b); for all b)
        xx_ = max(xx_, 1)  # avoid u_=0 if N is all 0
        p_ = self.var_p[state]

        if is_root_node and self.config.noise_eps > 0:  # Is it correct?? -> (1-e)p + e*Dir(alpha)
            noise = self.config.dirichlet.draw(1,self.config.vocab_size)
            p_ = (1 - self.config.noise_eps) * p_ + self.config.noise_eps * noise

        u_ = self.config.c_puct * p_ * xx_ / (1 + self.var_n[state])
        v_ = self.var_q(state) + u_
        # noinspection PyTypeChecker
        action_t = int(np.argmax(v_))
        return action_t