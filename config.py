import numpy as np
from scipy import special
import torch
import math

class Config:
    def __init__(self):
        self.vocab_size = 4840
        self.simulation_num_per_move = 80
        self.c_puct = 1
        self.noise_eps = 0.25
        self.dirichlet_alpha = 0.03  # or 0.003
        self.virtual_loss = 3
        #self.min_resign_turn = 3
        self.max_length = 50

        self.training_sleep_time = 0

        self.prediction_queue_size = 16
        self.parallel_search_num = 8
        self.multi_process_num = 16
        #self.resigned_turn = 16
        self.saving_period = 100

        self.model_ckpt = None
        self.use_cuda = torch.cuda.is_available()

        self.prediction_worker_sleep_sec = 0.00001
        self.wait_for_expanding_sleep_sec = 0.000001
        self.resign_threshold = -0.9
        self.disable_resignation_rate = 0.1
        self.false_positive_threshold = 0.05
        self.resign_threshold_delta = 0.01

        self.switch = True

        self.model_path = './ckpts'
        self.dataset_size = 80000
        self.total_iterations = 500000
        self.buffer_size = 500000
        self.seed = 88
        self.start_token = 4839
        self.blank_token = 1814
        self.period_token = 193
        self.positive_file = 'save/realtrain_cotra.txt'
        self.reference_file = 'save/realtest_coco.txt'
        self.sample_file = 'save/arch.txt'
        self.batch_size = 512
        self.pi_batch = 32
        self.arch = 'transformer'
        self.c = 1.0e-4
        # modification of arch hyperparameters requires modification of other files, too
        self.hidden_dim = 256
        self.depth = 6
        self.dropout_prob = 0.1
        self.clip = 5

        # Beware that this process takes ~10 sec, which means you shouldn't call __init__() multiple times.
        self.dirichlet = symm_dirichlet(self.dirichlet_alpha)


        self.num_heads = 8
        self.conv = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ckpt = None

        # half-precision
        self.fp16 = True
        self.cached = False
        self.rl_search = not True

        self.direct = False
        self.soft_em = False
        self.attend = False
        self.encoder = False
        self.use_encoder = True
        self.use_decoder = True
        self.fc_layers = True
        self.retain_cache = True and ((self.arch == 'AE' or self.arch == 'ED') and self.collect_outputs == 'up')
        self.signal_each = True
        self.blank_token = -100


# https://stackoverflow.com/questions/48959739/a-very-quick-method-to-approximate-np-random-dirichlet-with-large-dimension
class symm_dirichlet:
    def __init__(self, alpha, resolution=2 ** 16):
        self.alpha = alpha
        self.resolution = resolution
        self.range, delta = np.linspace(0, 1, resolution,
                                        endpoint=False, retstep=True)
        self.range += delta / 2
        self.table = special.gammaincinv(self.alpha, self.range)

    def draw(self, n_sampl, n_comp, interp='nearest'):
        if interp != 'nearest':
            raise NotImplementedError
        gamma = self.table[np.random.randint(0, self.resolution,
                                             (n_sampl, n_comp))]
        return gamma / gamma.sum(axis=1, keepdims=True)


