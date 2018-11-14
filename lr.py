'''A wrapper class for optimizer '''
import numpy as np

class ScheduledOptim(object):
    '''A simple wrapper class for learning rate scheduling'''

    def __init__(self, optimizer, d_model, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps

    def step(self):
        "Step by the inner optimizer"
        self.optimizer.step()

    def zero_grad(self):
        "Zero out the gradients by the inner optimizer"
        self.optimizer.zero_grad()

    def update_learning_rate(self, iter):
        ''' Learning rate scheduling per step '''

        new_lr = np.power(self.d_model, -0.5) * np.min([
            np.power(iter, -0.5),
            np.power(self.n_warmup_steps, -1.5) * iter])

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr