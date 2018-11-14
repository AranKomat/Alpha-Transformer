class TextEnv:
    def __init__(self, config):
        self.num_vocab = config.vocab_size
        self.ini_state = [config.start_token]
        self.config = config
        self.string = list(self.ini_state) #copy it
        self.done = False
        self.cur_length = len(self.ini_state)
        self.max_length = self.config.cur_length
    def update(self, state: tuple):
        self.string = list(state)
        self.cur_length = len(self.string)
        return self
    def state(self):
        return tuple(self.string)
    def finished(self):
        self.done = True
    def add(self, token):
        self.string.append(token)
        self.cur_length += 1
        if token == self.config.period_token or token == self.config.blank_token\
                or self.cur_length == self.max_length: #period
            self.finished()
    def legal_moves(self):
        return range(self.num_vocab)

