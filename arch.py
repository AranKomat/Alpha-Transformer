import torch
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math
import numpy as np
from torch.nn import LayerNorm as LayerNorm
from copy import copy
from time import time


def float_or_half(config, x):
    if config.fp16:
        return x.half()
    else:
        return x.float()


def half_or_none(config, x):
    if config.fp16:
        return x.half()
    else:
        return x


def initialize(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()


def positional_embedding(config, min_timescale=1.0, max_timescale=1e4):
    channels = config.hidden_dim
    # length = config.max_length // config.batch_width
    length = 10000
    assert (channels % 2 == 0)
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (float(num_timescales) - 1.))
    position = torch.arange(0, length).float()
    inv_timescales = torch.arange(0, num_timescales).float()
    position = position.to(config.device)
    inv_timescales = inv_timescales.to(config.device)

    inv_timescales.mul_(-log_timescale_increment).exp_().mul_(min_timescale)
    scaled_time = position.unsqueeze(1).expand(
        length, num_timescales) * inv_timescales.unsqueeze(0).expand(length, num_timescales)
    # scaled time is now length x num_timescales
    # length x channels
    return float_or_half(config, torch.cat([scaled_time.sin(), scaled_time.cos()], 1))


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.td = Transformer(config)
        self.emb = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.linear_v1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_v2 = nn.Linear(self.hidden_dim, 1)
        self.linear_d1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear_d2 = nn.Linear(self.hidden_dim, 1)
        self.ln_v = LayerNorm(self.hidden_dim)
        self.ln_d = LayerNorm(self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)

    def forward(self, inp, c=None, mask=None):
        # inp = (bs, l, voc)
        x = self.emb(inp)
        if self.training:
            outputs, c = self.td(x)
        else:
            outputs, c = self.td(x.squeeze(1), cache=c, decoding=True, mask=None)
        logits = F.softmax(self.decoder(outputs), -1)
        values = F.tanh(self.linear_v2(self.ln_v(F.relu(self.linear_v1(outputs)))))
        return logits, values.squeeze(-1), c



class Transformer(nn.Module):
    def __init__(self, config, weights=None):

        super(Transformer, self).__init__()
        self.config = config
        if self.config.cached:
            self.R = nn.Parameter(
                torch.zeros(1, config.window_size + (config.batch_size // config.batch_width), config.hidden_dim,
                            device=self.config.device))
            self.u = nn.Parameter(
                torch.zeros(1, config.num_heads, 1, config.hidden_dim // config.num_heads, device=self.config.device))
            self.v = nn.Parameter(
                torch.zeros(1, config.num_heads, 1, config.hidden_dim // config.num_heads, device=self.config.device))
            nn.init.xavier_uniform_(self.R)
            nn.init.xavier_uniform_(self.u)
            nn.init.xavier_uniform_(self.v)
            shared_params = [self.R, self.u, self.v]
        else:
            self.signal = positional_embedding(config)  # max_length x hidden_dim
            shared_params = None

        self.blocks = nn.ModuleList([Block(config, layer_id=i, shared_params=shared_params)
                                     for i in range(config.depth)
                                     ])

    def forward(self, x, cache=None, cur_time=0, decoding=False, code=None, encoder_cache=None, skip_beginning=False,
                tgt=None, mask=None):
        cache = [[None, None] for _ in range(self.config.depth)] if cache is None else cache
        # x = x.mul_(self.scale_embedding)
        if not self.config.cached:
            if not decoding:
                # q = self.signal[cur_time:cur_time + self.config.batch_size // self.config.batch_width]
                q = self.signal[:x.size(1)]
                x.add_(q)
            else:
                x.add_(self.signal[cur_time])
        for i, block in enumerate(self.blocks):

            if i == 0:
                x = F.dropout(x, p=self.config.dropout_prob, training=self.training)

            if not self.config.encoder:
                encoder_cache_i = encoder_cache[i] if encoder_cache is not None else None
                x, cache[i] = block(x, cache[i], cur_time=cur_time,
                                    decoding=decoding, code=code, encoder_cache=encoder_cache_i)
            else:
                x, _ = block(x, cache[i], cur_time, decoding, code)
                cache[i] = x
        return x, cache



class MultiHeadAttention(nn.Module):
    def __init__(self, config, layer_id, masked=True, shared_params=None):
        super(MultiHeadAttention, self).__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.masked = masked
        self.linear_q = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.linear_k = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.linear_v = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
        self.linear_out = nn.Linear(config.hidden_dim, config.hidden_dim)
        modes = config.attention_mode.split("/")
        if len(modes) == 1:
            self.attention_mode = config.attention_mode
        else:
            if modes[layer_id] == 'f':
                self.attention_mode = 'full'
            elif modes[layer_id] == 'd':
                self.attention_mode = 'dil'
            elif modes[layer_id] == 'c':
                self.attention_mode = 'c'
        if self.attention_mode == 'full':
            self.attention = SDPAttention(config, masked=masked)
        #elif self.attention_mode == 'dil' and masked:
        #    self.attention = DLAttention(config, layer_id, shared_params)
        elif self.attention_mode == 'c':
            self.attention = SDPAttention(config, masked=masked)
        else:
            NotImplementedError
        assert (config.hidden_dim % config.num_heads == 0)
        if self.attention_mode == 'c':
            self.conv = nn.Conv1d(config.hidden_dim, config.hidden_dim, 3, stride=3, bias=False)
        if not masked:
            self.attention = SDPAttention(config, masked=masked)
            self.attention_mode = 'full'

    def forward(self, q, k, v, cache=None, cur_time=0, decoding=False):
        if self.attention_mode == 'c':
            k_tmp = self.conv(k.transpose(1, 2)).transpose(1, 2)
            k_tmp = torch.cat([k[:, :1, :], k_tmp], 1)

            def construct_bias_vectors(t, axis):
                length_coordinates = float_or_half(self.config,
                                                   torch.range(0, t.size(1) - 1, device=self.config.device))
                length_coordinates = length_coordinates.unsqueeze(axis)
                # [1, length_k] or [length_q, 1]
                return length_coordinates

            bias = torch.ge(construct_bias_vectors(k_tmp, 0) * 3, construct_bias_vectors(k, 1) + 1e-3)
            k = k_tmp
            v = k_tmp
        else:
            bias = None
        q = self.linear_q(q)
        # print(k)
        if not self.config.cached or self.attention_mode == 'full':
            k = self.linear_k(k)
            v = self.linear_v(v)
        if self.attention_mode == 'full':
            if decoding or self.config.retain_cache:
                if cache[0] is not None:
                    k = torch.cat([cache[0], k], 1)
                    v = torch.cat([cache[1], v], 1)
                cache = [k, v]
        batch, length, dim = list(q.size())

        if self.config.cached and self.attention_mode != 'full':
            if not decoding:
                # TODO: cache decoding
                if cache[0] is None:
                    cache = [q.new_zeros(batch, self.config.window_size, dim),
                             q.new_zeros(batch, self.config.window_size, dim)]
                k = torch.cat([cache[0], k], 1)
                v = torch.cat([cache[1], v], 1)
                cache = [k[:, -self.config.window_size:].detach(),
                         v[:, -self.config.window_size:].detach()]
                k = self.linear_k(k)
                v = self.linear_v(v)
            else:
                k = self.linear_k(k)
                v = self.linear_v(v)
                if cache[0] is None:
                    cache = [q.new_zeros(batch, self.config.window_size - 1, dim),
                             q.new_zeros(batch, self.config.window_size - 1, dim)]
                k = torch.cat([cache[0], k], 1)
                v = torch.cat([cache[1], v], 1)
                cache = [k[:, -self.config.window_size:].detach(),
                         v[:, -self.config.window_size:].detach()]

        q = q.view(batch, q.size(1), self.num_heads, -1).transpose(1, 2)
        k = k.view(batch, k.size(1), self.num_heads, -1).transpose(1, 2)
        v = v.view(batch, v.size(1), self.num_heads, -1).transpose(1, 2)
        # t=time()
        out, tmp = self.attention(q, k, v, decoding=decoding, bias=bias, cache=cache)
        # print(time()-t)

        # if self.attention_mode == 'dil': cache = tmp
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch, length, heads * dim) if self.config.enas else out.view(batch, length, dim)
        return self.linear_out(out), cache


class SDPAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """

    def __init__(self, config, mask_scale=None, masked=True):
        super(SDPAttention, self).__init__()
        self.config = config
        self.masked = masked
        self.dropout = nn.Dropout(config.dropout_prob)
        self.masked = masked
        if masked:
            self.mask = None

    def forward(self, q, k, v, decoding=False, bias=None, cache=None):
        b_q, h_q, t_q, dim_q = list(q.size())
        b_k, h_k, t_k, dim_k = list(k.size())
        b_v, h_v, t_v, dim_v = list(v.size())
        assert (b_q == b_k and b_k == b_v)  # batch size should be equal
        assert (dim_q == dim_k)  # dims should be equal
        assert (t_k == t_v)  # times should be equal
        qk = torch.matmul(q, k.transpose(2, 3))  # b x t_q x t_k
        qk *= dim_q ** -0.5
        if not decoding and self.masked:
            if bias is not None:
                self.mask = bias
            elif self.mask is None:
                self.mask = torch.ones(t_q, t_k, device=self.config.device).byte().triu_(1)
            qk += half_or_none(self.config, self.mask.float() * (-1e9))
        sm_qk = F.softmax(qk, dim=-1)
        sm_qk = self.dropout(sm_qk)
        o = torch.matmul(sm_qk, v)
        return o, None


class Block(nn.Module):
    def __init__(self, config, layer_id, shared_params):

        super(Block, self).__init__()
        self.config = config
        hidden_size = config.hidden_dim
        inner_linear = hidden_size * 2
        self.lnorm1 = LayerNorm(hidden_size)
        self.lnorm2 = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(config.dropout_prob)

        self.masked_attention = MultiHeadAttention(config, layer_id=layer_id, masked=not config.encoder,
                                                   shared_params=shared_params)
        if config.attend:
            self.attention = MultiHeadAttention(config, layer_id=layer_id, masked=False)
            self.lnorm3 = LayerNorm(hidden_size)

        if True:
            self.fc = nn.Sequential(nn.Linear(hidden_size, inner_linear),
                                    nn.ReLU(),
                                    nn.Dropout(config.dropout_prob),
                                    nn.Linear(inner_linear, hidden_size))
        else:
            self.co = nn.Conv1d(hidden_size, hidden_size, 3, padding=1, bias=False)
            # self.lno = LayerNorm(hidden_size)

    def forward(self, x, cache=None, cur_time=0, decoding=False, code=None, encoder_cache=None):
        res = x
        x = self.lnorm1(x)
        x, cache = self.masked_attention(x, x, x, cache=cache, cur_time=cur_time, decoding=decoding)
        x = self.dropout(x).add_(res)
        res = x
        if self.config.attend:
            x = self.lnorm3(x)
            x, _ = self.attention(x, encoder_cache, encoder_cache, cache=None, cur_time=cur_time,
                                  decoding=decoding)
            x = self.dropout(x).add_(res)
            res = x
        if self.config.fc_layers:
            x = self.lnorm2(x)
            x = self.fc(x)

            # org_l = x.size(1)
            '''pad = (0, 0, 2, 0, 0, 0)
            x = F.pad(x, pad=pad)
            x = x.transpose(2, 1)
            x = self.co(x)
            x = x.transpose(2, 1)
            x = x[:, 1:-1]'''

            x = self.dropout(x).add_(res)

        return x, cache



def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float('-inf')).type_as(t)


'''class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.depth = config.depth
        self.hidden_dim = config.hidden_dim
        #TODO: instead use sru with 6 layers
        self.linear_v1 = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.linear_v2 = nn.Linear(self.hidden_dim,1)
        self.linear_d1 = nn.Linear(self.hidden_dim,self.hidden_dim)
        self.linear_d2 = nn.Linear(self.hidden_dim,1)
        self.ln_v = LayerNorm(self.hidden_dim)
        self.ln_d = LayerNorm(self.hidden_dim)
        self.encoder = nn.Embedding(self.vocab_size, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, self.vocab_size, bias=False)
        self.decoder.weight = self.encoder.weight
        #self.cell = Stack_Residual_RNNCell(self.depth, self.hidden_dim, config)
        self.cell = SRU(config.hidden_dim, config.hidden_dim, num_layers=6, dropout=config.dropout_prob,
                        layer_norm=True)
    def forward(self, inp, c):
        # inp = (bs, l, voc)
        x = self.encoder(inp)
        if self.training:
            outputs = []
            for y in torch.unbind(x, 1):
                output, c = self.cell(y, c)
                outputs += [output]
            outputs = torch.stack(outputs, 1)
        else:
            outputs, c = self.cell(x.squeeze(1), c)
        logits = F.softmax(self.decoder(outputs),-1)
        values = F.tanh(self.linear_v2(self.ln_v(F.relu(self.linear_v1(outputs)))))
        return logits, values.squeeze(-1), c

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class Stack_Residual_RNNCell(nn.Module):

    def __init__(self, depth, hidden_dim, config):
        super(Stack_Residual_RNNCell, self).__init__()
        self.config = config
        self.cells = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.lns2 = nn.ModuleList()
        for layer in range(depth):
            if layer % self.config.unit_depth == 0:
                self.cells += [nn.LSTMCell(hidden_dim, hidden_dim)]
            else:
                self.cells += [nn.Sequential(nn.Linear(hidden_dim, 4*hidden_dim),
                                             nn.ReLU(),
                                             LayerNorm(4*hidden_dim),
                                             nn.Linear(4*hidden_dim, hidden_dim),
                                             nn.ReLU(),
                                             LayerNorm(hidden_dim))]
            if layer != depth-1:
                self.lns += [LayerNorm(hidden_dim)]

    def forward(self, inputs, state):
        cur_inp = inputs
        past_inp = torch.zeros_like(cur_inp)
        new_states = []
        for i, cell in enumerate(self.cells):
            if i % self.config.unit_depth == 0:
                cur_state = state[i//self.config.unit_depth]
            cur_inp = cur_inp + past_inp
            if i != 0:
                cur_inp = self.lns[i-1](cur_inp)
            past_inp = cur_inp
            if i % self.config.unit_depth == 0:
                new_state = cell(cur_inp, cur_state)
                cur_inp = new_state[0]
            else:
                cur_inp = cell(cur_inp)
            if i % self.config.unit_depth == 0:
                new_states.append(new_state)
        new_states = tuple(new_states)
        return cur_inp, new_states

def initialize(m):
    if isinstance(m, nn.Linear):
        init.xavier_normal(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()'''