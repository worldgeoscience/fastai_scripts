
# coding: utf-8


import logging
from torchtext import vocab, data

from fastai.nlp import *
from fastai.lm_rnn import *


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


# ## Stateful model

# ### Have put this in a separate file

PATH='/mnt/samsung_1tb/Data/fastai/nietzsche/data/'

TRN_PATH = 'trn/'
VAL_PATH = 'val/'
TRN = f'{PATH}{TRN_PATH}'
VAL = f'{PATH}{VAL_PATH}'

#description of how to pre-process text- here using character model, list seps into chars
TEXT = data.Field(lower=True, tokenize=list)
#TEXT.vocab is a vocab.Vocab class which 'Defines a vocabulary object that will be used to numericalize a field'
bs=64; bptt=8; n_fac=42; n_hidden=256

FILES = dict(train=TRN_PATH, validation=VAL_PATH, test=VAL_PATH)
#min_freq redundant as never less than 3 instances of each char
md = LanguageModelData.from_text_files(PATH, TEXT, **FILES, bs=bs, bptt=bptt, min_freq=3)
#len(md.trn_dl)=batches ~ tokens/bs/bptt, as bptt is randomized (5% time cuts in half + std. dev)
logger.debug(f'trn_dl: {len(md.trn_dl)}, nt: {md.nt}, trn_ds: {len(md.trn_ds)}, trn_ds[0].text: {len(md.trn_ds[0].text)}')

# ### RNN
class CharSeqStatefulRnn(nn.Module):
    def __init__(self, vocab_size, n_fac, bs):
        self.vocab_size = vocab_size
        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)
        self.rnn = nn.RNN(n_fac, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        self.init_hidden(bs)

    def forward(self, cs):
        bs = cs[0].size(0)
        # very last minibatch may have less
        #h=height=num activations, width = minibatch size. If not same set to zeros
        if self.h.size(1) != bs: self.init_hidden(bs)
        outp,h = self.rnn(self.e(cs), self.h)
        #store h, repackage so size doesnt blow out - remember state but not history
        self.h = repackage_var(h)
        #pytorch loss functions dont like rank3 tensors (expects rank2) here .view flattens rows=as necc, cols=vocab_size
        #softmax, want last axis to sum over (prob. per letter)
        return F.log_softmax(self.l_out(outp), dim=-1).view(-1, self.vocab_size)
    
    def init_hidden(self, bs): self.h = V(torch.zeros(1, bs, n_hidden))

def stateful_rnn():

    m = CharSeqStatefulRnn(md.nt, n_fac, 512).cuda()
    opt = optim.Adam(m.parameters(), 1e-3)

    vals_1st = fit(m, md, 4, opt, F.nll_loss)
    logger.debug(vals_1st)
    set_lrs(opt, 1e-4)

    vals_2nd = fit(m, md, 4, opt, F.nll_loss)
    logger.debug(f'<<stateful_rnn Adam lr 1e-3: {vals_1st}, lr 1e-4: {vals_2nd}')


# ### RNN loop
# From the pytorch source, not used much in practice
def RNNCell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    #F.linear does matrix product followed by addition. see F.matmul
    return F.tanh(F.linear(input, w_ih, b_ih) + F.linear(hidden, w_hh, b_hh))

class CharSeqStatefulRnn2(nn.Module):
    def __init__(self, vocab_size, n_fac, bs):
        super().__init__()
        self.vocab_size = vocab_size
        self.e = nn.Embedding(vocab_size, n_fac)
        self.rnn = nn.RNNCell(n_fac, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        self.init_hidden(bs)
        
    def forward(self, cs):
        bs = cs[0].size(0)
        if self.h.size(1) != bs: self.init_hidden(bs)
        outp = []
        o = self.h
        for c in cs: 
            o = self.rnn(self.e(c), o)
            outp.append(o)
        outp = self.l_out(torch.stack(outp))
        self.h = repackage_var(o)
        return F.log_softmax(outp, dim=-1).view(-1, self.vocab_size)
    
    def init_hidden(self, bs): self.h = V(torch.zeros(1, bs, n_hidden))

def stateful_rnn_loop():
    m = CharSeqStatefulRnn2(md.nt, n_fac, 512).cuda()
    opt = optim.Adam(m.parameters(), 1e-3)

    vals_1st = fit(m, md, 4, opt, F.nll_loss)
    set_lrs(opt, 1e-4)
    vals_2nd = fit(m, md, 4, opt, F.nll_loss)
    logger.debug(f'<<stateful_rnn_loop Adam lr 1e-3: {vals_1st}, lr 1e-4: {vals_2nd}')


# ### GRU see wildml GRU and colah understanding lstms
class CharSeqStatefulGRU(nn.Module):
    def __init__(self, vocab_size, n_fac, bs):
        super().__init__()
        self.vocab_size = vocab_size
        self.e = nn.Embedding(vocab_size, n_fac)
        self.rnn = nn.GRU(n_fac, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        self.init_hidden(bs)
        
    def forward(self, cs):
        bs = cs[0].size(0)
        if self.h.size(1) != bs: self.init_hidden(bs)
        outp,h = self.rnn(self.e(cs), self.h)
        self.h = repackage_var(h)
        return F.log_softmax(self.l_out(outp), dim=-1).view(-1, self.vocab_size)
    
    def init_hidden(self, bs): self.h = V(torch.zeros(1, bs, n_hidden))

# From the pytorch source code - for reference
def GRUCell(input, hidden, w_ih, w_hh, b_ih, b_hh):
    gi = F.linear(input, w_ih, b_ih)
    gh = F.linear(hidden, w_hh, b_hh)
    i_r, i_i, i_n = gi.chunk(3, 1)
    h_r, h_i, h_n = gh.chunk(3, 1)

    resetgate = F.sigmoid(i_r + h_r)
    inputgate = F.sigmoid(i_i + h_i)
    newgate = F.tanh(i_n + resetgate * h_n)
    return newgate + inputgate * (hidden - newgate)

def stateful_gru():

    m = CharSeqStatefulGRU(md.nt, n_fac, 512).cuda()

    opt = optim.Adam(m.parameters(), 1e-3)

    vals_1st = fit(m, md, 6, opt, F.nll_loss)
    logger.debug(vals_1st)

    set_lrs(opt, 1e-4)

    vals_2nd = fit(m, md, 3, opt, F.nll_loss)
    logger.debug(f'<<stateful_gru Adam lr 1e-3: {vals_1st}, lr 1e-4: {vals_2nd}')


class CharSeqStatefulLSTM(nn.Module):
    def __init__(self, vocab_size, n_fac, bs, nl):
        super().__init__()
        self.vocab_size, self.nl = vocab_size, nl
        self.e = nn.Embedding(vocab_size, n_fac)
        #added dropout inside the RNN - does dropout after each timestep
        self.rnn = nn.LSTM(n_fac, n_hidden, nl, dropout=0.5)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        self.init_hidden(bs)

    def forward(self, cs):
        bs = cs[0].size(0)
        if self.h[0].size(1) != bs: self.init_hidden(bs)
        outp, h = self.rnn(self.e(cs), self.h)
        self.h = repackage_var(h)
        return F.log_softmax(self.l_out(outp), dim=-1).view(-1, self.vocab_size)

    def init_hidden(self, bs):
        #here we need to return a tuple of matrices
        self.h = (V(torch.zeros(self.nl, bs, n_hidden)),
                  V(torch.zeros(self.nl, bs, n_hidden)))

def stateful_lstm():
    #takes 509
    # ### Putting it all together: LSTM

    from fastai import sgdr
    logger.debug('>>stateful_lstm()')
    #doubled the size - as added 0.5 dropout
    n_hidden=512

    m = CharSeqStatefulLSTM(md.nt, n_fac, 512, 2).cuda()
    #from fastai, typically for diff lr and wd but here as we are using cllbacks need to use
    lo = LayerOptimizer(optim.Adam, m, 1e-2, 1e-5)

    os.makedirs(f'{PATH}models', exist_ok=True)

    vals_1st = fit(m, md, 2, lo.opt, F.nll_loss)
    logger.debug(vals_1st)

    on_end = lambda sched, cycle: save_model(m, f'{PATH}models/cyc_{cycle}')
    #fastai CosAnneal
    cb = [CosAnneal(lo, len(md.trn_dl), cycle_mult=2, on_cycle_end=on_end)]
    vals_2nd = fit(m, md, 2**4-1, lo.opt, F.nll_loss, callbacks=cb)
    logger.debug(vals_2nd)

    on_end = lambda sched, cycle: save_model(m, f'{PATH}models/cyc_{cycle}')
    cb = [CosAnneal(lo, len(md.trn_dl), cycle_mult=2, on_cycle_end=on_end)]
    vals_3rd = fit(m, md, 2**6-1, lo.opt, F.nll_loss, callbacks=cb)
    logger.debug(vals_3rd)

    logger.debug(f'<<stateful_lstm 2 epochs {vals_1st}, cosanealing epochs 2**4-1: {vals_2nd}, cosanealing epochs 2**6-1: {vals_3rd}')


# ### Test


def get_next(inp):
    idxs = TEXT.numericalize(inp)
    p = m(VV(idxs.transpose(0,1)))
    r = torch.multinomial(p[-1].exp(), 1)
    return TEXT.vocab.itos[to_np(r)[0]]



def get_next_n(inp, n):
    res = inp
    for i in range(n):
        c = get_next(inp)
        res += c
        inp = inp[1:]+c
    return res

def stateful_lstm_test():
    logger.debug(get_next('for thos'))
    logger.debug(print(get_next_n('for thos', 400)))

def workflow():
    start = time.time()

    #stateful_rnn()
    #stateful_rnn_loop()
    #stateful_gru()
    stateful_lstm()

    end = time.time()
    elapsed = end - start
    logger.debug(f'>>workflow() took {elapsed}sec')

if __name__ == "__main__":
    workflow()

