
# coding: utf-8


import logging
from fastai.io import *
from fastai.conv_learner import *

from fastai.column_data import *

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# ## Setup

# We're going to download the collected works of Nietzsche to use as our data for this class.


PATH='/mnt/samsung_1tb/Data/fastai/nietzsche/'


#get_data("https://s3.amazonaws.com/text-datasets/nietzsche.txt", f'{PATH}nietzsche.txt')
text = open(f'{PATH}nietzsche.txt').read()
print('corpus length:', len(text))


text[:400]


chars = sorted(list(set(text)))
vocab_size = len(chars)+1
print('total chars:', vocab_size)


# Sometimes it's useful to have a zero value in the dataset, e.g. for padding

# In[6]:


chars.insert(0, "\0")

''.join(chars[1:-6])


# Map from chars to indices and back again

char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


# *idx* will be the data we use from now own - it simply converts all the characters to their index (based on the mapping above)

idx = [char_indices[c] for c in text]

idx[:10]


''.join(indices_char[i] for i in idx[:70])


# ## Three char model

# ### Create inputs

# Create a list of every 4th character, starting at the 0th, 1st, 2nd, then 3rd characters


cs=3
c1_dat = [idx[i]   for i in range(0, len(idx)-cs, cs)]
c2_dat = [idx[i+1] for i in range(0, len(idx)-cs, cs)]
c3_dat = [idx[i+2] for i in range(0, len(idx)-cs, cs)]
c4_dat = [idx[i+3] for i in range(0, len(idx)-cs, cs)]


# Our inputs


x1 = np.stack(c1_dat)
x2 = np.stack(c2_dat)
x3 = np.stack(c3_dat)


# Our output


y = np.stack(c4_dat)


# The first 4 inputs and outputs


x1[:4], x2[:4], x3[:4]



y[:4]



x1.shape, y.shape


# ### Create and train model

# Pick a size for our hidden state
n_hidden = 256


# The number of latent factors to create (i.e. the size of the embedding matrix)
n_fac = 42

class Char3Model(nn.Module):
    def __init__(self, vocab_size, n_fac):
        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)

        # The 'green arrow' from our diagram - the layer operation from input to hidden
        #Linear(in_features, out_features)
        self.l_in = nn.Linear(n_fac, n_hidden)

        # The 'orange arrow' from our diagram - the layer operation from hidden to hidden
        #square weight matrix
        self.l_hidden = nn.Linear(n_hidden, n_hidden)
        
        # The 'blue arrow' from our diagram - the layer operation from hidden to output
        self.l_out = nn.Linear(n_hidden, vocab_size)
        
    def forward(self, c1, c2, c3):
        #self.e is size n_fac=42, self.l_in is size n_hidden=256
        in1 = F.relu(self.l_in(self.e(c1)))
        in2 = F.relu(self.l_in(self.e(c2)))
        in3 = F.relu(self.l_in(self.e(c3)))

        #in1 is size n_hidden=256, l_hidden is size n_hidden=256
        h = V(torch.zeros(in1.size()).cuda())
        h = F.tanh(self.l_hidden(h+in1))
        h = F.tanh(self.l_hidden(h+in2))
        h = F.tanh(self.l_hidden(h+in3))
        
        return F.log_softmax(self.l_out(h))


#ColumnarModelData.from_arrays(path, val_idxs, xs, y, bs=64, test_xs=None, shuffle=True)
#small dataset so can use larger batch size
md = ColumnarModelData.from_arrays('.', [-1], np.stack([x1,x2,x3], axis=1), y, bs=512)

#standard pytorch model - so need to write .cuda()
m = Char3Model(vocab_size, n_fac).cuda()

it = iter(md.trn_dl)
*xs,yt = next(it)
t = m(*V(xs))
logger.debug(xs)
logger.debug(t)

opt = optim.Adam(m.parameters(), 1e-2)

#manual fit
fit(m, md, 1, opt, F.nll_loss)

set_lrs(opt, 0.001)

fit(m, md, 1, opt, F.nll_loss)


# ### Test model
def get_next(inp):
    idxs = T(np.array([char_indices[c] for c in inp]))
    #why VV and not V?
    p = m(*VV(idxs))
    i = np.argmax(to_np(p))
    return chars[i]

#pass in 3 characters
logger.debug(get_next('y. '))
logger.debug(get_next('ppl'))
logger.debug(get_next(' th'))
logger.debug(get_next('and'))


# ## Our first RNN!

# ### Create inputs

# This is the size of our unrolled RNN.

cs=8

# For each of 0 through 7, create a list of every 8th character with that starting point. These will be the 8 inputs to out model.
c_in_dat = [[idx[i+j] for i in range(cs)] for j in range(len(idx)-cs)]


# Then create a list of the next character in each of these series. This will be the labels for our model.
c_out_dat = [idx[j+cs] for j in range(len(idx)-cs)]
xs = np.stack(c_in_dat, axis=0)

logger.debug(xs.shape)
y = np.stack(c_out_dat)

# So each column below is one series of 8 characters from the text.
logger.debug(xs[:cs,:cs])


# ...and this is the next character after each sequence.
logger.debug(y[:cs])


# ### Create and train model
val_idx = get_cv_idxs(len(idx)-cs-1)

md = ColumnarModelData.from_arrays('.', val_idx, xs, y, bs=512)

class CharLoopModel(nn.Module):
    # This is an RNN!
    def __init__(self, vocab_size, n_fac):
        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)
        self.l_in = nn.Linear(n_fac, n_hidden)
        self.l_hidden = nn.Linear(n_hidden, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        
    def forward(self, *cs):
        # same as Char3Model Model but here using a loop
        bs = cs[0].size(0)
        h = V(torch.zeros(bs, n_hidden).cuda())
        for c in cs:
            inp = F.relu(self.l_in(self.e(c)))
            #adding input and hidden - may be loosing information
            h = F.tanh(self.l_hidden(h+inp))
        
        return F.log_softmax(self.l_out(h), dim=-1)

m = CharLoopModel(vocab_size, n_fac).cuda()
opt = optim.Adam(m.parameters(), 1e-2)

fit(m, md, 1, opt, F.nll_loss)

set_lrs(opt, 0.001)

fit(m, md, 1, opt, F.nll_loss)

class CharLoopConcatModel(nn.Module):
    def __init__(self, vocab_size, n_fac):
        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)
        self.l_in = nn.Linear(n_fac+n_hidden, n_hidden)
        self.l_hidden = nn.Linear(n_hidden, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        
    def forward(self, *cs):
        bs = cs[0].size(0)
        h = V(torch.zeros(bs, n_hidden).cuda())
        for c in cs:
            #concat rather than add - size is n_fac+n_hidden
            inp = torch.cat((h, self.e(c)), 1)
            inp = F.relu(self.l_in(inp))
            h = F.tanh(self.l_hidden(inp))
        
        return F.log_softmax(self.l_out(h), dim=-1)

m = CharLoopConcatModel(vocab_size, n_fac).cuda()
opt = optim.Adam(m.parameters(), 1e-3)

it = iter(md.trn_dl)
*xs,yt = next(it)
t = m(*V(xs))

fit(m, md, 1, opt, F.nll_loss)

set_lrs(opt, 1e-4)

fit(m, md, 1, opt, F.nll_loss)


# ### Test model

def get_next(inp):
    idxs = T(np.array([char_indices[c] for c in inp]))
    p = m(*VV(idxs))
    i = np.argmax(to_np(p))
    return chars[i]

logger.debug(get_next('for thos'))
logger.debug(get_next('part of '))
logger.debug(get_next('queens a'))


# ## RNN with pytorch

class CharRnn(nn.Module):
    def __init__(self, vocab_size, n_fac):
        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)
        self.rnn = nn.RNN(n_fac, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        
    def forward(self, *cs):
        bs = cs[0].size(0)
        #like our code above pytorch needs a starting state
        #3rd rank tensor with unit axis
        h = V(torch.zeros(1, bs, n_hidden))
        inp = self.e(torch.stack(cs))
        outp,h = self.rnn(inp, h)
        #pytorch gives a list of all hidden states, just want last so using outp[-1]
        return F.log_softmax(self.l_out(outp[-1]), dim=-1)

m = CharRnn(vocab_size, n_fac).cuda()
opt = optim.Adam(m.parameters(), 1e-3)


it = iter(md.trn_dl)
*xs,yt = next(it)

t = m.e(V(torch.stack(xs)))
t.size()

ht = V(torch.zeros(1, 512,n_hidden))
outp, hn = m.rnn(t, ht)
outp.size(), hn.size()

t = m(*V(xs)); t.size()

fit(m, md, 4, opt, F.nll_loss)

set_lrs(opt, 1e-4)

fit(m, md, 2, opt, F.nll_loss)


# ### Test model

def get_next(inp):
    idxs = T(np.array([char_indices[c] for c in inp]))
    p = m(*VV(idxs))
    i = np.argmax(to_np(p))
    return chars[i]

get_next('for thos')

def get_next_n(inp, n):
    res = inp
    for i in range(n):
        c = get_next(inp)
        res += c
        inp = inp[1:]+c
    return res


get_next_n('for thos', 40)


# ## Multi-output model

# ### Setup

# Let's take non-overlapping sets of characters this time
#should have similar accuracy but more efficient

c_in_dat = [[idx[i+j] for i in range(cs)] for j in range(0, len(idx)-cs-1, cs)]


# Then create the exact same thing, offset by 1, as our labels
c_out_dat = [[idx[i+j] for i in range(cs)] for j in range(1, len(idx)-cs, cs)]


xs = np.stack(c_in_dat)
xs.shape

ys = np.stack(c_out_dat)
ys.shape

xs[:cs,:cs]


ys[:cs,:cs]


# ### Create and train model

val_idx = get_cv_idxs(len(xs)-cs-1)


md = ColumnarModelData.from_arrays('.', val_idx, xs, ys, bs=512)

class CharSeqRnn(nn.Module):
    def __init__(self, vocab_size, n_fac):
        super().__init__()
        self.e = nn.Embedding(vocab_size, n_fac)
        self.rnn = nn.RNN(n_fac, n_hidden)
        self.l_out = nn.Linear(n_hidden, vocab_size)
        
    def forward(self, *cs):
        bs = cs[0].size(0)
        h = V(torch.zeros(1, bs, n_hidden))
        inp = self.e(torch.stack(cs))
        outp,h = self.rnn(inp, h)
        return F.log_softmax(self.l_out(outp), dim=-1)

m = CharSeqRnn(vocab_size, n_fac).cuda()
opt = optim.Adam(m.parameters(), 1e-3)

it = iter(md.trn_dl)
*xst,yt = next(it)

def nll_loss_seq(inp, targ):
    #pytorch defaults, ist axis is sequence length ie time steps, 2nd axis is batch size, , 3rd axis is hidden state itself
    #ie 8 * 512 * 256
    sl,bs,nh = inp.size()
    #flatten targets, yt is [512, 8] so to make match need to transpose 1st two axes
    #.vew is same as numpy .reshape, here turning into single vector, -1 means as long as needed
    targ = targ.transpose(0,1).contiguous().view(-1)
    #flatten inputs
    return F.nll_loss(inp.view(-1,nh), targ)


#fastai, implements training
fit(m, md, 4, opt, nll_loss_seq)

set_lrs(opt, 1e-4)

fit(m, md, 1, opt, nll_loss_seq)


# ### Identity init!

m = CharSeqRnn(vocab_size, n_fac).cuda()
opt = optim.Adam(m.parameters(), 1e-2)

#look up code for m.rnn
#see Le, Jaitly, Hinton paper
m.rnn.weight_hh_l0.data.copy_(torch.eye(n_hidden))

fit(m, md, 4, opt, nll_loss_seq)

set_lrs(opt, 1e-3)

fit(m, md, 4, opt, nll_loss_seq)


# ## Stateful model - see lesson6-rnn_stateful.py

def workflow():
    start = time.time()

    end = time.time()
    elapsed = end - start
    logger.debug(f'>>workflow() took {elapsed}sec')

if __name__ == "__main__":
    workflow()
