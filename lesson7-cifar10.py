
# coding: utf-8

# ## CIFAR 10

# You can get the data via:
# 
#     wget http://pjreddie.com/media/files/cifar.tgz
#use data from here, above didn't work for me:
#http://forums.fast.ai/t/training-a-model-from-scratch-cifar-10/7897/11

import logging

from fastai.conv_learner import *

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


PATH = "/mnt/samsung_1tb/Data/fastai/cifar10/"
os.makedirs(PATH,exist_ok=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#normall use transforms from model - normalized
#this time training from scratch, have to tell it mead and std dev (per channel of all images) to normalize
stats = (np.array([ 0.4914 ,  0.48216,  0.44653]), np.array([ 0.24703,  0.24349,  0.26159]))


def get_data(sz,bs):
    tfms = tfms_from_stats(stats, sz, aug_tfms=[RandomFlip()], pad=sz//8)
    return ImageClassifierData.from_paths(PATH, val_name='test', tfms=tfms, bs=bs)


bs=256

def look_at_data():
    # ### Look at data
    data = get_data(32,4)

    x,y=next(iter(data.trn_dl))

    plt.imshow(data.trn_ds.denorm(x)[0])
    plt.show()


    plt.imshow(data.trn_ds.denorm(x)[1])
    plt.show()

# ## Fully connected model

data = get_data(32,bs)


# From [this notebook](https://github.com/KeremTurgutlu/deeplearning/blob/master/Exploring%20Optimizers.ipynb) by our student Kerem Turgutlu:
class SimpleNet(nn.Module):
    #general purpose fully connected nn
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        
    def forward(self, x):
        #flatted
        x = x.view(x.size(0), -1)
        for l in self.layers:
            l_x = l(x)
            x = F.relu(l_x)
        return F.log_softmax(l_x, dim=-1)

def simple_fc_net():

    lr = 1e-2

    #1 level higher than .fit, creates a initialized Learner class
    learn = ConvLearner.from_model_data(SimpleNet([32*32*3, 40,10]), data)

    learn, [o.numel() for o in learn.model.parameters()]

    logger.debug(f'learn.summary: {learn.summary()}')

    learn.lr_find()

    learn.sched.plot()

    start = time.time()
    vals_1st = learn.fit(lr, 2)
    end = time.time()
    elapsed = end - start
    logger.debug(f'vals_1st: {vals_1st}, fit took {elapsed}sec')

    start = time.time()
    #fit(lrs, n_cycle, wds=None, **kwargs) ie passing in the cycle_len kwarg
    #then in get_fit() if cycle_len is defined, it adds a 'Cosine Annealing'
    #   scheduler for varying the learning rate across iterations.
    vals_2nd = learn.fit(lr, 2, cycle_len=1)
    end = time.time()
    elapsed = end - start
    logger.debug(f'vals_2nd: {vals_2nd}, fit took {elapsed}sec')


# ## CNN
#first step to improve on fc model is to replace with a conv model
#in model above every pixel has a weight (not efficient)
#with a conv model we have weights per 3x3 kernel that match a specific pattern

class ConvNet(nn.Module):
    #fully convolutional network - every layer except last is convolutional
    def __init__(self, layers, c):
        super().__init__()
        self.layers = nn.ModuleList([
            # each time have a layer, make the next layer smaller-eg could use max pooling or stride conv (stride=2)
            nn.Conv2d(layers[i], layers[i + 1], kernel_size=3, stride=2)
            for i in range(len(layers) - 1)])
        #make last layer a 1x1 AdaptiveMaxPool2d -use single largest cell as actv, then is 1x1xn_feature dim tensor
        #note this doesnt have any state
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.out = nn.Linear(layers[-1], c)
        
    def forward(self, x):
        #go through every conv layer, do conv, do relu
        for l in self.layers: x = F.relu(l(x))
        #do adaptive max pool
        x = self.pool(x)
        #get rid of trailing axes (returns matrix of minibatch*n_features)
        x = x.view(x.size(0), -1)
        #feed info final linear layer which returns something of size c (see self.out) which here is 10
        return F.log_softmax(self.out(x), dim=-1)

def cnn():
    #10 classes to predict
    learn = ConvLearner.from_model_data(ConvNet([3, 20, 40, 80], 10), data)

    logger.debug(learn.summary())

    #override final lr to make 100 as was still getting better at default end of 10
    learn.lr_find(end_lr=100)
    learn.sched.plot()
    plt.show()

    start_1 = time.time()
    #fit(lrs, n_cycle)
    vals_1st=learn.fit(1e-1, 2)
    end_1 = time.time()
    elapsed_1 = end_1 - start_1
    logger.debug(f'vals_1st: {vals_1st}, fit took {elapsed_1}sec')

    start = time.time()
    vals_2nd=learn.fit(1e-1, 4, cycle_len=1)
    end = time.time()
    elapsed = end - start
    #starting to flatten out at c. 60%
    logger.debug(f'vals_1st: {vals_1st}, fit took {elapsed_1}sec')
    logger.debug(f'vals_2nd: {vals_2nd}, fit took {elapsed}sec')



# ## Refactored (relu inside for loop in forward in ConvNet 'not ideal')
class ConvLayer(nn.Module):
    def __init__(self, ni, nf):
        super().__init__()
        #pad layer of zeros around edge
        self.conv = nn.Conv2d(ni, nf, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x): return F.relu(self.conv(x))


class ConvNet2(nn.Module):
    def __init__(self, layers, c):
        super().__init__()
        #using Module define above
        self.layers = nn.ModuleList([ConvLayer(layers[i], layers[i + 1])
            for i in range(len(layers) - 1)])
        self.out = nn.Linear(layers[-1], c)
        
    def forward(self, x):
        for l in self.layers: x = l(x)
        #calling as a function-dont need the class as has no state
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return F.log_softmax(self.out(x), dim=-1)

def refactored_cnn():
    logger.debug('>>refactored_cnn()')
    learn = ConvLearner.from_model_data(ConvNet2([3, 20, 40, 80], 10), data)

    logger.debug(learn.summary())

    start_1st = time.time()
    vals_1st = learn.fit(1e-1, 2)
    end_1st = time.time()
    elapsed_1st = end_1st - start_1st
    logger.debug(f'vals_1st: {vals_1st}, fit took {elapsed_1st}sec')

    start_2nd = time.time()
    vals_2nd = learn.fit(1e-1, 2, cycle_len=1)
    end_2nd = time.time()
    elapsed_2nd = end_2nd - start_2nd
    logger.debug(f'vals_1st: {vals_1st}, fit took {elapsed_1st}sec')
    logger.debug(f'vals_2nd: {vals_2nd}, fit took {elapsed_2nd}sec')


# ## BatchNorm
class BnLayer(nn.Module):
    def __init__(self, ni, nf, stride=2, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(ni, nf, kernel_size=kernel_size, stride=stride,
                              bias=False, padding=1)
        self.a = nn.Parameter(torch.zeros(nf,1,1))
        self.m = nn.Parameter(torch.ones(nf,1,1))
        
    def forward(self, x):
        #relu before batchnorm cf preact resnet
        x = F.relu(self.conv(x))
        x_chan = x.transpose(0,1).contiguous().view(x.size(1), -1)
        #calc means and std dev
        if self.training:
            self.means = x_chan.mean(1)[:,None,None]
            self.stds  = x_chan.std (1)[:,None,None]
        #normalize then sgd can can scale by m and shift by a
        return (x-self.means) / self.stds *self.m + self.a

class ConvBnNet(nn.Module):
    def __init__(self, layers, c):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5, stride=1, padding=2)
        self.layers = nn.ModuleList([BnLayer(layers[i], layers[i + 1])
            for i in range(len(layers) - 1)])
        self.out = nn.Linear(layers[-1], c)
        
    def forward(self, x):
        x = self.conv1(x)
        for l in self.layers: x = l(x)
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return F.log_softmax(self.out(x), dim=-1)

def batch_norm():
    #using more layers - ConvNet2 wasnt stable with more layers
    logger.debug('>>batch_norm')
    learn = ConvLearner.from_model_data(ConvBnNet([10, 20, 40, 80, 160], 10), data)

    logger.debug(learn.summary())

    start_1 = time.time()
    vals_1 = learn.fit(3e-2, 2)
    end_1 = time.time()
    elapsed_1 = end_1 - start_1

    start_2 = time.time()
    vals_2 = learn.fit(1e-1, 4, cycle_len=1)
    end_2 = time.time()
    elapsed_2 = end_2 - start_2
    logger.debug(f'vals_1: {vals_1}, fit took {elapsed_1}sec')
    logger.debug(f'vals_2: {vals_2}, fit took {elapsed_2}sec')


# ## Deep BatchNorm
class ConvBnNet2(nn.Module):
    def __init__(self, layers, c):
        super().__init__()
        #bigger kernel, 10 out channels (5x5 filters), padding used is (kernel_size-1)/2
        #Conv2d(in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5, stride=1, padding=2)
        self.layers = nn.ModuleList([BnLayer(layers[i], layers[i+1])
            for i in range(len(layers) - 1)])
        #create a stride 1 layer
        self.layers2 = nn.ModuleList([BnLayer(layers[i+1], layers[i + 1], 1)
            for i in range(len(layers) - 1)])
        self.out = nn.Linear(layers[-1], c)
        
    def forward(self, x):
        x = self.conv1(x)
        #zip the stride 2 and 1 layers together - now twice as deep
        for l,l2 in zip(self.layers, self.layers2):
            x = l(x)
            x = l2(x)
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return F.log_softmax(self.out(x), dim=-1)

def deep_batch_norm():
    logger.debug('>>deep_batch_norm()')
    learn = ConvLearner.from_model_data(ConvBnNet2([10, 20, 40, 80, 160], 10), data)

    start_1 = time.time()
    vals_1 = learn.fit(1e-2, 2)
    end_1 = time.time()
    elapsed_1 = end_1 - start_1

    start_2 = time.time()
    vals_2 = learn.fit(1e-2, 2, cycle_len=1)
    end_2 = time.time()
    elapsed_2 = end_2 - start_2
    logger.debug(f'vals_1st: {vals_1}, fit took {elapsed_1}sec')
    logger.debug(f'vals_2nd: {vals_2}, fit took {elapsed_2}sec')
    #12 layers deep - too deep even for batch norm, no improvement over prev 'shallow' batch norm


# ## Resnet
class ResnetLayer(BnLayer):
    #y=x + f(x); f(x) = y - x where y-x is the residual
    #find weights to add on the ammount we were off by: boosting
    def forward(self, x): return x + super().forward(x)

class Resnet(nn.Module):
    def __init__(self, layers, c):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5, stride=1, padding=2)
        self.layers = nn.ModuleList([BnLayer(layers[i], layers[i+1])
            for i in range(len(layers) - 1)])
        self.layers2 = nn.ModuleList([ResnetLayer(layers[i+1], layers[i + 1], 1)
            for i in range(len(layers) - 1)])
        self.layers3 = nn.ModuleList([ResnetLayer(layers[i+1], layers[i + 1], 1)
            for i in range(len(layers) - 1)])
        self.out = nn.Linear(layers[-1], c)
        
    def forward(self, x):
        x = self.conv1(x)
        for l,l2,l3 in zip(self.layers, self.layers2, self.layers3):
            x = l3(l2(l(x)))
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return F.log_softmax(self.out(x), dim=-1)

def resnet():
    logger.debug('>>resnet()')
    learn = ConvLearner.from_model_data(Resnet([10, 20, 40, 80, 160], 10), data)

    wd=1e-5

    start_1 = time.time()
    vals_1 = learn.fit(1e-2, 2, wds=wd)
    end_1 = time.time()
    elapsed_1 = end_1 - start_1

    start_2 = time.time()
    learn.fit(1e-2, 3, cycle_len=1, cycle_mult=2, wds=wd)
    vals_2 = end_2 = time.time()
    elapsed_2 = end_2 - start_2

    start_3 = time.time()
    vals_3 = learn.fit(1e-2, 8, cycle_len=4, wds=wd)
    end_3 = time.time()
    elapsed_3 = end_3 - start_3
    logger.debug(f'vals_1: {vals_1}, fit took {elapsed_1}sec')
    logger.debug(f'vals_2: {vals_2}, fit took {elapsed_2}sec')
    logger.debug(f'vals_3: {vals_3}, fit took {elapsed_3}sec')


# ## Resnet 2
class Resnet2(nn.Module):
    #good starting point for a modern architecture
    #state of the art in 2012
    def __init__(self, layers, c, p=0.5):
        super().__init__()
        self.conv1 = BnLayer(3, 16, stride=1, kernel_size=7)
        self.layers = nn.ModuleList([BnLayer(layers[i], layers[i+1])
            for i in range(len(layers) - 1)])
        self.layers2 = nn.ModuleList([ResnetLayer(layers[i+1], layers[i + 1], 1)
            for i in range(len(layers) - 1)])
        self.layers3 = nn.ModuleList([ResnetLayer(layers[i+1], layers[i + 1], 1)
            for i in range(len(layers) - 1)])
        self.out = nn.Linear(layers[-1], c)
        self.drop = nn.Dropout(p)
        
    def forward(self, x):
        x = self.conv1(x)
        for l,l2,l3 in zip(self.layers, self.layers2, self.layers3):
            x = l3(l2(l(x)))
        x = F.adaptive_max_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        return F.log_softmax(self.out(x), dim=-1)

def resnet2():

    logger.debug('>>resnet2()')

    learn = ConvLearner.from_model_data(Resnet2([16, 32, 64, 128, 256], 10, 0.2), data)

    wd=1e-6

    start_1 = time.time()
    vals_1 = learn.fit(1e-2, 2, wds=wd)
    end_1 = time.time()
    elapsed_1 = end_1 - start_1

    start_2 = time.time()
    vals_2 = learn.fit(1e-2, 3, cycle_len=1, cycle_mult=2, wds=wd)
    end_2 = time.time()
    elapsed_2 = end_2 - start_2

    start_3 = time.time()
    vals_3 = learn.fit(1e-2, 8, cycle_len=4, wds=wd)
    end_3 = time.time()
    elapsed_3 = end_3 - start_3
    logger.debug(f'vals_1: {vals_1}, fit took {elapsed_1}sec')
    logger.debug(f'vals_2: {vals_2}, fit took {elapsed_2}sec')
    logger.debug(f'vals_3: {vals_3}, fit took {elapsed_3}sec')

    learn.save('tmp3')

    log_preds,y = learn.TTA()
    preds = np.mean(np.exp(log_preds),0)

    logger.debug(metrics.log_loss(y,preds), accuracy(preds,y))


def workflow():
    start = time.time()
    #look_at_data()
    #simple_fc_net()
    #cnn() #0.60791
    #refactored_cnn() #0.577832
    #batch_norm() #0.5986328125 and 0.72607421875 - higher than when run in notebook 0.58515625 and 0.708984375 why?
    #deep_batch_norm()
    resnet()
    #resnet2()

    end = time.time()
    elapsed = end - start
    logger.debug(f'>>workflow() took {elapsed}sec')

if __name__ == "__main__":
    workflow()

