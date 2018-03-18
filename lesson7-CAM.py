
# coding: utf-8

# ## Dogs v Cats
# 2:02 in lesson 7
import logging



from fastai.imports import *

from fastai.transforms import *
from fastai.conv_learner import *
from fastai.model import *
from fastai.dataset import *
from fastai.sgdr import *


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


PATH = "/mnt/samsung_1tb/Data/fastai/dogs_cats/data/dogscats/"
sz = 224
#
arch = resnet34
bs = 64

tfms = tfms_from_model(arch, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_paths(PATH, tfms=tfms, bs=bs)

m = arch(True)
logger.debug(m)

#delete last 2 layers
m = nn.Sequential(*children(m)[:-2],
                  #conv with 2 outputs, 2 filters of 7x7 will prod 2 numbers
                  nn.Conv2d(512, 2, 3, padding=1),
                  #avg pooling
                  nn.AdaptiveAvgPool2d(1), Flatten(),
                  nn.LogSoftmax())
#note no fc layer at end

learn = ConvLearner.from_model_data(m, data)
#feeze to 4th last layer
learn.freeze_to(-4)

def default_sequential():

    logger.debug(f'm[-1]: {m[-1].trainable}')

    logger.debug(f'm[-4]: {m[-4].trainable}')

    vals_1st = learn.fit(0.01, 1)

    vals_2nd = learn.fit(0.01, 1, cycle_len=1)
    logger.debug(f'{vals_1st}, {vals_2nd}')


# ## CAM class activation maps
class SaveFeatures():
    features=None
    #fwd hook is like a callback
    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)
    def hook_fn(self, module, input, output): self.features = to_np(output)
    def remove(self): self.hook.remove()

def cam():

    x,y = next(iter(data.val_dl))
    x,y = x[None,1], y[None,1]

    vx = Variable(x.cuda(), requires_grad=True)

    dx = data.val_ds.denorm(x)[0]
    plt.imshow(dx)
    plt.show()

    sf = SaveFeatures(m[-4])
    #value of predictions
    py = m(Variable(x.cuda()))
    sf.remove()

    py = np.exp(to_np(py)[0])
    logger.debug(f'py: {py}')

    #vals in final conv layer - 2x7x7
    feat = np.maximum(0, sf.features[0])
    logger.debug(f'feat.shape: {feat.shape}')

    #mult the pred by feat -> cattyness
    f2=np.dot(np.rollaxis(feat,0,3), py)
    f2-=f2.min()
    f2/=f2.max()
    logger.debug(f'f2: {f2}')

    plt.imshow(dx)
    plt.imshow(scipy.misc.imresize(f2, dx.shape), alpha=0.5, cmap='hot');
    plt.show()


def model():
    # ## Model
    #TODO debug why getting AssertionError
    #  File "/mnt/963GB/Data/Python/Courses/fastai/fastai/fastai/layer_optimizer.py", line 20, in opt_params
    #assert(len(self.layer_groups) == len(self.lrs))

    learn.unfreeze()
    learn.bn_freeze(True)

    lr=np.array([1e-6,1e-4,1e-2])

    vals_1st = learn.fit(lr, 2, cycle_len=1)
    acc_1st = accuracy(*learn.TTA())
    logger.debug(f'vals_1st: {vals_1st}, acc_1st: {acc_1st}')

    vals_2nd = learn.fit(lr, 2, cycle_len=1)
    acc_2nd = accuracy(*learn.TTA())
    logger.debug(f'vals_2nd: {vals_2nd}, acc_2nd: {acc_2nd}')

def workflow():
    start = time.time()
    #default_sequential()
    #cam()
    model()

    end = time.time()
    elapsed = end - start
    logger.debug(f'>>workflow() took {elapsed}sec')

if __name__ == "__main__":
    workflow()
