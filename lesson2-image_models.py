
# coding: utf-8
# ## Multi-label classification
import logging
from fastai.conv_learner import *
from fastai.plots import *
from planet import f2, opt_th

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)
PATH = '/mnt/samsung_1tb/Data/fastai/planet/'

def cuda_setup():
    logger.debug(f'cuda available: {torch.cuda.is_available()}, enabled: {torch.backends.cudnn.enabled}')
    # See how many devices are around
    logger.debug(f'device count: {torch.cuda.device_count()}')
    # Set torch to use a particular device
    torch.cuda.set_device(1)
    logger.debug(f'using device: {torch.cuda.current_device()}')



def data_prep():

    # Data preparation:

    os.makedirs(PATH + 'models', exist_ok=True)
    os.makedirs(PATH + 'tmp', exist_ok=True)
    #ls {PATH}


# ## Multi-label versus single-label classification

def get_1st(path): return glob(f'{path}/*.*')[0]


def cats_dogs_single_label():
    dc_path = "/mnt/samsung_1tb/Data/fastai/dogs_cats/data/dogscats/valid/"
    list_paths = [get_1st(f"{dc_path}cats"), get_1st(f"{dc_path}dogs")]
    plots_from_files(list_paths, titles=["cat", "dog"], maintitle="Single-label classification")


# In single-label classification each sample belongs to one class. In the previous example, each image is either a *dog* or a *cat*.

def planet_multi_label():
    list_paths = [f"{PATH}train-jpg/train_0.jpg", f"{PATH}train-jpg/train_1.jpg"]
    titles=["haze primary", "agriculture clear primary water"]
    plots_from_files(list_paths, titles=titles, maintitle="Multi-label classification")


# In multi-label classification each sample can belong to one or more clases.
# In the previous example, the first images belongs to two clases: *haze* and *primary*.
# The second image belongs to four clases: *agriculture*, *clear*, *primary* and  *water*.

# ## Multi-label models for Planet dataset

def multi_label_models():
    logger.debug('>>multi_label_models')
    metrics=[f2]
    f_model = resnet34

    label_csv = f'{PATH}train_v2.csv'
    n = len(list(open(label_csv)))-1
    val_idxs = get_cv_idxs(n)
    logger.debug(f'--multi_label_models() {val_idxs}, {f_model}, {metrics}, {label_csv}')
    return val_idxs, f_model, metrics, label_csv


# We use a different set of data augmentations for this dataset - we also allow vertical flips,
# since we don't expect vertical orientation of satellite images to change our classifications.

def get_data(sz, val_idxs, f_model, label_csv):
    tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_top_down, max_zoom=1.05)
    return ImageClassifierData.from_csv(PATH, 'train-jpg', label_csv, tfms=tfms,
                    suffix='.jpg', val_idxs=val_idxs, test_name='test-jpg')


def train_data(val_idxs, f_model, label_csv):
    start = time.time()
    logger.debug('>>train_data')
    data = get_data(256, val_idxs, f_model, label_csv)
    x,y = next(iter(data.val_dl))
    logger.debug('--train_data() y: {0}'.format(y))

    logger.debug('--train_data() data.classes, y: {0}'.format(list(zip(data.classes, y[0]))))
    plt.imshow(data.val_ds.denorm(to_np(x))[0]*1.4);
    #very small, Jeremy wouldn't use this for imagenet
    #but satellite imagery is very different, actually orks well
    sz=64
    data = get_data(sz, val_idxs, f_model, label_csv)
    data = data.resize(int(sz*1.3), 'tmp')
    end = time.time()
    elapsed = end - start
    logger.debug('>>train_data() took {elapsed}sec')
    return data, sz

def find_lr(f_model, data, metrics):
    start = time.time()
    logger.debug('>>find_lr')
    learn = ConvLearner.pretrained(f_model, data, metrics=metrics)

    lrf=learn.lr_find()
    learn.sched.plot()
    #added  'if not in_ipynb(): plt.show()' to sgdr.py LR_Finder
    end = time.time()
    elapsed = end - start
    logger.debug('>>find_lr() took {elapsed}sec')
    return learn

def iterate_0(learn, val_idxs, f_model, label_csv, sz):
    start = time.time()
    lr = 0.2
    learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
    #differential lr's, finer rate on later later layers
    lrs = np.array([lr/9,lr/3,lr])
    learn.unfreeze()
    learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
    learn.save(f'{sz}')
    learn.sched.plot_loss()
    end = time.time()
    elapsed = end - start
    logger.debug('>>iterate_0() took {elapsed}sec')
    return lrs, lr, learn

def iterate_1(learn, lr, lrs, val_idxs, f_model, label_csv):
    start = time.time()
    logger.debug('>>iterate_1')
    #reset size
    sz = 128
    learn.set_data(get_data(sz, val_idxs, f_model, label_csv))
    learn.freeze()
    learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
    learn.unfreeze()
    learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
    learn.save(f'{sz}')
    end = time.time()
    elapsed = end - start
    logger.debug('>>iterate_1() took {elapsed}sec')

def iterate_2(learn, lr, lrs, val_idxs, f_model, label_csv):
    start = time.time()
    logger.debug('>>iterate_2')
    sz=256
    learn.set_data(get_data(sz, val_idxs, f_model, label_csv))
    learn.freeze()
    learn.fit(lr, 3, cycle_len=1, cycle_mult=2)
    learn.unfreeze()
    learn.fit(lrs, 3, cycle_len=1, cycle_mult=2)
    learn.save(f'{sz}')
    end = time.time()
    elapsed = end - start
    logger.debug('>>iterate_2() took {elapsed}sec')

def run_tta(learn, data):
    start = time.time()
    logger.debug('>>run_tta')
    probs = learn.predict()
    f2_without_TTA = f2(probs, data.val_y)
    probs, y = learn.TTA()
    probs = np.mean(probs, 0)
    f2_with_TTA = f2(probs, y)
    logger.debug(f' without tta {f2_without_TTA} , with tta {f2_with_TTA}')
    end = time.time()
    elapsed = end - start
    logger.debug('>>run_tta() took {elapsed}sec')
    return probs, y

def create_submission(learn, data, probs, y, iteration):
    start = time.time()
    logger.debug('>>create_sumission')
    #see https://towardsdatascience.com/kaggle-planet-competition-how-to-land-in-top-4-a679ff0013ba

    threshold = opt_th(probs, y)
    logger.debug(f'threshold: {threshold}')
    test_preds, _ = learn.TTA(is_test=True)
    preds = np.mean(test_preds, axis=0)
    classes = np.array(data.classes)
    res = np.array(["".join(classes[(np.where(pp>threshold))]) for pp in preds])
    filenames = np.array([os.path.basename(fn).split('.')[0] for fn in data.test_ds.fnames])
    frame = pd.DataFrame(res, index=filenames, columns=['tags'])
    frame.to_csv(f'{PATH}submission/planet_amazon_resnet34_submission{iteration}.csv', index_label='image_name')

    logger.debug(f'submission{iteration} file created')
    end = time.time()
    elapsed = end - start
    logger.debug('>>create_sumission() took {elapsed}sec')

def create_submission_b(learn, data, probs, y, iteration):

    logger.debug('>>create_sumission_b')
    threshold = opt_th(probs, y)
    prob_preds, y = learn.TTA(is_test=True)
    #error IndexError: too many indices for array
    classes = np.array(data.classes, dtype=str)
    res = [" ".join(classes[np.where(pp > threshold)]) for pp in prob_preds]
    test_fnames = [os.path.basename(f).split(".")[0] for f in data.test_ds.fnames]
    test_df = pd.DataFrame(res, index=test_fnames, columns=['tags'])
    test_df.to_csv(f'{PATH}submission/planet_amazon_resnet34_submission_b_{iteration}.csv', index_label='image_name')

def workflow():
    start = time.time()
    cuda_setup()
    #data_prep()
    #cats_dogs_single_label()
    '''
    planet_multi_label()
    val_idxs, f_model, metrics, label_csv = multi_label_models()
    data, sz = train_data(val_idxs, f_model, label_csv)
    learn = find_lr(f_model, data, metrics)

    lrs, lr, learn = iterate_0(learn, val_idxs, f_model, label_csv, sz)
    iterate_1(learn, lr, lrs, val_idxs, f_model, label_csv)
    iterate_2(learn, lr, lrs, val_idxs, f_model, label_csv)
    probs, y = run_tta(learn, data)
    create_submission(learn, data, probs, y, iteration=1)
    create_submission_b(learn, data, probs, y, iteration=1)
    end = time.time()
    elapsed = end - start
    logger.debug('>>workflow() took {elapsed}sec')
    '''

if __name__ == "__main__":
    workflow()