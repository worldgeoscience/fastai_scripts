
# coding: utf-8

# ## Movielens

import logging

from fastai.learner import *
from fastai.column_data import *

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Data available from http://files.grouplens.org/datasets/movielens/ml-latest-small.zip

path='/mnt/samsung_1tb/Data/fastai/movielens/ml-latest-small/'

# We're working with the movielens data, which contains one rating per row, like this:
ratings = pd.read_csv(path+'ratings.csv')
logger.debug(ratings.head(n=2))

# Just for display purposes, let's read in the movie names too.
movies = pd.read_csv(path+'movies.csv')
logger.debug(movies.head(n=2))

links = pd.read_csv(path + 'links.csv')
logger.debug(links.head(n=2))

tags = pd.read_csv(path + 'tags.csv')
logger.debug(tags.head(n=2))

# get 20% at random and store indexes for validation
val_idxs = get_cv_idxs(len(ratings))
# weight decay
wd = 2e-4
# size of embedding matrix
n_factors = 50

# ## Collaborative filtering
#pass in data, one row (effectively), one col (effectively), one values (effectively)
cf = CollabFilterDataset.from_csv(path, 'ratings.csv', 'userId', 'movieId', 'rating')

def excel_crosstab():
    # ## Create subset for Excel

    # We create a crosstab of the most popular movies and most movie-addicted users which we'll copy into Excel for creating a simple example. This isn't necessary for any of the modeling below however.

    g=ratings.groupby('userId')['rating'].count()
    topUsers=g.sort_values(ascending=False)[:15]

    g=ratings.groupby('movieId')['rating'].count()
    topMovies=g.sort_values(ascending=False)[:15]

    top_r = ratings.join(topUsers, rsuffix='_r', how='inner', on='userId')
    top_r = top_r.join(topMovies, rsuffix='_r', how='inner', on='movieId')

    logger.debug(pd.crosstab(top_r.userId, top_r.movieId, top_r.rating, aggfunc=np.sum))


def find_optimal_learning_rate():
    logger.debug('>>collaborative_filtering')

    #embedding matrix size, validation data, batch size = 64, optimizer = Adam
    learn = cf.get_learner(n_factors, val_idxs, 64, opt_fn=optim.Adam)

    learn.lr_find()
    learn.sched.plot()
    plt.show()
    return learn, wd

def collaborative_filtering_fit(learn, wd):
    learn.fit(1e-2, 2, wds=wd, cycle_len=1, cycle_mult=2)
    # Let's compare to some benchmarks. Here's [some benchmarks](https://www.librec.net/release/v1.3/example.html) on the same dataset for the popular Librec system for collaborative filtering. They show best results based on [RMSE](http://www.statisticshowto.com/rmse/) of 0.91. We'll need to take the square root of our loss, since we use plain MSE.

    logger.debug(math.sqrt(0.776))
    # Looking good - we've found a solution better than any of those benchmarks! Let's take a look at how the predictions compare to actuals for this model.

    preds = learn.predict()

    y=learn.data.val_y
    sns.jointplot(preds, y, kind='hex', stat_func=None)
    plt.show()
    return cf

def analyze_results(learn):
    # ## Analyze results

    # ### Movie bias

    movie_names = movies.set_index('movieId')['title'].to_dict()
    g=ratings.groupby('movieId')['rating'].count()
    topMovies=g.sort_values(ascending=False).index.values[:3000]
    topMovieIdx = np.array([cf.item2idx[o] for o in topMovies])
    #get the pytorch model, call the fasiai learner property
    m=learn.model
    m.cuda()

    # First, we'll look at the movie bias term. Here, our input is the movie id (a single id), and the output is the movie bias (a single float).
    #m.ib is movie bias, note models require variables not tensors so use V
    movie_bias = to_np(m.ib(V(topMovieIdx)))
    #can move model to cpu with m.cpu()

    logger.debug(movie_bias)

    movie_ratings = [(b[0], movie_names[i]) for i,b in zip(topMovies,movie_bias)]

    # Now we can look at the top and bottom rated movies. These ratings are corrected for different levels of reviewer sentiment, as well as different types of movies that different reviewers watch.
    logger.debug('worst movies')
    logger.debug(sorted(movie_ratings, key=lambda o: o[0])[:15])
    logger.debug(sorted(movie_ratings, key=itemgetter(0))[:15])
    logger.debug('best movies')
    logger.debug(sorted(movie_ratings, key=lambda o: o[0], reverse=True)[:15])

    # ### Embedding interpretation

    # We can now do the same thing for the embeddings.
    movie_emb = to_np(m.i(V(topMovieIdx)))
    logger.debug(movie_emb.shape)
    return movie_emb, movie_names, topMovies

def simplify_results(movie_emb, movie_names, topMovies):
    # Because it's hard to interpret 50 embeddings, we use [PCA](https://plot.ly/ipython-notebooks/principal-component-analysis/) to simplify them down to just 3 vectors.

    from sklearn.decomposition import PCA
    #lower rank approximation of our matrix
    pca = PCA(n_components=3)
    movie_pca = pca.fit(movie_emb.T).components_

    logger.debug(movie_pca.shape)

    fac0 = movie_pca[0]
    movie_comp = [(f, movie_names[i]) for f,i in zip(fac0, topMovies)]

    # Here's the 1st component. It seems to be 'easy watching' vs 'serious'.

    logger.debug(sorted(movie_comp, key=itemgetter(0), reverse=True)[:10])

    logger.debug(sorted(movie_comp, key=itemgetter(0))[:10])

    fac1 = movie_pca[1]
    movie_comp = [(f, movie_names[i]) for f,i in zip(fac1, topMovies)]

    # Here's the 2nd component. It seems to be 'CGI' vs 'dialog driven'.

    logger.debug(sorted(movie_comp, key=itemgetter(0), reverse=True)[:10])

    logger.debug(sorted(movie_comp, key=itemgetter(0))[:10])

    # We can draw a picture to see how various movies appear on the map of these components. This picture shows the first two components.

    idxs = np.random.choice(len(topMovies), 50, replace=False)
    X = fac0[idxs]
    Y = fac1[idxs]
    plt.figure(figsize=(15,15))
    plt.scatter(X, Y)
    for i, x, y in zip(topMovies[idxs], X, Y):
        plt.text(x,y,movie_names[i], color=np.random.rand(3)*0.7, fontsize=11)
    plt.show()


# ## Collab filtering from scratch
def dot_product_example():
    logger.debug('>>dot_product_example()')
    # ### Dot product example

    #create pytorch Tensor
    a = T([[1.,2],[3,4]])
    b = T([[2.,2],[10,10]])
    logger.debug(f'{a},{b}')

    logger.debug(a*b)

    #Dot Product: mult then sum on 1st dimension ie column
    (a*b).sum(1)

    model = DotProduct()
    #pytorch automatically appses this to forward()
    calc = model(a, b)
    logger.debug(calc)

#create a pytorch module - can use as a layer
class DotProduct(nn.Module):
    def forward(self, u, m): return (u*m).sum(1)


def dot_product_model():
    logger.debug('>>dot_product_model()')
    # ### Dot product model

    u_uniq = ratings.userId.unique()
    #super handy and well worth learning how to do this in general for ML problems
    user2idx = {o:i for i,o in enumerate(u_uniq)}
    ratings.userId = ratings.userId.apply(lambda x: user2idx[x])

    m_uniq = ratings.movieId.unique()
    movie2idx = {o:i for i,o in enumerate(m_uniq)}
    ratings.movieId = ratings.movieId.apply(lambda x: movie2idx[x])

    n_users=int(ratings.userId.nunique())
    n_movies=int(ratings.movieId.nunique())

    x = ratings.drop(['rating', 'timestamp'], axis=1)
    y = ratings['rating'].astype(np.float32)

    #Rossmann ColumnarModelData - only fastai bit
    data = ColumnarModelData.from_data_frame(path, val_idxs, x, y, ['userId', 'movieId'], 64)

    wd = 1e-5

    #as avoiding using fastai, not running learner_find()

    #instantiate this pytorch object
    model = EmbeddingDot(n_users, n_movies).cuda()

    #pytorch SGD: params, lr=required, momentum=0, dampening=0, weight_decay=0, nesterov=False
    #pytorch optimizer will update the weights (called parameters by pytorch)
    opt = optim.SGD(model.parameters(), 1e-1, momentum=0.9, weight_decay=wd)

    #fastai fit - a lower level function taking:
    # model (model): any pytorch module,
    # data (ModelData): see ModelData class and subclasses
    # epochs(int): number of epochs
    # opt: optimizer. Example: opt=optim.Adam(net.parameters())
    # crit: loss function to optimize. Example: F.cross_entropy

    #for each epoch go through each minibatch and do one step of our optimizer
    vals = fit(model, data, 3, opt, F.mse_loss)
    logger.debug(vals)

    logger.debug('--dot_product_model() revise learning rate')
    #fastai
    set_lrs(opt, 0.01)

    vals = fit(model, data, 3, opt, F.mse_loss)
    logger.debug(vals)
    #a small improvement over first pass
    return data, n_users, n_movies

class EmbeddingDot(nn.Module):
    def __init__(self, n_users, n_movies):
        super().__init__()
        #A simple lookup table that stores embeddings of a fixed dictionary and size.
        #rows = n_users, cols = n_factors
        self.u = nn.Embedding(n_users, n_factors)
        self.m = nn.Embedding(n_movies, n_factors)
        #see kaiming he initialization, eg http://www.jefkine.com/deep/2016/08/08/initialization-of-deep-networks-case-of-rectifiers/
        #note we use _ after uniform to fill in this matrix in place
        self.u.weight.data.uniform_(0,0.05)
        self.m.weight.data.uniform_(0,0.05)
        
    def forward(self, cats, conts):
        #passing categorical vars and continuous vars (note there are none in this case)
        #where usera and movies are mini-batches
        users,movies = cats[:,0],cats[:,1]
        u,m = self.u(users),self.m(movies)
        #dot product
        return (u*m).sum(1)

def get_emb(ni,nf):
    '''
    :param ni: number of rows
    :param nf: number of columns in embedding matrix
    :return e: initialized embedding
    '''
    e = nn.Embedding(ni, nf)
    e.weight.data.uniform_(-0.01,0.01)
    return e

class EmbeddingDotBias(nn.Module):
    def __init__(self, n_users, n_movies, max_rating, min_rating):
        super().__init__()
        #ub = user bias note 1 column, mb is movie bias
        (self.u, self.m, self.ub, self.mb) = [get_emb(*o) for o in [
            (n_users, n_factors), (n_movies, n_factors), (n_users,1), (n_movies,1)
        ]]
        self.max_rating = max_rating
        self.min_rating = min_rating

    #in pytorch forward takes *args
    #stuck on how to pass in min_rating, max_rating to forward() method
    #see fastai.model.fit()
    #       ...
    #       loss = stepper.step(V(x),V(y))
    #then fastai.model.stepper.step
    #       ...
    #       output = self.m(*xs) -> calls forward in module but how to pass in *args?
    #passed in max and min ratings to initializer instead
    def forward(self, cats, conts):
        users,movies = cats[:,0],cats[:,1]
        um = (self.u(users)* self.m(movies)).sum(1)
        #using pytorch .squeeze to add an additional axis - broadcasting to make vector same dim as matrix
        res = um + self.ub(users).squeeze() + self.mb(movies).squeeze()
        #F is torch.nn.functional, contains a bunch of useful functions
        #here we convert result to siggmoid (between 0-1) then scale to between 1-5
        res = F.sigmoid(res) * (self.max_rating-self.min_rating) + self.min_rating
        return res

def bias(data, cf):
    logger.debug('>>bias()')
    # ### Bias
    #add a constant for user and for movie, and add in this bias

    min_rating,max_rating = ratings.rating.min(),ratings.rating.max()
    logger.debug(min_rating)
    logger.debug(min_rating)

    wd = 2e-4
    model = EmbeddingDotBias(cf.n_users, cf.n_items, max_rating, min_rating).cuda()
    opt = optim.SGD(model.parameters(), 1e-1, weight_decay=wd, momentum=0.9)
    #fastai fit: model, data, epochs, opt, crit, metrics=None, callbacks=None, stepper=Stepper, **kwargs
    vals = fit(model, data, 3, opt, F.mse_loss)
    logger.debug(vals)

    #decrease learning rate
    set_lrs(opt, 1e-2)

    vals = fit(model, data, 3, opt, F.mse_loss)
    logger.debug(vals)
    #pretty good


# ### Mini net

class EmbeddingNet(nn.Module):
    #NN with one hidden layer
    def __init__(self, n_users, n_movies, n_factors, max_rating, min_rating, nh=10, p1=0.05, p2=0.75):
        super().__init__()
        (self.u, self.m) = [get_emb(*o) for o in [
            (n_users, n_factors), (n_movies, n_factors)]]
        #nn.Linear, using rows=n_factors*2 as added user and movie embedding, columns=nh (number of activations)
        self.lin1 = nn.Linear(n_factors*2, nh)
        #columns = 1 as want to predict a single rating
        self.lin2 = nn.Linear(nh, 1)
        self.drop1 = nn.Dropout(p1)
        self.drop2 = nn.Dropout(p2)
        self.max_rating = max_rating
        self.min_rating = min_rating
        
    def forward(self, cats, conts):
        users,movies = cats[:,0],cats[:,1]
        #torch.cat conactenates, here specifying to use 1st dimension-concat cols to get get longer rows
        x = self.drop1(torch.cat([self.u(users),self.m(movies)], dim=1))
        x = self.drop2(F.relu(self.lin1(x)))
        return F.sigmoid(self.lin2(x)) * (self.max_rating-self.min_rating+1) + self.min_rating-0.5

def mini_net(n_users, n_movies, data):
    logger.debug('>>bias()')
    wd=1e-5
    min_rating, max_rating = ratings.rating.min(), ratings.rating.max()
    model = EmbeddingNet(n_users, n_movies, n_factors, max_rating, min_rating).cuda()
    opt = optim.Adam(model.parameters(), 1e-3, weight_decay=wd)

    vals = fit(model, data, 3, opt, F.mse_loss)
    logger.debug(vals)

    set_lrs(opt, 1e-3)

    vals = fit(model, data, 3, opt, F.mse_loss)
    logger.debug(vals)


def workflow():
    start = time.time()

    excel_crosstab()
    learn, wd = find_optimal_learning_rate()
    collaborative_filtering_fit(learn, wd)
    movie_emb, movie_names, topMovies = analyze_results(learn)
    simplify_results(movie_emb, movie_names, topMovies)

    #part2

    #dot_product_example()
    data, n_users, n_movies = dot_product_model()
    #bias(data, cf)
    mini_net(n_users, n_movies, data)

    end = time.time()
    elapsed = end - start
    logger.debug(f'>>workflow() took {elapsed}sec')

if __name__ == "__main__":
    workflow()