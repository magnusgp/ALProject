from collections import namedtuple
import pandas as pd
import numpy as np
import sys
#!{sys.executable} -m pip install tqdm
#!{sys.executable} -m pip install modAL
from modAL.disagreement import vote_entropy_sampling
from modAL.models import ActiveLearner, Committee
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
#from tqdm.notebook import tqdm, trange
#the line above may fail on google colab so you can use the line below in that case but progress bars will looks less nice
from tqdm import tqdm, trange
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn

lr=RandomForestClassifier()


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

ResultsRecord = namedtuple('ResultsRecord', ['estimator', 'query_id', 'score'])

SEED = 1 # Set our RNG seed for reproducibility.

n_queries = 75 # You can lower this to decrease run time

# You can increase this to get error bars on your evaluation.
# You probably need to use the parallel code to make this reasonable to compute
n_repeats = 3

from sklearn.datasets import load_iris, load_digits

# loading the data dataset
data_set = load_iris()
X = data_set['data']
y = data_set['target']

# Train test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=1/3, random_state=SEED)

Xpool = Xtrain.copy()
ypool = ytrain.copy()

np.random.seed(42) # random seed to ensure same results but feel free to change
addn=2 #samples to add each time
#randomize order of pool to avoid sampling the same subject sequentially
order=np.random.permutation(range(len(Xpool)))

### Random sampling baseline ###

#samples in the pool
poolidx=np.arange(len(Xpool),dtype='int32')
ninit = 10 #initial samples
#initial training set
trainset=order[:ninit]
Xtrain=np.take(Xpool,trainset,axis=0)
ytrain=np.take(ypool,trainset,axis=0)

#remove data from pool
poolidx=np.setdiff1d(poolidx,trainset)

model=lr
testacc=[]
print("Random Sampling")
for i in tqdm(range(25)):
    #TODO fit model
    #Hints below:
    data = np.take(Xpool,order[:ninit+i*addn],axis=0)
    labels = np.take(ypool,order[:ninit+i*addn],axis=0)
    model.fit(data, labels)
    #predict and calculate the accuracy
    score = model.score(Xtest, ytest)
    #calculate accuracy on test set
    testacc.append((ninit+i*addn,score)) #add in the accuracy
    # print('Model: LR, %i random samples'%(ninit+i*addn))

### Uncertainty Sampling ###

testacc_al=[]
trainset=order[:ninit]
Xtrain=np.take(Xpool,trainset,axis=0)
ytrain=np.take(ypool,trainset,axis=0)
poolidx=np.arange(len(Xpool),dtype=np.int32)
poolidx=np.setdiff1d(poolidx,trainset)
print("Uncertainty Sampling")
for i in tqdm(range(25)):
    # Fit and Accuracy
    ml = model.fit(Xtrain, ytrain)
    acc = model.score(Xtest, ytest)

    # Find most uncertain sample
    label_probs = ml.predict_proba(Xpool[poolidx])
    x_star = np.argmax(1 - np.max(label_probs, axis=1))
    # print(f"Most uncertain sample index: {x_star}")
    
    # Remove datapoint from pool and add to dataset
    poolidx = np.delete(poolidx, x_star, axis = 0)

    Xtrain = np.vstack((Xtrain, Xpool[x_star]))
    ytrain = np.append(ytrain, ypool[x_star])

    testacc_al.append((ninit+i*addn,acc))

### Query By Committee ###
testacc_qbc=[]
ncomm=10
trainset=order[:ninit]
Xtrain=np.take(Xpool,trainset,axis=0)
ytrain=np.take(ypool,trainset,axis=0)
poolidx=np.arange(len(Xpool),dtype=np.int32)
poolidx=np.setdiff1d(poolidx,trainset)
print("Query by Committee")
for i in tqdm(range(25)):
    ypool_lab = []

    for k in range(ncomm):
        Xtr, ytr = sklearn.utils.resample(Xtrain, ytrain, stratify=ytrain)

        model.fit(Xtr, ytr)

        ypool_lab.append(model.predict(Xpool[poolidx]))
    
    ypool_p = (np.mean(np.array(ypool_lab) == 1,0), np.mean(np.array(ypool_lab)==2,0))
    ypool_p = np.array(ypool_p).T

    model.fit(Xtrain, ytrain)
    ye = model.predict(Xtest)
    testacc_qbc.append((len(Xtrain), sklearn.metrics.accuracy_score(ytest,ye)))

    ypool_p_sort_idx = np.argsort(-ypool_p.max(1))
    Xtrain = np.concatenate((Xtrain, Xpool[poolidx[ypool_p_sort_idx][-addn:]]))
    ytrain = np.concatenate((ytrain, ypool[poolidx[ypool_p_sort_idx][-addn:]]))

    poolidx = np.setdiff1d(poolidx, poolidx[ypool_p_sort_idx[-addn:]])

#Plot learning curve
random = tuple(np.array(testacc).T)
least_confidence = tuple(np.array(testacc_al).T)
qbc = tuple(np.array(testacc_qbc).T)

plt.plot(*random);
plt.plot(*least_confidence);
plt.plot(*qbc);
plt.legend(('random sampling','uncertainty sampling','QBC'));
print(f"Final accuracies: {random[-1]} (Random), {least_confidence[-1]} (QBC), {qbc[-1]} (Least Confidence)")
plt.show();