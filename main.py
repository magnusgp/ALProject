from collections import namedtuple
from random import Random
import pandas as pd
import numpy as np
import sys
#!{sys.executable} -m pip install tqdm
#!{sys.executable} -m pip install modAL
from modAL.disagreement import vote_entropy_sampling
from modAL.models import ActiveLearner, Committee
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
#from tqdm.notebook import tqdm, trange
#the line above may fail on google colab so you can use the line below in that case but progress bars will looks less nice
from tqdm import tqdm, trange
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

SEED = 665
N_samples = 25
lr = RandomForestClassifier(random_state=SEED)

from sklearn.datasets import load_iris, load_digits

# loading the data dataset
data_set = load_digits()
X = data_set['data']
y = data_set['target']

# Train test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=1/3, random_state=SEED)

Xpool = Xtrain.copy()
ypool = ytrain.copy()

#randomize order of pool to avoid sampling the same subject sequentially
np.random.seed(SEED) # random seed to ensure same results but feel free to change
order=np.random.permutation(range(len(Xpool)))
addn=2 #samples to add each time
ninit = 10 #initial samples


### Random sampling baseline ###
model=lr
testacc=[]
trainset=order[:ninit]
Xtrain=np.take(Xpool,trainset,axis=0)
ytrain=np.take(ypool,trainset,axis=0)
poolidx=np.arange(len(Xpool),dtype=np.int32)
poolidx=np.setdiff1d(poolidx,trainset)
print("Random Sampling")
for i in tqdm(range(N_samples)):
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
for i in tqdm(range(N_samples)):
    # Fit and Accuracy
    ml = model.fit(Xtrain, ytrain)
    acc = model.score(Xtest, ytest)
    # testacc_al.append((len(Xtrain), acc))

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
print("QBC")
for i in tqdm(range(N_samples)):
    ypool_lab = []

    for k in range(ncomm):
        Xtr, ytr = sklearn.utils.resample(Xtrain, ytrain, stratify=ytrain)

        model.fit(Xtr, ytr)

        ypool_lab.append(model.predict(Xpool[poolidx]))
    
    ypool_p = (np.mean(np.array(ypool_lab) == 0,0), np.mean(np.array(ypool_lab) == 1,0),
                                                    np.mean(np.array(ypool_lab) ==2,0),
                                                    np.mean(np.array(ypool_lab) ==3,0),
                                                    np.mean(np.array(ypool_lab) ==4,0),
                                                    np.mean(np.array(ypool_lab) ==5,0),
                                                    np.mean(np.array(ypool_lab) ==6,0),
                                                    np.mean(np.array(ypool_lab) ==7,0),
                                                    np.mean(np.array(ypool_lab) ==8,0),
                                                    np.mean(np.array(ypool_lab)==9,0))
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
print(f"Final accuracies: {random[-1]} (Random), {qbc[-1]} (QBC), {least_confidence[-1]} (Least Confidence)")
plt.show();