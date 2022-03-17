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
from tqdm.notebook import tqdm, trange
#the line above may fail on google colab so you can use the line below in that case but progress bars will looks less nice
#from tqdm import tqdm, trange
import matplotlib as mpl
import matplotlib.pyplot as plt

ModelClass=RandomForestClassifier


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
data_set = load_digits()
X = data_set['data']
y = data_set['target']


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/3, random_state=SEED)

# in case repetitions are desired
permutations=[np.random.permutation(X_train.shape[0]) for _ in range(n_repeats)]

random_results = []

for i_repeat in tqdm(range(n_repeats)):
    learner = ModelClass()
    for i_query in tqdm(range(1,n_queries),leave=False):
        query_indices=permutations[i_repeat][:1+i_query]
        learner=learner.fit(X=X_train[query_indices, :], y=y_train[query_indices])
        score = learner.score(X_test, y_test)
        
        random_results.append(ResultsRecord('random', i_query, score))

committee_results = []

n_members=[2, 4, 8, 16]

for i_repeat in tqdm(range(n_repeats)):
    for i_members in tqdm(n_members, desc=f'Round (no. members) {i_repeat}',leave=False):
        X_pool = X_train.copy()
        y_pool = y_train.copy()

        start_indices = permutations[i_repeat][:1]

        committee_members = [ActiveLearner(estimator=ModelClass(),
                                           X_training=X_train[start_indices, :],
                                           y_training=y_train[start_indices],
                                           ) for _ in range(i_members)]

        committee = Committee(learner_list=committee_members,
                              query_strategy=vote_entropy_sampling)

        X_pool = np.delete(X_pool, start_indices, axis=0)
        y_pool = np.delete(y_pool, start_indices)

        for i_query in tqdm(range(1, n_queries),desc=f'Points {i_repeat}',leave=False):
            query_idx, query_instance = committee.query(X_pool)

            committee.teach(
                X=X_pool[query_idx].reshape(1, -1),
                y=y_pool[query_idx].reshape(1, )
            )
            committee._set_classes() #this is needed to update for unknown class labels

            X_pool = np.delete(X_pool, query_idx, axis=0)
            y_pool = np.delete(y_pool, query_idx)

            score = committee.score(X_test, y_test)

            committee_results.append(ResultsRecord(
                f'committe_{i_members}',
                i_query,
                score))

df_results = pd.concat([pd.DataFrame(results)
                        for results in
                        [random_results, committee_results]])

df_results_mean=df_results.groupby(['estimator','query_id']).mean()
df_results_std=df_results.groupby(['estimator','query_id']).std()

df_mean=df_results_mean.reset_index().pivot(index='query_id', columns='estimator', values='score')
df_std=df_results_std.reset_index().pivot(index='query_id', columns='estimator', values='score')

df_mean.plot(figsize=(8.5,6), yerr=df_std)
plt.grid('on')