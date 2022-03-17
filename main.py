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
%matplotlib inline

ModelClass=RandomForestClassifier


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

ResultsRecord = namedtuple('ResultsRecord', ['estimator', 'query_id', 'score'])