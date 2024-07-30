import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils import *


train = pd.read_csv('./input/train.csv')
train['number'] = train['number'].astype(str)
train = train.loc[~train[label_name].isna()].reset_index(drop=True)
train['idx'] = train.index

if args.label_name != 'delivering efficiency (% ID)':
    train = train.loc[(train['onehot_type of the preparation_free drug']!=1)&(train[label_name]>=0.1)].reset_index(drop=True)
    # train[label_name] = np.log1p(train[label_name].values)
    train[label_name] = np.log10(train[label_name].apply(lambda x:max(x,1e-6)))
else:
    train[label_name] = np.log10(train[label_name].apply(lambda x:max(x,1e-6)))
print('train nsamples:',train.shape)
# preparation = 'polymer-based nanoparticles'
# preparation = 'liposome'
# preparation = 'solid lipid nanoparticles'
# train = train.loc[train[f'onehot_type of the preparation_{preparation}']==1].reset_index(drop=True)
feature_name = ['feature%s'%i if col not in [id_name,'delivering efficiency (% ID)','ratio of preparation/free drug','number'] else col for i,col in enumerate(train.columns) ]
feature_map_dic = {}
for i,col in enumerate(train.columns):
    if col not in [id_name,'delivering efficiency (% ID)','ratio of preparation/free drug','number']:
        feature_map_dic['feature%s'%i] = col
train.columns = feature_name
feature_name = [col for i,col in enumerate(train.columns) if col not in [id_name,'delivering efficiency (% ID)','ratio of preparation/free drug','number']]
lgb_config = {
    'lgb_params':{
                  'objective' : 'regression',
#                   'metric' : 'auc',
                  'metric': 'rmse',
                  'boosting': 'gbdt',
                  'max_depth' : 5,
                  'num_leaves' : 12,
                  'learning_rate' : 0.03,
                  'bagging_freq': 1,
                  'bagging_fraction' : 0.7,
                  'feature_fraction' : 0.7,
                  'max_bin': 7,
                  'min_data_in_leaf': 20,
                  # 'min_sum_heassian_in_leaf': 10,
#                   'tree_learner': 'serial',
                  'boost_from_average': 'false',
                  # 'lambda_l1' : 1,
                  # 'lambda_l2' : 1,
                  'verbosity' : -1,
                  'num_threads': 12,
                  'seed': args.seed,
    },
    'rounds':2000,
    'early_stopping_rounds':300,
    'verbose_eval':100,
    'folds':10,
    'seed':args.seed,
    'remark':args.remark
}


print(train.shape)

oof,sub = Lgb_train_and_predict(train,None,feature_map_dic,feature_name,lgb_config,run_id=None)

# oof = pd.read_csv('./output/run_lgb_20220106_110141/oof.csv')
if label_name == 'delivering efficiency (% ID)':
    train['error'] = (oof[label_name].rank()-train[label_name].rank()).abs()
    ns = train.loc[train['error']>300,'number'].unique()
    train = train.loc[~train['number'].isin(ns)].drop(['error'],axis=1).reset_index(drop=True)
    # lgb_config['lgb_params']['min_data_in_leaf'] = 5
else:
    train['error'] = (oof[label_name].rank()-train[label_name].rank()).abs()
    ns = train.loc[(train['error']>200),'number'].unique()
    train = train.loc[(~train['number'].isin(ns))].drop(['error'],axis=1).reset_index(drop=True)
    print(train[label_name].min(),train[label_name].max())
    train = train.loc[(train[label_name]>=np.log10(0.5))&(train[label_name]<=np.log10(100))].reset_index(drop=True)

print(train.shape)

oof,sub = Lgb_train_and_predict(train,None,feature_map_dic,feature_name,lgb_config,run_id=None)
