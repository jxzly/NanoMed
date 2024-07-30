import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from typing import List, Tuple, Union
import os,time,datetime
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, pearsonr
import lightgbm as lgb
# import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, StratifiedKFold,GroupKFold
from tqdm import tqdm
import random
import pickle
from contextlib import contextmanager

@contextmanager
def Timer(title):
    'timing function'
    t0 = datetime.datetime.now()
    yield
    print("%s - done in %s"%(title, (datetime.datetime.now() - t0)))
    return None

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, default='./input/')
parser.add_argument("--label_name", type=str, default='ratio of preparation/free drug')
parser.add_argument("--do_train", action='store_true', default=False)
parser.add_argument("--test", action='store_true', default=False)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--remark", type=str, default='')

args = parser.parse_args()

os.makedirs('./input',exist_ok=True)
os.makedirs('./output',exist_ok=True)

root = args.root
id_name = 'idx'
label_name = args.label_name

def Seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

Seed_everything(args.seed)

def Write_log(logFile,text,isPrint=True):
    if isPrint:
        print(text)
    logFile.write(text)
    logFile.write('\n')
    return None

def Metric(preds,labels):
    rmse = np.sqrt(np.mean((preds-labels)**2))
    spearman = spearmanr(preds,labels)[0]
    pearson = pearsonr(preds,labels)[0]
    return rmse, pearson

def Rf_train_and_predict(train, test, features, config, output_root='./output/', run_id=None):
    if not run_id:
        run_id = 'run_rf_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        while os.path.exists(output_root+run_id+'/'):
            time.sleep(1)
            run_id = 'run_rf_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_root + 'rf_tmp1/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
    else:
        output_path = output_root + run_id + '/'

    oof,sub = None ,None
    if train is not None:
        os.system(f'cp ./*.py {output_path}')
        os.system(f'cp ./*.sh {output_path}')

        oof = train[[id_name]]
        oof['fold'] = -1

        if isinstance(label_name,list):
            for l in label_name:
                oof[l] = 0.0
        else:
            oof[label_name] = 0.0
    else:
        oof = None

    if train is not None:
        log = open(output_path + 'train.log','w',buffering=1)
        log.write(str(config)+'\n')
        log.write(str(features)+'\n')
        params = config['rf_params']
        rounds = config['rounds']
        verbose = config['verbose_eval']
        early_stopping_rounds = config['early_stopping_rounds']
        folds = config['folds']
        seed = config['seed']
        remark = config['remark']

        all_valid_metric = []
        all_valid_rmse = []
        feature_importance = []
        # kf = GroupKFold(n_splits=folds)
        # for fold, (trn_index, val_index) in enumerate(kf.split(train,train[label_name],train['cluster'])):
        kf = KFold(n_splits = folds,shuffle=True,random_state=seed)
        for fold, (trn_index, val_index) in enumerate(kf.split(train,train[label_name])):
            print('fold%s begin'%fold)
            print(val_index)
            trn_data = lgb.Dataset(train.loc[trn_index,features], label=train.loc[trn_index,label_name])#,weight=train.loc[trn_index,'weight']
            val_data = lgb.Dataset(train.loc[val_index,features], label=train.loc[val_index,label_name])#,weight=train.loc[val_index,'weight_val']
            model = RandomForestRegressor()
            model.set_params(**params)
            model.fit(train.loc[trn_index,features].values,train.loc[trn_index,label_name].values)

            train_preds = model.predict(train.loc[trn_index,features].values)
            train_metric = Metric(train.loc[trn_index,label_name].values,train_preds)
            Write_log(log,'- fold%s train rmse: %.6f, train_spearman: %.6f, train_mean:%.6f\n'%(fold,train_metric[0],train_metric[1],np.mean(train_preds)))

            valid_preds = model.predict(train.loc[val_index,features].values)
            oof.loc[val_index,label_name] = valid_preds
            # ori_index = train.loc[val_index,'source']=='ext'
            # print('ori metric:',Metric(train.loc[val_index,label_name].values[ori_index],valid_preds[ori_index]))
            # for i in range(len(evals_result_dic['valid_1'][params['metric']])//verbose):
            #     Write_log(log,' - %i round - train_metric: %.6f - valid_metric: %.6f\n'%(i*verbose,evals_result_dic['training'][params['metric']][i*verbose],evals_result_dic['valid_1'][params['metric']][i*verbose]))
            all_valid_rmse.append(Metric(train.loc[val_index,label_name].values,valid_preds)[0])
            all_valid_metric.append(Metric(train.loc[val_index,label_name].values,valid_preds)[-1])
            Write_log(log,'- fold%s valid metric: %.6f, valid_mean:%.6f\n'%(fold,all_valid_metric[-1],np.mean(valid_preds)))

            if all_valid_metric[-1] > 0.1:
                with open(output_path + '/fold%s.ckpt'%fold,'wb') as f:
                    pickle.dump(model,f)
                    f.close()
            else:
                with open(output_path + '/fold%s_.ckpt'%fold,'wb') as f:
                    pickle.dump(model,f)
                    f.close()
            importance = model.feature_importances_
            feature_name = features
            feature_importance.append(pd.DataFrame({'feature_name':feature_name,'importance':importance}))

        feature_importance_df = pd.concat(feature_importance)
        feature_importance_df = feature_importance_df.groupby(['feature_name']).mean().reset_index()
        feature_importance_df = feature_importance_df.sort_values(by=['importance'],ascending=False)
        feature_importance_df.to_csv(output_path + '/feature_importance.csv',index=False)

        mean_valid_metric = np.mean(all_valid_metric)
        mean_valid_rmse = np.mean(all_valid_rmse)
        Write_log(log,'all valid mean metric:%.6f'%(mean_valid_metric))

        oof.to_csv(output_path + '/oof.csv',index=False)
        if test is None:
            log.close()
            os.rename(output_path + '/train.log', output_path + '/train_%.6f.log'%mean_valid_metric)

        log_df = pd.DataFrame({'run_id':[run_id],'metric':[f'{mean_valid_rmse},{mean_valid_metric}'],'lb':[np.nan],'remark':[remark]})
        if not os.path.exists(output_root + '/experiment_log.csv'):
            log_df.to_csv(output_root + '/experiment_log.csv',index=False)
        else:
            log_df.to_csv(output_root + '/experiment_log.csv',index=False,mode='a',header=None)
    if test is not None:
        sub = test[[id_name,'source']]
        sub[label_name] = 0
        for fold in range(folds):
            with open(output_path + '/fold%s.ckpt'%fold,'rb') as f:
                model = pickle.load(f)
                f.close()
            test_preds = model.predict(test[features].values)
            Write_log(log,'fold%s test pred mean:%.6f'%(fold,np.mean(test_preds)))
            sub[label_name] += (test_preds / folds)
        Write_log(log,'all test pred mean:%.6f'%(np.mean(sub[label_name])))
        sub[[id_name,'source',label_name]].to_csv(output_path + '/submission.csv',index=False)
        sub.loc[sub['source']=='ori',['id','delta_g']].to_csv(output_path + 'sub.csv',index=False)
        if train is not None:
            os.rename(output_path + 'train.log', output_path + 'train_%.6f.log'%mean_valid_metric)
    if 'rf_tmp1' in output_path:
        os.rename(output_path,output_root+run_id+'/')
    return oof,sub

def Xgb_train_and_predict(train, test, features, config, output_root='./output/', run_id=None):
    if not run_id:
        run_id = 'run_xgb_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        while os.path.exists(output_root+run_id+'/'):
            time.sleep(1)
            run_id = 'run_xgb_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_root + 'xgb_tmp1/'
        if not os.path.exists(output_path):
            os.mkdir(output_path)
    else:
        output_path = output_root + run_id + '/'

    oof,sub = None ,None
    if train is not None:
        os.system(f'cp ./*.py {output_path}')
        os.system(f'cp ./*.sh {output_path}')

        oof = train[[id_name]]
        oof['fold'] = -1

        if isinstance(label_name,list):
            for l in label_name:
                oof[l] = 0.0
        else:
            oof[label_name] = 0.0
    else:
        oof = None

    if train is not None:
        log = open(output_path + 'train.log','w',buffering=1)
        log.write(str(config)+'\n')
        log.write(str(features)+'\n')
        params = config['xgb_params']
        rounds = config['rounds']
        verbose = config['verbose_eval']
        early_stopping_rounds = config['early_stopping_rounds']
        folds = config['folds']
        seed = config['seed']
        remark = config['remark']

        all_valid_metric = []
        all_valid_rmse = []
        feature_importance = []
        # kf = GroupKFold(n_splits=folds)
        # for fold, (trn_index, val_index) in enumerate(kf.split(train,train[label_name],train['cluster'])):
        kf = KFold(n_splits = folds,shuffle=True,random_state=seed)
        for fold, (trn_index, val_index) in enumerate(kf.split(train,train[label_name])):
        # for fold in range(1,11):
        #     trn_index = train.loc[train['sampling_fold']!=fold].index
        #     val_index = train.loc[train['sampling_fold']==fold].index
            evals_result_dic = {}
            # trn_index = train.loc[(train.index.isin(trn_index))&(train['source']=='ori')].index
            # bins = np.arange(0,3/0.3+1) * 0.3
            # trn_sim = train.loc[trn_index,'seq_sim_max'].values
            # val_sim = train.loc[val_index,'seq_sim_max'].values
            # trn_bin_count = np.histogram(trn_sim.reshape(-1),bins=bins)[0].astype(float)
            # val_bin_count = np.histogram(val_sim.reshape(-1),bins=bins)[0].astype(float)
            # tes_bin_count = np.histogram(test['seq_sim_max'].values.reshape(-1),bins=bins)[0].astype(float)
            # # print(trn_bin_count,val_bin_count)
            # trn_weight = []
            # val_weight = []
            # for sim in trn_sim:
            #     bc = np.histogram([sim],bins=bins)[0]
            #     idx = np.where(bc==1)[0][0]
            #     trn_weight.append(tes_bin_count[idx]/trn_bin_count[idx])
            # for sim in val_sim:
            #     bc = np.histogram([sim],bins=bins)[0]
            #     idx = np.where(bc==1)[0][0]
            #     val_weight.append(tes_bin_count[idx]/val_bin_count[idx])
            trn_data = xgb.DMatrix(train.loc[trn_index,features].values, label=train.loc[trn_index,label_name].values)#,weight=train.loc[trn_index,'weight']
            val_data = xgb.DMatrix(train.loc[val_index,features].values, label=train.loc[val_index,label_name].values)#,weight=train.loc[val_index,'weight_val']
            watchlist = [(trn_data, 'train'), (val_data, 'valid')]
            model = xgb.train(params,
                dtrain  = trn_data,
                num_boost_round  = rounds,
                evals = watchlist,
                evals_result = evals_result_dic,
                early_stopping_rounds = early_stopping_rounds,
                verbose_eval = verbose
            )
            model.save_model(output_path + '/fold%s.ckpt'%fold)
            val_data = xgb.DMatrix(train.loc[val_index,features].values)
            valid_preds = model.predict(val_data,iteration_range=(0, model.best_iteration))
            oof.loc[val_index,label_name] = valid_preds
            # ori_index = train.loc[val_index,'source']=='ext'
            # print('ori metric:',Metric(train.loc[val_index,label_name].values[ori_index],valid_preds[ori_index]))
            for i in range(len(evals_result_dic['valid'][params['metric']])//verbose):
                Write_log(log,' - %i round - train_metric: %.6f - valid_metric: %.6f\n'%(i*verbose,evals_result_dic['train'][params['metric']][i*verbose],evals_result_dic['valid'][params['metric']][i*verbose]))
            all_valid_rmse.append(Metric(train.loc[val_index,label_name].values,valid_preds)[0])
            all_valid_metric.append(Metric(train.loc[val_index,label_name].values,valid_preds)[-1])
            Write_log(log,'- fold%s valid metric: %.6f, valid_mean:%.6f\n'%(fold,all_valid_metric[-1],np.mean(valid_preds)))

            importance = model.get_score(importance_type='gain')
            importance_gain = []
            for i in range(len(features)):
                if f'f{i}' in importance:
                    importance_gain.append(importance[f'f{i}'])
                else:
                    importance_gain.append(0.0)
            feature_name = features
            feature_importance.append(pd.DataFrame({'feature_name':feature_name,'importance':importance_gain}))

        feature_importance_df = pd.concat(feature_importance)
        feature_importance_df = feature_importance_df.groupby(['feature_name']).mean().reset_index()
        feature_importance_df = feature_importance_df.sort_values(by=['importance'],ascending=False)
        feature_importance_df.to_csv(output_path + '/feature_importance.csv',index=False)

        mean_valid_metric = np.mean(all_valid_metric)
        mean_valid_rmse = np.mean(all_valid_rmse)
        Write_log(log,'all valid mean metric:%.6f'%(mean_valid_metric))

        oof.to_csv(output_path + '/oof.csv',index=False)
        if test is None:
            log.close()
            os.rename(output_path + '/train.log', output_path + '/train_%.6f.log'%mean_valid_metric)

        log_df = pd.DataFrame({'run_id':[run_id],'metric':[f'{mean_valid_rmse},{mean_valid_metric}'],'lb':[np.nan],'remark':[remark]})
        if not os.path.exists(output_root + '/experiment_log.csv'):
            log_df.to_csv(output_root + '/experiment_log.csv',index=False)
        else:
            log_df.to_csv(output_root + '/experiment_log.csv',index=False,mode='a',header=None)
    if test is not None:
        sub = test[[id_name,'source']]
        sub[label_name] = 0
        for fold in range(folds):
            model = xgb.Booster(model_file=output_path + '/fold%s.ckpt'%fold)
            test_data = xgb.DMatrix(test[features].values)
            test_preds = model.predict(test_data,iteration_range=(0, model.best_iteration))
            Write_log(log,'fold%s test pred mean:%.6f'%(fold,np.mean(test_preds)))
            sub[label_name] += (test_preds / folds)
        Write_log(log,'all test pred mean:%.6f'%(np.mean(sub[label_name])))
        sub[[id_name,'source',label_name]].to_csv(output_path + '/submission.csv',index=False)
        sub.loc[sub['source']=='ori',['id','delta_g']].to_csv(output_path + 'sub.csv',index=False)
        if train is not None:
            os.rename(output_path + 'train.log', output_path + 'train_%.6f.log'%mean_valid_metric)
    if 'xgb_tmp1' in output_path:
        os.rename(output_path,output_root+run_id+'/')
    return oof,sub

def Lgb_train_and_predict(train, test, feature_map_dic, features, config, output_root='./output/', run_id=None):
    if not run_id:
        run_id = 'run_lgb_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        while os.path.exists(output_root+run_id+'/'):
            time.sleep(1)
            run_id = 'run_lgb_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_root + 'lgb_tmp1/'

    else:
        output_path = output_root + run_id + '/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    oof,sub = None ,None
    if train is not None:
        os.system(f'cp ./*.py {output_path}')
        os.system(f'cp ./*.sh {output_path}')

        oof = train[[id_name]]
        oof['fold'] = -1

        if isinstance(label_name,list):
            for l in label_name:
                oof[l] = 0.0
        else:
            oof[label_name] = 0.0
    else:
        oof = None

    if train is not None:
        log = open(output_path + 'train.log','w',buffering=1)
        log.write(str(config)+'\n')
        log.write(str(features)+'\n')
        params = config['lgb_params']
        rounds = config['rounds']
        verbose = config['verbose_eval']
        early_stopping_rounds = config['early_stopping_rounds']
        folds = config['folds']
        seed = config['seed']
        remark = config['remark']

        all_valid_metric = []
        all_valid_rmse = []
        feature_importance = []
        kf = GroupKFold(n_splits = folds)
        # kf = KFold(n_splits = folds,shuffle=True,random_state=seed)
        for fold,(trn_index,val_index) in enumerate(kf.split(train,train[label_name],train['number'])):
        # for fold,(trn_index,val_index) in enumerate(kf.split(train,train[label_name])):
            print(val_index)

            evals_result_dic = {}

            trn_data = lgb.Dataset(train.loc[trn_index,features], label=train.loc[trn_index,label_name])#,weight=train.loc[trn_index,'weight']
            val_data = lgb.Dataset(train.loc[val_index,features], label=train.loc[val_index,label_name])#,weight=train.loc[val_index,'weight_val']
            model = lgb.train(params,
                train_set  = trn_data,
                num_boost_round  = rounds,
                valid_sets = [trn_data,val_data],
                evals_result = evals_result_dic,
                early_stopping_rounds = early_stopping_rounds,
                verbose_eval = verbose
            )


            valid_preds = model.predict(train.loc[val_index,features], num_iteration=model.best_iteration)
            oof.loc[val_index,label_name] = valid_preds
            for i in range(len(evals_result_dic['valid_1'][params['metric']])//verbose):
                Write_log(log,' - %i round - train_metric: %.6f - valid_metric: %.6f\n'%(i*verbose,evals_result_dic['training'][params['metric']][i*verbose],evals_result_dic['valid_1'][params['metric']][i*verbose]))
            all_valid_rmse.append(Metric(train.loc[val_index,label_name].values,valid_preds)[0])
            all_valid_metric.append(Metric(train.loc[val_index,label_name].values,valid_preds)[-1])
            Write_log(log,'- fold%s valid metric: %.6f, valid_mean:%.6f\n'%(fold,all_valid_metric[-1],np.mean(valid_preds)))

            if all_valid_metric[-1] > 0.1:
                model.save_model(output_path + '/fold%s.ckpt'%fold)
            else:
                model.save_model(output_path + '/fold%s_.ckpt'%fold)
            importance_gain = model.feature_importance(importance_type='gain')
            importance_split = model.feature_importance(importance_type='split')
            feature_name = model.feature_name()
            feature_name = [feature_map_dic[f] for f in feature_name]
            feature_importance.append(pd.DataFrame({'feature_name':feature_name,'importance_gain':importance_gain,'importance_split':importance_split}))

        feature_importance_df = pd.concat(feature_importance)
        feature_importance_df = feature_importance_df.groupby(['feature_name']).mean().reset_index()
        feature_importance_df = feature_importance_df.sort_values(by=['importance_gain'],ascending=False)
        feature_importance_df.to_csv(output_path + '/feature_importance.csv',index=False)

        mean_valid_metric = np.mean(all_valid_metric)
        mean_valid_rmse = np.mean(all_valid_rmse)
        best_valid_rmse,best_valid_pearson = Metric(train[label_name].values,oof[label_name].values)
        Write_log(log,'all valid best metric:%.6f'%(best_valid_pearson))

        oof.to_csv(output_path + '/oof.csv',index=False)
        if test is None:
            log.close()
            os.rename(output_path + '/train.log', output_path + '/train_%.6f.log'%best_valid_pearson)

        log_df = pd.DataFrame({'run_id':[run_id],'metric':[f'{best_valid_rmse},{best_valid_pearson}'],'lb':[np.nan],'remark':[remark]})
        if not os.path.exists(output_root + '/experiment_log.csv'):
            log_df.to_csv(output_root + '/experiment_log.csv',index=False)
        else:
            log_df.to_csv(output_root + '/experiment_log.csv',index=False,mode='a',header=None)
    if test is not None:
        sub = test[[id_name]]
        sub[label_name] = 0
        for fold in range(folds):
            features = all_features[fold]
            model = lgb.Booster(model_file=output_path + '/fold%s.ckpt'%fold)
            test_preds = model.predict(test[features], num_iteration=model.best_iteration)
            Write_log(log,'fold%s test pred mean:%.6f'%(fold,np.mean(test_preds)))
            sub[label_name] += (test_preds / folds)
        Write_log(log,'all test pred mean:%.6f'%(np.mean(sub[label_name])))
        sub[[id_name,label_name]].to_csv(output_path + '/submission.csv',index=False)
        sub[['id','delta_g']].to_csv(output_path + 'sub.csv',index=False)
        if train is not None:
            os.rename(output_path + 'train.log', output_path + 'train_%.6f.log'%mean_valid_metric)
    if 'lgb_tmp1' in output_path:
        os.rename(output_path,output_root+run_id+'/')
    return oof,sub
