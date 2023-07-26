# -*- coding: utf-8 -*-
'''
#-------------------------------------------------
#          File Name:           *.py
#          Description:   *
#          Date:                 2023.06.01
#-------------------------------------------------
'''

import os
import sys
#import time
import random
import pickle
#import logging
import argparse
import textwrap
import numpy as np
import pandas as pd
from math import sqrt
from tqdm import tqdm, trange
from itertools import combinations
import matplotlib.pylab as plt
import multiprocessing as mp
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,precision_recall_curve,roc_curve,auc,matthews_corrcoef
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,StackingClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,GridSearchCV

description = '''
------------------------------------------------------------------------------------------------------------------------
This script is designed to
Usage:
python *.py
--*: s* [required]
------------------------------------------------------------------------------------------------------------------------
'''
parse = argparse.ArgumentParser(prog='PROG', formatter_class=argparse.RawDescriptionHelpFormatter, description=textwrap.dedent(description))
parse.add_argument("--csv",help="Test dataset in CSV format",required=True)
#parse.add_argument("--",help="Negative datasets in CSV format",required=True)
parse.add_argument("--outdir",help="output dir",required=False,default='.')
parse.add_argument("--sample",help="sample",required=False,default='out')
parse.add_argument("--feature",help="feature name: SSA, PSSM_AC, RPSSM, ESM",required=True)
parse.add_argument("--model",help="model dir path",required=True)

args  = parse.parse_args()

###performance assessment

def performance(y_verified_list,y_pred_list):
    fpr_v,tpr_v,threshols_v = roc_curve(list(y_verified_list),list(y_pred_list))

    pred_y_array = np.where(np.array(y_pred_list)>=0.5,1,np.array(y_pred_list))
    pred_y_array = np.where(pred_y_array<0.5,0,pred_y_array)
    #SN,SP,PRE,ACC,F-Score,MCC
    REC = recall_score(y_verified_list,pred_y_array)
    PRE = precision_score(y_verified_list,pred_y_array)
    ACC = accuracy_score(y_verified_list, pred_y_array)
    F1_score = f1_score(y_verified_list,pred_y_array)
    MCC = matthews_corrcoef(y_verified_list, pred_y_array)
    tn, fp, fn, tp = confusion_matrix(y_verified_list, pred_y_array).ravel()
    SPE = tn / (tn+fp)
    per_list = [PRE,REC,SPE,F1_score,ACC,MCC,auc(fpr_v,tpr_v)]
    return per_list

def auc_pred(test_pred_score_list,test_verified_list,fig_path):
    fpr,tpr,threshold = roc_curve(test_verified_list, test_pred_score_list) ###
    roc_auc = auc(fpr,tpr) ###
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='b',lw=lw, label='ROC curve (area = %0.3f)' % roc_auc) ###
    plt.plot([0, 1], [0, 1], color='r', lw=lw, linestyle='--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],labels=['0.0','0.2','0.4','0.6','0.8','1.0'])
    plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],labels=['0.0','0.2','0.4','0.6','0.8','1.0'])
    plt.ylabel('True Positive Rate',fontsize=16)
    plt.xlabel('False Positive Rate',fontsize=16)
    plt.legend(loc="lower right",fontsize=15)
    #plt.show()
    plt.savefig(fig_path)
    plt.close()

def pr_curve(list_test_pred_score,list_test_verified,fig_path):
    precision,recall,thresholds=precision_recall_curve(list_test_verified,list_test_pred_score)
    plt.figure(figsize=(10,10))
    plt.plot(recall,precision)
    plt.rc('legend',fontsize=16)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall',fontsize=16)
    plt.ylabel('Precision',fontsize=16)
    plt.title('Precision/Recall Curve of {}'.format('PreAcrs'),fontsize=18)
    #plt.show()
    plt.savefig(fig_path)
    plt.close()

# Load features dataframe 
df = pd.read_csv(args.csv)
pos = df[df['Class'] == 1]
neg = df[df['Class'] == 0]

random.seed(1)
len_pos = len(pos)
len_neg = len(neg)

if len(neg) > len(pos):
    test_row_neg = random.sample(list(range(0,len_neg)),len_pos)
    test_data = pd.concat([pos,neg.iloc[test_row_neg,:]])
else:
    test_row_pos = random.sample(list(range(0,len_pos)),len_neg)
    test_data = pd.concat([pos.iloc[test_row_pos,:],neg])
   	
model_dir = os.path.abspath(args.model)

if args.outdir != '.':
    os.chdir(args.outdir)
sample = args.sample

if not os.path.exists(sample+'/results'):
    os.makedirs(sample+'/results')
    #os.makedirs(sample+'/model')
os.chdir(sample)

test_feature=test_data.iloc[:,1:]
test_lable=test_data.iloc[:,0]

feature=args.feature

y_proba_all = []

for i in range(5):
    f = open(os.path.join(model_dir,'model_'+feature+'_'+str(i)+'.model'),'rb')
    model = pickle.loads(f.read())
    f.close()
    X = test_feature[list(model.feature_names_in_)]
    X = pd.DataFrame(preprocessing.minmax_scale(X),columns=X.columns)
    y_proba = model.predict_proba(X)[:,1]
    y_proba_all.append(y_proba)
    
pred_score = np.mean(y_proba_all,axis=0)
pred_df = pd.DataFrame({'Predict score':pred_score,'Verified':test_lable})

os.chdir('results')
test_data.to_csv('Test_data.csv')

pred_df.to_csv(feature+'_Pred_Score.csv',index=None)

metrics_cols = ['PRE','REC','SPE','F1_score','ACC','MCC','AUC']

data_performance=performance(test_lable,pred_score)

data_metrics = {}
for i,v in enumerate(metrics_cols):
    data_metrics[v] = data_performance[i]
    
#Compute and plot the results
data_performance = pd.DataFrame.from_dict(data_metrics,orient='index').T
data_performance.to_csv(feature+'_Test_Performance.csv',index=None)
roc_curve(list(test_lable),list(pred_score))
auc_pred(list(pred_score),list(test_lable),feature+'_ROC.jpg')
pr_curve(list(pred_score),list(test_lable),feature+'_PR_curve.jpg')




