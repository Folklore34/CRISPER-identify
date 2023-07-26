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

# minimum redundancy maximum relevance (mrmr)
from mrmr import mrmr_classif
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
parse.add_argument("--pos",help="Positive datasets in CSV format",required=True)
parse.add_argument("--neg",help="Negative datasets in CSV format",required=True)
parse.add_argument("--outdir",help="output dir",required=False,default='.')
parse.add_argument("--sample",help="sample",required=False,default='out')
parse.add_argument("--feature",help="feature name: SSA, PSSM_AC, RPSSM, ESM",required=True)
parse.add_argument("--mrmrK",help="200",required=False,default=200,type=int)

args  = parse.parse_args()

###Construct the model
def individual_model(x_train,y_train):
    max_par=int(max(x_train.columns.size/2,sqrt(x_train.columns.size)))
    model_list=[]
    #SVM
    par=[2**i for i in range(-6,7)]
    param_grid_sv=[{'kernel':['rbf'],'gamma':par,'C':par},{'kernel':['linear'],'C':par}]
    svm_clf=GridSearchCV(SVC(probability=True),param_grid_sv,cv=10,n_jobs=-1).fit(x_train,y_train)
    svm=svm_clf.best_estimator_.fit(x_train,y_train)
    model_list.append(svm)
    #KNN
    param_grid_knn={'n_neighbors':range(1,max_par)}
    knn_clf=GridSearchCV(KNeighborsClassifier(),param_grid_knn,cv=10,n_jobs=-1).fit(x_train,y_train)
    knn=knn_clf.best_estimator_.fit(x_train,y_train)
    model_list.append(knn)
    #RF
    param_grid_rf={'n_estimators':range(1,max_par),'max_features':range(1,20,5)}
    rf_clf=GridSearchCV(RandomForestClassifier(),param_grid_rf,cv=10,n_jobs=-1).fit(x_train,y_train)
    rf=rf_clf.best_estimator_.fit(x_train,y_train)
    model_list.append(rf)
    #MLP
    mlp=MLPClassifier(hidden_layer_sizes=[64,32],max_iter=1000).fit(x_train,y_train)
    model_list.append(mlp)
    #LR
    lr=LogisticRegression().fit(x_train,y_train)
    model_list.append(lr)
    #XGB
    XGB=XGBClassifier(learning_rate=0.1,eval_metric=['logloss','auc','error'],use_label_encoder=False,objective="binary:logistic").fit(x_train,y_train)
    model_list.append(XGB)
    #Catboost
    cat=CatBoostClassifier(verbose=1000).fit(x_train,y_train)
    model_list.append(cat)
    #Light
    light=LGBMClassifier().fit(x_train,y_train)
    model_list.append(light)
    return model_list

###performance assessment
# Accuracy (ACC), Sensitivity (Sn), Specificity (Sp), Matthews correlation coefficient (MCC), Area under receiver operating characteristic curves (AUC)


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
    SPE = tn / (tn + fp)
    per_list = [PRE,REC,SPE,F1_score,ACC,MCC,auc(fpr_v,tpr_v)]
    return per_list

#### ROC curve of 5-fold cross validation
def ROC_5_fold(proba_valid_y,validation_y,fig_path):
    plt.figure(figsize=(10,8))
    i = 0
    tprs,aucs=[],[]
    for proba_y, verified_y in zip(proba_valid_y, validation_y):
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(list(verified_y), list(proba_y))
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0]= 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                label='ROC fold %d (AUC = %0.3f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
            label='Mean ROC (AUC = %0.3f $\\pm$ %0.3f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3,
                    label='$\\pm$ 1 std. dev.')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0],labels=['0.0','0.2','0.4','0.6','0.8','1.0'])
    plt.xticks([0.0,0.2,0.4,0.6,0.8,1.0],labels=['0.0','0.2','0.4','0.6','0.8','1.0'])
    plt.ylabel('True Positive Rate',fontsize=16)
    plt.xlabel('False Positive Rate',fontsize=16)
    plt.legend(loc="lower right",fontsize=15)
    plt.savefig(fig_path)
    plt.close()

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

# Load  dataframe 
pos = pd.read_csv(args.pos)
neg = pd.read_csv(args.neg)

# Seperate into train and test set
random.seed(1)
# train:test = 7:3
len_pos = len(pos)
len_neg = len(neg)

len_pos_07 = int(len_pos * 0.7)
len_pos_03 = len_pos - len_pos_07

# Training sets positive sample: 70 %
train_row_pos=random.sample(list(range(0,len_pos)),len_pos_07)

# Test sets positive sample: 30 %
test_row_pos = list(set(range(0,len_pos)).difference(train_row_pos))

# Training sets negative sample: 70 %
random.seed(1)
train_row_neg=random.sample(list(range(0,len_neg)),len_pos_07)

# Test sets negative sample : 30 %
test_row_neg = list(set(range(0,len_neg)).difference(train_row_neg)) 

random.seed(1)
test_row_neg=random.sample(test_row_neg,len_pos_03)

train_data = pd.concat([pos.iloc[train_row_pos,:],neg.iloc[train_row_neg,:]])
test_data = pd.concat([pos.iloc[test_row_pos,:],neg.iloc[test_row_neg,:]])

if args.outdir != '.':
    os.chdir(args.outdir)
sample = args.sample
if not os.path.exists(sample+'/results'):
    os.makedirs(sample+'/results')
    os.makedirs(sample+'/model')
os.chdir(sample)
train_data.to_csv('results/train_data.csv')
test_data.to_csv('results/test_data.csv')

# Train
y_verified_valid_all,y_pred_valid_all,y_proba_test_all=[],[],[]
     
kf=StratifiedKFold(n_splits=5) ###5-fold-cross-validation
train_features,train_label = train_data.iloc[:,1:],train_data.iloc[:,0] 

feature_name = args.feature
split_c = -1

for train_row, valid_row in kf.split(train_features,train_label):
    split_c = split_c + 1

    #for i in range(0,len(train_data)): #[PSSM_AC,RPSSM,SSA]
    Y_train,Y_valid=train_data.iloc[train_row,:].iloc[:,0],train_data.iloc[valid_row,:].iloc[:,0]
  
    X_train = train_data.iloc[train_row,:].iloc[:,1:]
    X_train = pd.DataFrame(preprocessing.minmax_scale(X_train),columns=X_train.columns)

    # mrmr_feature_selection
    mrmrK = args.mrmrK
    if (X_train.columns.size > mrmrK) and (mrmrK != 0):
        mrmr_y = pd.Series(list(Y_train))
        X_train =X_train[mrmr_classif(X = X_train, y = mrmr_y, K = mrmrK)]

    X_valid=pd.DataFrame(preprocessing.minmax_scale(train_data.iloc[valid_row,:][X_train.columns]),columns=X_train.columns)
    X_test=pd.DataFrame(preprocessing.minmax_scale(test_data[X_train.columns]),columns=X_train.columns)

    model_name_list = ['SVM','KNN','RF','MLP','LR','XGB','Light']
    model_list = individual_model(X_train,Y_train)

    ensemble_clf = StackingClassifier(estimators=list(zip(model_name_list,model_list)),final_estimator=LogisticRegression()).fit(X_train,Y_train)

    with open('model/model_'+feature_name+'_'+str(split_c)+'.model','wb+') as f:
        f.write(pickle.dumps(ensemble_clf))

    y_pred_valid_all.append(ensemble_clf.predict_proba(X_valid)[:,1])
    y_proba_test_all.append(ensemble_clf.predict_proba(X_test)[:,1])
    y_verified_valid_all.append(Y_valid)
    
test_pred_score=np.mean(y_proba_test_all,axis=0)#the predicted probability of the test data

mean_fpr=np.linspace(0,1,100)
per,tprs=[],[]

for y_verified,y_pred in zip(y_verified_valid_all,y_pred_valid_all):
    fpr,tpr,threshols=roc_curve(list(y_verified),list(y_pred))
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    per.append(performance(y_verified,y_pred))

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
mean_per=np.mean(per,axis=0)
mean_per[-1]=mean_auc

metrics_cols = ['PRE','REC','SPE','F1_score','ACC','MCC','AUC']
validation_performance=pd.DataFrame(per,columns=metrics_cols)
validation_performance.loc[5]=list(mean_per)
validation_performance.loc[6]=list(np.std(per,axis=0))
validation_performance.insert(0,'Category',['fold1','fold2','fold3','fold4','fold5','mean','std'])

os.chdir('results')

validation_performance.to_csv(feature_name+'_Validation_Performance.csv',index=None)

ROC_5_fold(y_pred_valid_all,y_verified_valid_all,feature_name+'_ROC_5_fold.jpg')

## Performance of the testing dataset
pred_test=pd.DataFrame({'Predict score':test_pred_score,'Verified':test_data.iloc[:,0]}).reset_index(drop=True)
pred_test.to_csv(feature_name+'_Test_Pred_Score.csv',index=None)

test_performance=performance(test_data.iloc[:,0],test_pred_score)

test_metrics = {}
for i,v in enumerate(metrics_cols):
    test_metrics[v] = test_performance[i]

test_performance = pd.DataFrame.from_dict(test_metrics,orient='index').T
test_performance.to_csv(feature_name+'_Test_Performance.csv',index=None)

##ROC 
auc_pred(list(test_pred_score),list(test_data.iloc[:,0]),feature_name+'_Test_ROC.jpg')

###PR curve
pr_curve(list(test_pred_score),list(test_data.iloc[:,0]),feature_name+'_Test_PR_curve.jpg')



