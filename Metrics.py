from sklearn.metrics import roc_curve,roc_auc_score,auc,precision_recall_curve,f1_score,precision_score,matthews_corrcoef,recall_score,accuracy_score
import pandas as pd
import numpy as np

def regu_tar(columns):
    index = []
    for regu in columns:
        for tar in columns:
            ind = '(' + regu +',' + tar +')'
            index.append(ind)
    blank = pd.Series(np.zeros(len(index)),index=index)
    return blank

def label_generate(columns,label_data):
    if not isinstance(label_data,np.ndarray):
        raise TypeError(r'label_data must be numpy.ndarray')

    index = []
    for lab in label_data:
        if lab[2] == 1:
            ind = '(' + lab[0] +',' + lab[1] +')'
            index.append(ind)

    label = regu_tar(columns)

    label[index] = 1

    return label




def Evaluate(y_prob,label_file,columns):
    label_data = pd.read_csv(label_file,sep='\t',header=None).values

    y_real = label_generate(columns,label_data)

    if len(y_prob) != len(y_real):
        blank = regu_tar(columns)
        blank[y_prob.index.values] = y_prob.values
        y_prob = blank

    AUC = roc_auc_score(y_real,y_prob)

    pre,rec,thre__ = precision_recall_curve(y_real,y_prob)
    AUPR = auc(rec,pre)



    return AUC,AUPR
