import numpy as np
import pandas as pd
from Evaluation import *

class ASW_GRA:
    def __init__(self,data,columns,iteration=0,numpoint=21,max_win=15,win_len=None):

        self.data = data[iteration * numpoint:iteration * numpoint + numpoint, :]
        self.columns = columns
        self.win_len = win_len
        self.max_win= max_win

        self.m, self.n = self.data.shape

        if self.win_len == None:
            self.win_len = int(0.66*numpoint)

    def data_process(self,ind):

        data = self.data.copy()
        col  =  self.columns.copy()

        data_y = data[:, ind].reshape([-1,1])
        data_x = np.delete(data, ind, axis=1)

        data = np.hstack([data_x,data_y])

        re_col = col[ind]
        col = np.delete(col,ind)
        col = np.append(col,re_col)

        return data,col


    def __bilateral_softmax(self,seq):

        seq = abs(seq)

        seq = np.exp(seq)

        p_val = seq/seq.sum()
        entropy = np.sum(-p_val*np.log2(p_val))

        return entropy

    def adaptive_sliding_window(self,data):


        data_win = data[:self.win_len,-1]  # Targrt squence is placed in the last column.

        data_diff = np.diff(data_win)

        p_val = self.__bilateral_softmax(data_diff)
        H0 = np.sum(-p_val*np.log2(p_val))
        win_list = []
        H = []

        for i in np.arange(0,self.m):
            if i == 0:
                win_list.append(self.win_len)
                H.append(H0)
            else :
                win_diff = np.diff(data[i:int(i+win_list[-1]),-1])
                entro = self.__bilateral_softmax(win_diff)

                win_coff = entro / H[-1]
                H.append(entro)

                length = win_list[-1] / win_coff
                length = int(length)


                if length > self.max_win:
                    length = self.max_win

                elif length < self.win_len:
                    length = self.win_len

                elif i  + length - 1 > self.m:
                    length = self.m - i + 1

                win_list.append(length)

                if i  +length+ 1>self.m:
                    length = int(self.m - i + 1)
                    win_list.append(length)
                    break

        return win_list

    def gra_cal(self,co_seq,re_seq,rho=0.5):
        m1,n1 = co_seq.shape

        re_seq = np.array(re_seq).reshape([-1,1])
        coff = abs(co_seq - re_seq)

        max_value = coff.max()
        min_value = coff.min()

        result = np.zeros([m1, n1])

        for i in range(n1):
            for j in range(m1):
                result[j, i] = (min_value + rho * max_value) / (coff[j, i] + rho * max_value)
        result = result.mean(axis=0)

        return result

    def GRA(self,data1,win_list, columns):

        data1 = np.array(data1)
        columns = np.array(columns)

        m, n = data1.shape
        data_trans = np.zeros([m, n])
        for i in range(n):
            MEAN = data1[:, i].mean()
            MAX = data1[:, i].max()
            MIN = data1[:, i].min()
            data_trans[:, i] = (data1[:, i] - MEAN) / (MAX - MIN)

        result = np.zeros([len(win_list),n-1])

        for i,win in  enumerate(win_list):
            re_seq = data1[i:i+win,-1]
            co_seq = data1[i:i+win,:-1]
            result_win = self.gra_cal(co_seq,re_seq)
            result[i,:] = result_win.reshape([1,-1])


        result =result.mean(axis=0)

        result = np.append(result,0)

        reslult = pd.Series(result,index=columns)

        result = reslult.sort_values(ascending=False)

        return result.index.values,result.values.tolist()

    def ADAPATIVE_GRA(self):

        index = []
        value = np.zeros([self.n,self.n])

        for i in range(self.n):
            data,col = self.data_process(ind=i)
            win_list = self.adaptive_sliding_window(data)
            ind,val = self.GRA(data,win_list,col)
            index.append(ind)
            value[i,:] = val

        index = np.array(index)
        return index,value


def regu_tar(columns):
    index = []
    for regu in columns:
        for tar in columns:
            ind = '(' + regu +',' + tar +')'
            index.append(ind)
    blank = pd.Series(np.zeros(len(index)),index=index)
    return blank

def GRA_Series(index,val,columns):
    index_list = []
    for ind1 in index:
        for ind2 in ind1:
            idx = '('+ ind2 + ',' + ind1[-1] + ')'
            index_list.append(idx)
    val = val.flatten()
    gra_series = regu_tar(columns)
    gra_series[index_list] = val
    return gra_series

def normalize(gra,thre):

    for ind in gra.index.values:
        if gra[ind] >= thre:
            gra[ind] = 1
        else:
            gra[ind] = 0

    return gra


