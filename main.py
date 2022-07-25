import numpy as np
import pandas as pd
from Regression import *
from DyGreyAssociation import *
from Metrics import Evaluate


input_file = r'Data/DREAM4/size10/insilico_size10_1_timeseries.tsv'
label_file = r'Data/DREAM4/size10/DREAM4_GoldStandard_InSilico_Size10_1.tsv'

data = pd.read_csv(input_file, sep='\t').iloc[:, 1:]
data_val = data.values
col= data.columns.values

gea = ASW_GRA(data=data_val, columns=col, iteration=0)
index, value = gea.ADAPATIVE_GRA()

score = GRA_Series(index, value, col)

model = Regression(data)
y_prob = model.fit()
score = normalize(score, 0.4)

y_pred = y_prob * score

AUC, AUPR= Evaluate(y_prob * score, label_file, col)
print(AUC,AUPR)










