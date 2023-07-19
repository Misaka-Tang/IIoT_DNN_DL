import clkmean
import evaluation
import numpy as np
import pandas as pd

# Load data
df = pd.read_csv('datas/dataKmeanSampleOversampling.csv')

# split feature and label
X = df.drop(['Attack_type'], axis=1)
y = df['Attack_type']

clk = clkmean.ClKmeans(3)
clk.fit(X)
y_pred = clk.predict(X)

eval = evaluation.Evaluation()
eval.evalution(y, y_pred)