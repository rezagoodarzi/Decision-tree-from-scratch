import pandas as pd
import numpy as np

data = pd.read_csv("satisfied.csv")
data2 = pd.read_csv("dissatisfied.csv")


test1 = data[:15000]
test2 = data2[:15000]

result = pd.concat([test1, test2])

train1 = data.iloc[15000:16000]
train2 = data2.iloc[15000:16000]

train = pd.concat([train1, train2])

result.to_csv("test.csv", index=False)
train.to_csv("train.csv", index=False)

