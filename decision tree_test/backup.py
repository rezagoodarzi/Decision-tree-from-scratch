import pandas as pd

file = pd.read_csv("Airplane_clean_split.csv")

data = file[file['satisfaction'] == 'neutral or dissatisfied']
data2 = file[file['satisfaction'] == 'satisfied']

test1 = data[:15000]
test2 = data2[:15000]

result = pd.concat([test1, test2])

train1 = data.iloc[15000:16000]
train2 = data2.iloc[15000:16000]

train = pd.concat([train1, train2])

result.to_csv("train.csv", index=False)
train.to_csv("test.csv", index=False)
