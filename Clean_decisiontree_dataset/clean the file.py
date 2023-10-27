import pandas as pd

data = pd.read_csv("Airplane_clean_split.csv")

dissatisfied = data[data['satisfaction'] == 'neutral or dissatisfied']
satisfied = data[data['satisfaction'] == 'satisfied']

satisfied.to_csv("satisfied.csv",index=False)
dissatisfied.to_csv("dissatisfied.csv",index=False)

