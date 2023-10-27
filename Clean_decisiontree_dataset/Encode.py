import pandas as pd

data = pd.read_csv("Airplane.csv")
data = data.dropna()
data = data.drop_duplicates()

data = data[data['Type of Travel'] != 'Busi ness travel']
data = data[data['Gate location'] != 0]
data = data[data['Seat comfort'] != 0]
data = data[data['On-board service'] != 0]
data = data[data['Inflight entertainment'] != 0]
data = data[data['Leg room service'] != 0]

data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})
data['Gender'] = data['Gender'].astype(int)

data['Age'] = data['Age'].astype(int)

data['Flight Distance'] = data['Flight Distance'].astype(int)

data['Class'] = data['Class'].map({'Eco': 0, 'Business': 1, 'Eco Plus': 2})
data['Class'] = data['Class'].astype(int)

data['Departure Delay in Minutes'] = data['Departure Delay in Minutes'].astype(int)
data['Arrival Delay in Minutes'] = data['Arrival Delay in Minutes'].astype(int)

data['Customer Type'] = data['Customer Type'].map({'Loyal Customer': 0, 'disloyal Customer': 1})
data['Customer Type'] = data['Customer Type'].astype(int)

data['Type of Travel'] = data['Type of Travel'].map({'Business travel': 0, 'Personal Travel': 1})
data['Type of Travel'] = data['Type of Travel'].astype(int)

data['Inflight wifi service'] = data['Inflight wifi service'].astype(int)

data['Departure/Arrival time convenient'] = data['Departure/Arrival time convenient'].astype(int)

data['Ease of Online booking'] = data['Ease of Online booking'].astype(int)

data['Gate location'] = data['Gate location'].astype(int)

data['Food and drink'] = data['Food and drink'].astype(int)

data['Online boarding'] = data['Online boarding'].astype(int)

data['Seat comfort'] = data['Seat comfort'].astype(int)

data['Inflight entertainment'] = data['Inflight entertainment'].astype(int)

data['On-board service'] = data['On-board service'].astype(int)

data['Leg room service'] = data['Leg room service'].astype(int)
data = data.drop('Unnamed: 0', axis=1)
data.to_csv('Airplane_clean.csv')
print(data)
