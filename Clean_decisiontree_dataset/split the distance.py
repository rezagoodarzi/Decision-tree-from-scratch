import pandas as pd


data = pd.read_csv("Airplane_clean.csv")

bins_distance = [0, 400, 700, 1000, 1200, 1400, 1600, 2000, 2600, 3800, 5000]
bins_age = [-1, 6, 18, 28, 32, 36, 40, 44, 50, 300]
bins_Departure = [0, 2, 8, 15, 30, 120, 300, 1200, 2000]
bins_Arrival = [0, 2, 8, 15, 30, 120, 300, 1200, 2000]


# using mean-max scaling This technique scales your data linearly to a specific range.


data['Flight Distance'] = pd.cut(data['Flight Distance'], bins_distance,right=True,include_lowest=True,labels=False)
data['Age'] = pd.cut(data['Age'], bins_age,right=True,include_lowest=True,labels=False )
data['Departure Delay in Minutes'] = pd.cut(data['Departure Delay in Minutes'], bins_Departure,right=True,include_lowest=True,labels=False)
data['Arrival Delay in Minutes'] = pd.cut(data['Arrival Delay in Minutes'], bins_Arrival,right=True,include_lowest=True, labels=False)

'''
def map_flight_distance(flight_distance):
    if int(flight_distance // (2300 / 10)) > 10:
        return 10

    return round(int(flight_distance // (2300 / 10)))

def map_age(Age):
    if int(Age // (80 / 10)) > 10:
        return 10

    return round(int(Age // (80 / 10)))


def map_Arrival(Delay):
    if int(Delay // (30 / 5)) > 5:
        return 5

    return round(int(Delay // (30 / 5)))


def map_Departure(Delay):
    if int(Delay // (30 / 5)) > 5:
        return 5

    return round(int(Delay // (30 / 5)))


data['Flight Distance'] = data['Flight Distance'].apply(map_flight_distance)
data['Age'] = data['Age'].apply(map_age)
data['Arrival Delay in Minutes'] = data['Arrival Delay in Minutes'].apply(map_Arrival)
data['Departure Delay in Minutes'] = data['Departure Delay in Minutes'].apply(map_Departure)
'''
data.to_csv('Airplane_clean_split.csv', index=False)
