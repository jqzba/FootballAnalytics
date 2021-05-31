#data preprocessing
import pandas as pd
#produces a prediction model in the form of an ensemble of weak prediction models, typically decisioni
from IPython.display import display


loc = "C:/Users/User/Documents/Data/"
data = pd.read_csv(loc + "final_dataset.csv")

print(data)


data_before = data[:667]
data_covid = data[668:]

print(data_covid)

n_matches = data_covid.shape[0]

n_features = data.shape[1] - 1

n_homewins = len(data_covid[data_covid.FTR == 'H'])

win_rate = (float(n_homewins) / (n_matches)) * 100

print(win_rate)




