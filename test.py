import pandas as pd
from datetime import datetime as dt


metrics_data = pd.read_csv("https://footystats.org/c-dl.php?type=league&comp=2012")

print(metrics_data.head())