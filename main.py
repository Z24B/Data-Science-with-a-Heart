#FILE NO LONGER IN USE (from first code runs)
from data_clean import load_and_clean
from data_visuals import plot_visual
from risk_factors import risk_factors
from data_correlation import plot_correlation
from data_explore import group_by

'''
NOTE: this file is only to be used to execute final code
'''

file_path = "M:\CE101 (Team Project)\heart.csv"

#Data preparation
df = load_and_clean(file_path)
print(df.head())

#Data visualisations
plot_visual(df)

#Analyse the risk factors
risk_factors(df)

#Correlation matrix
heat_map_correlation(df)

#Using group by to explore data
group_by(df)
