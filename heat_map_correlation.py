# Import modules
import matplotlib.pyplot as mp
import pandas as pd
import seaborn as sb

# Import file with data
df = pd.read_csv("heart.csv")

def heat_map_correlation():
    #print(data.corr(numeric_only=True))
    dataplot = sb.heatmap(df.corr(numeric_only=True), cmap="YlGnBu", annot=True)
    mp.show()
