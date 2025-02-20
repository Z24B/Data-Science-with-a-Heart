#code moved to python notebook

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


'''OLD CODE FOR HEATMAP
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_correlation(df):
    # Visualize correlations
    correlation_matrix = df.corr()

    plt.figure(figsize=(20, 15))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix of Heart Disease Dataset')
    plt.show()

def heat_map_correlation(df):
    #print(data.corr(numeric_only=True))
    dataplot = sb.heatmap(df.corr(numeric_only=True), cmap="YlGnBu", annot=True)
    mp.show()
    '''
