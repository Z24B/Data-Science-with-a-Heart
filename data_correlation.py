import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation(df):
    # Visualize correlations
    correlation_matrix = df.corr()

    plt.figure(figsize=(20, 15))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix of Heart Disease Dataset')
    plt.show()
   
