import matplotlib.pyplot as plt
import seaborn as sns

'''
This file is to analyse some statistical infomation for all the data.
'''

def group_by(df):

    # Group by 'Target' and calculate the mean for each group
    grouped_data = df.groupby('Target').mean()
    print(grouped_data)

    # Visualize the grouped data
    grouped_data.plot(kind='bar', figsize=(12, 8), colormap='viridis')
    plt.title('Mean Values of Features by Target')
    plt.xlabel('Target (0 = No Disease, 1 = Disease Present)')
    plt.ylabel('Mean Values')
    plt.xticks(rotation=0)
    plt.legend(loc='upper right')
    plt.show()
