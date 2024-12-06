import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
import plotly.graph_objs as go
import plotly.offline as py

#get the file
heart = pd.read_csv("heart.csv")

# Rename origical columns
heart.columns = ['Age', 'Sex', 'Chest_pain_type', 'Resting_bp', 
              'Cholesterol', 'Fasting_bs', 'Resting_ecg', 
              'Max_heart_rate', 'Exercise_induced_angina', 
              'ST_depression', 'ST_slope', 'Num_major_vessels',
              'Thallium_test', 'Condition']

#Limit the rows and columns
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 50)

#Sorted Data, select columns
"""sorted_data = heart.sort_index(ascending=True)
other_data = sorted_data[['restecg', 'oldpeak']]
sorted_data = sorted_data[['age', 'chol']]

print(sorted_data.describe())
print(other_data.describe())

sorted_data.plot(kind='hist', x='age', y='chol')"""
def condition_ratio(data):
    """
    Make a pie chart of 'Condition' values
    Condition: 0 = Benign, 1 = Malignant
    """
    results = data['Condition'].value_counts()
    values = [results[0], results[1]]
    labels = ['Benign', 'Malignant']
    colors = ['MediumSeaGreen', 'Coral']
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker=dict(
            colors=colors,
            line=dict(color='Black', width=2)
        )
    )])

    #py.iplot([fig_pie])
    fig_pie.show()
    return py

def risk_factors_dist_sex(data):
    """
    Show distributions of risk factors for each sex
    """
    fig = plt.figure(figsize=(18, 8))
    
    # Resting blood pressure for each sex
    plt.subplot(2, 3, 1)
    trestbps_female = data[data['Sex']==0]['Resting_bp']
    trestbps_male = data[data['Sex']==1]['Resting_bp']
    sns.distplot(trestbps_female, color='Orange') 
    sns.distplot(trestbps_male, color='Blue')  
    plt.title('Distribution Resting Blood Pressure (mmHg)', fontweight='bold')
    plt.gca().legend(title='Sex', labels=['Female','Male'])
    plt.axvline(x=130, color='r', linestyle='--', label='Hypertension: over 130 mmHg')
    
    plt.subplot(2, 3, 4)
    sns.boxplot(x=data['Resting_bp'], y=data['Sex'], 
                palette='Set1', orient='h')
    
    
    # Serum cholesterol distribution for each sex
    plt.subplot(2, 3, 2)
    chol_female = data[data['Sex']==0]['Cholesterol']
    chol_male = data[data['Sex']==1]['Cholesterol']
    sns.distplot(chol_female, color='Orange')   
    sns.distplot(chol_male, color='Blue')
    plt.title('Distribution Serum Cholesterol (mg/dl)', fontweight='bold')
    plt.gca().legend(title='Sex', labels=['Female','Male'])
    plt.axvline(x=200, color='r', linestyle='--', label='High Cholesterol: over 200 mg/dl')
    
    plt.subplot(2, 3, 5)
    sns.boxplot(x=data['Cholesterol'], y=data['Sex'], 
                palette='Set1', orient='h')
    
    
    # Max heart rate distribution for each sex 
    plt.subplot(2, 3, 3)
    thalach_female = data[data['Sex']==0]['Max_heart_rate']
    thalach_male = data[data['Sex']==1]['Max_heart_rate']
    sns.distplot(thalach_female, color='Orange')   
    sns.distplot(thalach_male, color='Blue')
    plt.title('Distribution Max Heart Rate (bpm)', fontweight='bold')
    plt.gca().legend(title='Sex', labels=['Female','Male'])
    
    plt.subplot(2, 3, 6)
    sns.boxplot(x=data['Max_heart_rate'], y=data['Sex'], 
                palette='Set1', orient='h')
    
    plt.tight_layout()
    plt.show()
    
    
risk_factors_dist_sex(heart)    
condition_ratio(heart)

