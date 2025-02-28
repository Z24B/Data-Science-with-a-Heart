import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('heart.csv')

#Change the name of columns
data.columns = ['Age', 'Sex', 'Chest_pain_type', 'Resting_bp', 
                'Cholesterol', 'Fasting_bs', 'Resting_ecg', 
                'Max_heart_rate', 'Exercise_induced_angina', 
                'ST_depression', 'ST_slope', 'Num_major_vessels',
                'Thallium_test', 'Condition']

#Use the columns we want to be using
selected_features = ['Exercise_induced_angina', 'Chest_pain_type', 'ST_depression', 'Max_heart_rate', 'ST_slope']

#preparing data
X = data[selected_features]
y = data['Condition']

#split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Create the trees of the model
model = RandomForestClassifier(n_estimators=200, max_depth=10,random_state=42)
model.fit(X_train_scaled, y_train)

#Get the accuracy of the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

while True:
    #Get user parameters
    print("\nPlease enter the following values for a patient:")
    exercise_angina = float(input("Exercise induced angina (0 for no, 1 for yes): "))
    chest_pain = float(input("Chest pain type (0-3): "))
    st_depression = float(input("ST depression (0.0 to 6.2): "))
    max_heart_rate = float(input("Max heart rate (70-200): "))
    st_slope = float(input("Slope of peak exercise (0-2): "))

    #place input into a list, and turn it into a usable dataframe
    patient_data = [exercise_angina, chest_pain, st_depression, max_heart_rate, st_slope]
    patient_df = pd.DataFrame([patient_data], columns=selected_features)
    patient_scaled = scaler.transform(patient_df)

    #predict with the input information
    prediction = model.predict(patient_scaled)
    probability = model.predict_proba(patient_scaled)[0][1]

    #Output the results
    print("\nResults:")
    #print("Prediction:", prediction[0])
    print("Probability of heart disease:", round(probability, 2))
    if probability >= 0.75:
        print("Interpretation: Has heart disease")
    else:
        print("Interpretation: No heart disease")
