import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import ttk, messagebox

# Load and preprocess the data (same as before)
df = pd.read_csv('heart.csv')
df.rename(columns={
    'age': 'Age',
    'sex': 'Sex',
    'cp': 'ChestPainType',
    'trestbps': 'RestingBP',
    'chol': 'Cholesterol',
    'fbs': 'FastingBS',
    'restecg': 'RestingECG',
    'thalach': 'MaxHR',
    'exang': 'ExerciseAngina',
    'oldpeak': 'OldPeak',
    'slope': 'SlopePeakExercise',
    'ca': 'NumMajorVessels',
    'thal': 'Thalium',
    'target': 'Condition'
}, inplace=True)

# Selected features based on importance
selected_features = ['ChestPainType', 'MaxHR', 'OldPeak', 'Thalium', 
                     'Cholesterol', 'NumMajorVessels', 'Age']

# Prepare the data
X = df[selected_features]
y = df['Condition']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the RandomForest model
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model (optional print for debugging)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy on Test Set: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Tkinter GUI class
class HeartPredictorApp:
    def __init__(self, root):
        #Establish the caracteristiques of the window
        self.root = root
        self.root.title("CureZea Heart Disease Predictor")
        self.root.geometry("750x700")
        self.root.configure(bg="#f0f0f0")
        self.root.resizable(False, False)

        # Configure root to expand
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Main container frame
        self.main_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.main_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Configure main frame to expand
        self.main_frame.grid_rowconfigure(1, weight=1)  # Input frame row
        self.main_frame.grid_rowconfigure(3, weight=2)  # Result text row
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Title Label
        tk.Label(self.main_frame, text="Heart Disease Prediction", font=("Arial", 20, "bold"), 
                 bg="#f0f0f0", fg="#333").grid(row=0, column=0, pady=10, sticky="ew")

        # Frame for input fields
        self.input_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        self.input_frame.grid(row=1, column=0, pady=10, sticky="nsew")

        # Configure input frame to expand and center
        self.input_frame.grid_columnconfigure(0, weight=1)
        self.input_frame.grid_columnconfigure(1, weight=0)  # Fixed column for entries
        self.input_frame.grid_columnconfigure(2, weight=1)

        # Input fields and labels
        self.entries = {}
        self.create_input_fields()

        # Creating a frame to place the Buttons
        button_frame = tk.Frame(self.main_frame, bg="#f0f0f0")
        button_frame.grid(row=2, column=0, pady=20, sticky="ew")
        button_frame.grid_columnconfigure((0, 1, 2), weight=1)

        #Establishing the actual buttons
        tk.Button(button_frame, text="Predict", command=self.predict, 
                  bg="#4CAF50", fg="white", font=("Arial", 12), width=20, height=3).grid(row=0, column=0, padx=10)
        tk.Button(button_frame, text="Clear", command=self.clear_fields, 
                  bg="#f44336", fg="white", font=("Arial", 12), width=20, height=3).grid(row=0, column=1, padx=10)
        tk.Button(button_frame, text="Exit", command=self.root.quit, 
                  bg="#555", fg="white", font=("Arial", 12), width=20, height=3).grid(row=0, column=2, padx=10)

        # Result Text Area
        self.result_text = tk.Text(self.main_frame, height=6, width=50, font=("Arial", 12), state="disabled")
        self.result_text.grid(row=3, column=0, pady=20, padx=20, sticky="nsew")

    def create_input_fields(self):
        #create the tags, description and ranges of each parameter
        fields = [
            ("Chest Pain Type (0-3):", "0 = No pain, 1 = Typical angina, 2 = Atypical angina, 3 = Non-anginal pain", (0, 3)),
            ("Max Heart Rate (70-200):", "Maximum heart rate during exercise (beats per minute)", (70, 200)),
            ("Old Peak (0.0-6.2):", "ST depression observed in ECG", (0.0, 6.2)),
            ("Thalium Test (0-3):", "0 = Normal, 1 = Abnormal, 2 = Borderline, 3 = Severe", (0, 3)),
            ("Cholesterol (100-600):", "Cholesterol level in mg/dL", (100, 600)),
            ("Num Major Vessels (0-3):", "Number of major blood vessels affected", (0, 3)),
            ("Age (20-80):", "Patient's age in years", (20, 80))
        ]

        for i, (label_text, tooltip, range_vals) in enumerate(fields):
            # Create labels and entry without a sub-frame, directly in the input_frame
            tk.Label(self.input_frame, text=label_text, font=("Arial", 12), bg="#f0f0f0", anchor="w").grid(row=i, column=0, sticky="e", padx=5, pady=5)
            entry = tk.Entry(self.input_frame, font=("Arial", 12), width=10)
            entry.grid(row=i, column=1, padx=5, pady=5)
            tk.Label(self.input_frame, text=tooltip, font=("Arial", 10, "italic"), bg="#f0f0f0", fg="#666", anchor="w").grid(row=i, column=2, sticky="w", padx=5, pady=5)
            self.entries[label_text] = (entry, range_vals)

    #Make sure the inputs are correct
    def validate_inputs(self):
        patient_data = []
        for label, (entry, (min_val, max_val)) in self.entries.items():
            try:
                value = float(entry.get())
                #If the input value does not go with the ranges
                if value < min_val or value > max_val:
                    #error message
                    messagebox.showerror("Input Error", f"{label} must be between {min_val} and {max_val}")
                    return None
                patient_data.append(value)
            except ValueError:
                messagebox.showerror("Input Error", f"Invalid input for {label}. Please enter a valid number.")
                return None
        return patient_data

    #the actual prediction of the model
    def predict(self):
        patient_data = self.validate_inputs()
        if patient_data is None:
            return

        # Prepare data for prediction
        patient_df = pd.DataFrame([patient_data], columns=selected_features)
        patient_scaled = scaler.transform(patient_df)

        # Make prediction
        prediction = model.predict(patient_scaled)
        probability = model.predict_proba(patient_scaled)[0][1]

        # Display results
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Probability of Heart Disease: {round(probability, 2)}\n\n")
        
        if probability >= 0.70:
            self.result_text.insert(tk.END, "Interpretation: Potential risk of heart disease.\n")
            self.result_text.insert(tk.END, "Recommendation: Consult a healthcare provider for further tests.")
        else:
            self.result_text.insert(tk.END, "Interpretation: No heart disease.\n")
            self.result_text.insert(tk.END, "Recommendation: Maintain a healthy lifestyle and regular check-ups.")
        
        self.result_text.config(state="disabled")

    #Clear the inputs and the output boxes
    def clear_fields(self):
        #clear the text
        for _, (entry, _) in self.entries.items():
            entry.delete(0, tk.END)
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state="disabled")

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = HeartPredictorApp(root)
    root.mainloop()
