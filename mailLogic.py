import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# =========================
# 1. Load Dataset
# =========================
data = pd.read_csv("D:\myNewProject\StudentPerformancePredictor\data.csv")

# print("Dataset Preview:")
print(data.head())

# =========================
# 2. Prepare Data                                           
# =========================
# Features (input)
X = data[['study_hours','attendance','previous_marks','sleep_hours']]

# Target (output)
y = data['result']

# =========================
# 3. Split Data
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 4. Train Model
# =========================
model = RandomForestClassifier()
model.fit(X_train, y_train)

# =========================
# 5. Accuracy Check
# =========================
accuracy = model.score(X_test, y_test)
print("\nModel Accuracy:", round(accuracy*100,2), "%")

# =========================
# 6. User Input Prediction
# =========================
print("\nEnter Student Details:")

study = float(input("Study Hours: "))
att = float(input("Attendance %: "))
marks = float(input("Previous Marks: "))
sleep = float(input("Sleep Hours: "))



# User input ko DataFrame me convert karo
user_data = pd.DataFrame({
    'study_hours': [study],
    'attendance': [att],
    'previous_marks': [marks],
    'sleep_hours': [sleep]
})

# Prediction
prediction = model.predict(user_data)

print("\nPrediction Result:", prediction[0])

# Probability
prob = model.predict_proba(user_data)

print("Confidence:")
print("Fail probability:", round(prob[0][0] * 100, 2), "%")
print("Pass probability:", round(prob[0][1] * 100, 2), "%")
