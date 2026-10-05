import pandas as pd
from sklearn.linear_model import LinearRegression

# Load data
data = pd.read_csv("student_data.csv")

# Input and output
X = data[["Hours"]]
y = data["Marks"]

# Create model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Get study hours
hours = float(input("Enter study hours: "))

# Predict marks
prediction = model.predict([[hours]])

print("Predicted marks:", prediction[0])