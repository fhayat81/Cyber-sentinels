# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the CSV file
def regress(feature1, feature2, feature3, feature4):
    file_path = 'data/sample_data.csv'  # Replace with your CSV file path
    data = pd.read_csv(file_path)

    # Step 2: Separate features (X) and output (y)
    X = data[['Fact%', 'Keyword%', 'Para%', 'Gr error%']]  # Replace with actual column names
    y = data['Publishable']  # Replace with actual output column name

    # Step 3: Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 4: Create and train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)  # Train the model

    # Step 5: Predict on the test data
    y_pred = model.predict(X_test)

    # Step 6: Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Step 7: Manual Input for Prediction

    # Combine inputs into a single array and reshape for prediction
    manual_input = np.array([feature1, feature2, feature3, feature4]).reshape(1, -1)

    # Predict the output
    manual_prediction = model.predict(manual_input)
    manual_probability = model.predict_proba(manual_input)

    # Display the result
    print(f"\nPredicted Output: {manual_prediction[0]}")
    print(f"Prediction Probability: {manual_probability[0]}")
    return manual_prediction[0]
