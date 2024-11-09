import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl

print("Starting data loading...")
# Load data
file_path = 'c:/Users/johnw/OneDrive/Desktop/JobAppData.xlsx'
df = pd.read_excel(file_path, sheet_name='ALL')
print("Data loaded successfully. Data shape:", df.shape)

# Select relevant columns
df = df[['Company', 'Job Title', 'Phase 2', 'Rejected', 'Interval', 
         'Application Source', 'Resume Version']]
print("Selected relevant columns. Data shape:", df.shape)

# Check for missing values
print("Missing values per column:\n", df.isnull().sum())

# Drop rows where 'Rejected' is missing, as it's the target variable
df = df.dropna(subset=['Rejected'])
print("Dropped rows with missing target values ('Rejected'). Data shape:", df.shape)

# Fill missing values in 'Interval' with the median of the column
df['Interval'] = df['Interval'].fillna(df['Interval'].median())
print("Filled missing values in 'Interval'.")

# Encode categorical columns and handle binary columns
df['Company'] = df['Company'].astype('category').cat.codes
df['Job Title'] = df['Job Title'].astype('category').cat.codes
df['Application Source'] = df['Application Source'].astype('category').cat.codes
df['Resume Version'] = df['Resume Version'].astype('category').cat.codes
df['Phase 2'] = df['Phase 2'].apply(lambda x: 1 if pd.notnull(x) else 0)  # Binary encoding for Phase 2
print("Categorical encoding and binary conversion completed.")

# Define features and target (use 'Rejected' as target instead of 'Hired')
X = df.drop(['Rejected'], axis=1)  # Features
y = df['Rejected']  # Target: 1 = rejected, 0 = not rejected
print("Features and target defined. X shape:", X.shape, "y shape:", y.shape)

# Ensure there is enough data to split
if len(X) > 0:
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split into training and testing sets.")

    # Initialize and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model training completed.")

    # Make predictions and evaluate
    y_pred = model.predict(X_test)
    print("Predictions made. Evaluation results:")
    print(classification_report(y_test, y_pred))
    print('Accuracy:', accuracy_score(y_test, y_pred))

    # Initialize Isolation Forest for anomaly detection
    iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    iso_forest.fit(X)
    print("Anomaly detection model trained.")

    # Detect anomalies (-1 = anomaly, 1 = normal)
    df['Anomaly'] = iso_forest.predict(X)
    anomalous_data = df[df['Anomaly'] == -1]
    print("Anomalies Detected:")
    print(anomalous_data)

    # Feature importance visualization
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    feature_importances.nlargest(10).plot(kind='barh')
    plt.xlabel("Importance")
    plt.title("Top Features Impacting Rejection")
    plt.show()

    # Save results back to Excel
    print("Saving results to Excel...")
    with pd.ExcelWriter('JobAppResults.xlsx') as writer:
        df.to_excel(writer, sheet_name='Data with Anomalies', index=False)
        pd.DataFrame({'Actual': y_test, 'Predicted': y_pred}).to_excel(writer, sheet_name='Predictions', index=False)
    print("Results saved to JobAppResults.xlsx.")
else:
    print("Not enough data to proceed with model training.")
