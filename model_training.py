import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

df=pd.read_csv('CollegeDistance.csv')

df.drop(columns=['rownames'],inplace=True)

label_encoders = {}
categorical_columns = ['gender', 'ethnicity', 'income', 'region']
boolean_columns = ['fcollege', 'mcollege', 'home', 'urban']
numerical_columns = ['score', 'unemp', 'wage', 'distance', 'tuition', 'education']

# Encoding categorical columns
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le  # Store label encoders for future use if needed

# Converting yes/no columns to binary 1/0
for column in boolean_columns:
    df[column] = df[column].apply(lambda x: 1 if x == 'yes' else 0)

# Scale numerical columns
scaler = StandardScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])



# Split data into train and test sets
def split_df(df):
    y = df['score']
    X = df.drop(columns='score')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, y_train, X_test, y_test

# Initialize models
grad_reg = GradientBoostingRegressor()

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    # Store results
    results = {"R^2": r2, "MSE": mse, "MAE": mae}
    # Print model performance
    print(f"Gradient Boosting Regressor Performance:")
    print(f"R^2 Score: {r2:.2f}")
    print(f"Root Mean Squared Error (MSE): {mse:.2f}")
    print(f"Mean Absolute Error (MAE): {mae:.2f}\n")
    print('\n')
    return results
#evaluate models
X_train, y_train, X_test, y_test=split_df(df)

results=evaluate_model(grad_reg,X_train, y_train, X_test, y_test)

