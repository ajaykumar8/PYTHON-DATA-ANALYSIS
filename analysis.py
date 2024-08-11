import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlalchemy
import pymysql
import sqlite3
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def load_data(source_type, source_path, query=None):
    if source_type == 'csv':
        return pd.read_csv(source_path)
    elif source_type == 'excel':
        if source_path.endswith('.xlsx'):
            return pd.read_excel(source_path, engine='openpyxl')
        elif source_path.endswith('.xls'):
            return pd.read_excel(source_path, engine='xlrd')
        else:
            raise ValueError("Unsupported Excel file format!")
    elif source_type == 'sql':
        engine = sqlalchemy.create_engine(source_path)
        return pd.read_sql_query(query, engine)
    elif source_type == 'sqlite':
        conn = sqlite3.connect(source_path)
        return pd.read_sql_query(query, conn)
    elif source_type == 'mysql':
        conn = pymysql.connect(source_path)
        return pd.read_sql_query(query, conn)
    else:
        raise ValueError("Unsupported source type!")

def data_cleaning(df):
    print("Initial Data Overview:")
    print(df.info())
    print(df.head())
    
    # Handle missing values
    df.fillna(method='ffill', inplace=True)
    df.dropna(inplace=True)
    
    # Handle duplicates
    df.drop_duplicates(inplace=True)
    
    print("\nData Overview after Cleaning:")
    print(df.info())
    print(df.describe())

def descriptive_analysis(df):
    print("\nDescriptive Statistics:")
    print(df.describe())
    
    print("\nCorrelation Matrix:")
    print(df.corr())
    
    # Visualizations
    df.hist(figsize=(12, 10))
    plt.tight_layout()
    plt.show()
    
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()

def diagnostic_analysis(df):
    print("\nDiagnostic Analysis:")
    
    # Example: Distribution of a specific variable
    sns.histplot(df['sales_amount'], kde=True)
    plt.title('Distribution of Sales Amount')
    plt.xlabel('Sales Amount')
    plt.ylabel('Frequency')
    plt.show()
    
    # Example: Identify anomalies or patterns
    df['sales_amount_diff'] = df['sales_amount'].diff()
    sns.lineplot(data=df, x=df.index, y='sales_amount_diff')
    plt.title('Sales Amount Differences Over Time')
    plt.xlabel('Date')
    plt.ylabel('Difference in Sales Amount')
    plt.show()

def predictive_analysis(df, dependent_var, independent_vars):
    print("\nPredictive Analysis:")
    X = df[independent_vars]
    y = df[dependent_var]
    
    # Scaling features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    predictions = model.predict(X_scaled)
    
    print(f'Regression Coefficients: {model.coef_}')
    print(f'Regression Intercept: {model.intercept_}')
    print(f'Mean Squared Error: {mean_squared_error(y, predictions)}')
    print(f'R^2 Score: {r2_score(y, predictions)}')
    
    # Plotting regression
    plt.figure(figsize=(12, 6))
    plt.scatter(y, predictions)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predictive Analysis: Actual vs Predicted')
    plt.show()

def prescriptive_analysis(df, recommendations):
    print("\nPrescriptive Analysis:")
    
    # Example of applying a recommendation based on analysis
    print("Recommendations:")
    for recommendation in recommendations:
        print(f"- {recommendation}")

def execute_query(connection_string, query):
    engine = sqlalchemy.create_engine(connection_string)
    df = pd.read_sql_query(query, engine)
    return df

def main():
    # User Input
    source_type = input("Enter the source type (csv, excel, sql, sqlite, mysql): ").lower()
    source_path = input("Enter the source path (file path or connection string): ")
    if source_type in ['sql', 'sqlite', 'mysql']:
        query = input("Enter the SQL query: ")
    else:
        query = None
    df = load_data(source_type, source_path, query)
    
    data_cleaning(df)
    
    analysis_type = input("Enter the type of analysis (descriptive, diagnostic, predictive, prescriptive): ").lower()
    
    if analysis_type == 'descriptive':
        descriptive_analysis(df)
    
    elif analysis_type == 'diagnostic':
        diagnostic_analysis(df)
    
    elif analysis_type == 'predictive':
        dependent_var = input("Enter the dependent variable: ")
        independent_vars = input("Enter the independent variables (comma-separated): ").split(',')
        predictive_analysis(df, dependent_var, independent_vars)
    
    elif analysis_type == 'prescriptive':
        recommendations = input("Enter recommendations (comma-separated): ").split(',')
        prescriptive_analysis(df, recommendations)
    
    else:
        print("Unsupported analysis type!")

if __name__ == "__main__":
    main()
