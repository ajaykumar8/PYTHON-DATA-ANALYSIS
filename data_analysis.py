import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import pymysql
import sqlalchemy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

def load_data(source_type, source_path, query=None):
    if source_type == 'csv':
        return pd.read_csv(source_path)
    elif source_type == 'excel':
        # Specify the engine manually based on the file extension
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

def exploratory_data_analysis(df):
    print("\nExploratory Data Analysis:")
    
    # Distribution of numerical features
    df.hist(figsize=(12, 10))
    plt.tight_layout()
    plt.show()
    
    # Correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.show()
    
    # Pair plot
    sns.pairplot(df)
    plt.show()

def trend_analysis(df, column, date_column=None):
    if date_column:
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
    plt.figure(figsize=(12,6))
    sns.lineplot(data=df, x=df.index, y=column)
    plt.title(f'Trend Analysis of {column}')
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.show()

def regression_analysis(df, dependent_var, independent_vars):
    X = df[independent_vars]
    y = df[dependent_var]
    
    # Scaling features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    predictions = model.predict(X_scaled)
    
    print("\nRegression Analysis:")
    print(f'Regression Coefficients: {model.coef_}')
    print(f'Regression Intercept: {model.intercept_}')
    print(f'Mean Squared Error: {mean_squared_error(y, predictions)}')
    print(f'R^2 Score: {r2_score(y, predictions)}')
    
    # Plotting regression
    plt.figure(figsize=(12, 6))
    plt.scatter(y, predictions)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Regression Analysis: Actual vs Predicted')
    plt.show()

def full_statistics(df):
    print("\nFull Statistical Summary:")
    print(df.describe())
    
    print("\nCorrelation Matrix:")
    print(df.corr())

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
    
    analysis_type = input("Enter the type of analysis (trend, regression, statistics, eda): ").lower()
    
    if analysis_type == 'trend':
        column = input("Enter the column for trend analysis: ")
        date_column = input("Enter the date column (or leave blank if not applicable): ")
        trend_analysis(df, column, date_column)
    
    elif analysis_type == 'regression':
        dependent_var = input("Enter the dependent variable: ")
        independent_vars = input("Enter the independent variables (comma-separated): ").split(',')
        regression_analysis(df, dependent_var, independent_vars)
    
    elif analysis_type == 'statistics':
        full_statistics(df)
    
    elif analysis_type == 'eda':
        exploratory_data_analysis(df)
    
    else:
        print("Unsupported analysis type!")

if __name__ == "__main__":
    main()
