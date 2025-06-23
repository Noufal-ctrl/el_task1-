# Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

#Load Dataset
df = pd.read_csv("/content/Titanic-Dataset.csv")

# Check data types and nulls
print(df.info())         

# Summary stats
print(df.describe())     

# First few rows
print(df.head())         

#Handle Missing Values
df['Age'].fillna(df['Age'].median(), inplace=True)  # Median 
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # Mode 
df.drop(columns=['Cabin'], inplace=True)  # Droping column with too many missing values

#Encoding Categorical Features
le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])  # here Male = 1, Female = 0
df['Embarked'] = le.fit_transform(df['Embarked'])  # Encode Embarked

# Normalizing Numerical Features
scaler = StandardScaler()
num_cols = ['Age', 'Fare']
df[num_cols] = scaler.fit_transform(df[num_cols])

# Visualizing the  Outliers
plt.figure(figsize=(10, 4))
sns.boxplot(x=df['Age'])
plt.title("Boxplot of Age")
plt.show()

plt.figure(figsize=(10, 4))
sns.boxplot(x=df['Fare'])
plt.title("Boxplot of Fare")
plt.show()

#remove outliers (using IQR for 'Fare')
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['Fare'] >= Q1 - 1.5 * IQR) & (df['Fare'] <= Q3 + 1.5 * IQR)]

# Final Cleaned Data
print(df.head())
