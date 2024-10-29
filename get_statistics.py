import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv('CollegeDistance.csv')

print("Podstawowe informacje o danych:")
print(df.info())
print("\nPodgląd danych:")
print(df.head())

# Checking for missing values
print("\nBrakujące wartości w każdej kolumnie:")
print(df.isnull().sum())

# Descriptive statistics for variables
print("\nStatystyki opisowe:")
print(df.describe().round(2))

# Conversion of yes/no values to 1/0
cols_to_convert = ['fcollege', 'mcollege','home','urban']
df[cols_to_convert] = df[cols_to_convert].replace({'yes': 1, 'no': 0})


# Convert boolean columns to integers for histogram purposes
bool_cols = df.select_dtypes(include=['bool']).columns
for col in bool_cols:
    df[col] = df[col].astype(int)

# Distribution of numerical variables - histograms
numeric_df = df.select_dtypes(include=['float64', 'int64'])
axes=numeric_df.hist(bins=15, figsize=(15, 10))
plt.suptitle("Histograms of numerical variables",fontsize=16)
for ax in axes.flatten():
    ax.set_xlabel('Value',fontsize=8)
    ax.set_ylabel('Frequency',fontsize=8)
plt.subplots_adjust(top=0.9, hspace=0.5)
plt.savefig('Histograms.png', format='png', dpi=300)
print("Histograms saved to 'Histograms.png'.")

# Correlation between numerical variables (if present)
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation matrix of numerical variables")
plt.savefig('Correlations.png', format='png', dpi=300)
print("Correlation matrix saved to 'Correlations.png'.")

# Create a figure with subplots
num_columns = len(numeric_df.columns)
fig, axes = plt.subplots(nrows=1, ncols=num_columns, figsize=(15, 5))

# Create box plots for all numeric columns
num_plots_box = len(numeric_df.columns)
nrows_box = (num_plots_box // 3) + (1 if num_plots_box % 3 else 0)

fig_box, axs_box = plt.subplots(nrows=nrows_box, ncols=3, figsize=(15, 5 * nrows_box))
axs_box = axs_box.flatten()  # Flatten to 1D array for easy iteration

for i, col in enumerate(numeric_df.columns):
    numeric_df.boxplot(column=[col], ax=axs_box[i])
    axs_box[i].set_title(f'Box Plot of {col}')

for j in range(i + 1, len(axs_box)):
    fig_box.delaxes(axs_box[j])

plt.tight_layout()
plt.savefig('Boxplot.png')
plt.close(fig_box)
print("Box plots saved to 'Boxplot.png'.")

# Calculate IQR for each column
Q1 = numeric_df.quantile(0.25)  # First quartile
Q3 = numeric_df.quantile(0.75)  # Third quartile
IQR = Q3 - Q1           # Interquartile Range

# Oblicz dolną i górną granicę
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Zidentyfikuj wartości odstające
outliers = numeric_df[(numeric_df < lower_bound) | (numeric_df > upper_bound)]

outliers.drop(['fcollege','mcollege','home','urban', 'rownames'],axis=1,inplace=True)


print("Wartości odstające:\n", outliers.count())