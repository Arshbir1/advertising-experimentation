"""Executable analysis script generated from the corresponding notebook.

Run from the repository root after placing ABAdRecall.csv in data/.
"""

# %%
# linear algebra and data processing
import numpy as np
import pandas as pd

# plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

# math and statistics libraries
from scipy import stats
from scipy.stats import skew, norm
import math

# others
import datetime

# column transformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# ignore  warnings
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

# %%
from pathlib import Path

DATA_CANDIDATES = [
    Path.cwd() / "data" / "ABAdRecall.csv",
    Path.cwd().parent / "data" / "ABAdRecall.csv",
]

DATA_PATH = next((path for path in DATA_CANDIDATES if path.exists()), None)

if DATA_PATH is None:
    raise FileNotFoundError(
        "ABAdRecall.csv was not found. Place the dataset in the repository's data/ directory."
    )

data = pd.read_csv(DATA_PATH)
data.head()

# %%
# the shape of our data
print('the size of the dataset is: ', data.shape[0], 'rows and', data.shape[1], 'columns')

# %%
# segment the features into 2 categories: categorical and numerical features
categorical = []
numerical = []
for col in data.columns:
  if data[col].dtype == object:
    categorical.append(col)
  elif data[col].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
    numerical.append(col)

# check the data types
features = categorical + numerical 
data = data[features]
data.dtypes

# %%
# change the date col to a timestamp feature
data.date = pd.to_datetime(data["date"])

# %%
# measures of spread and central tendency of numerical features
data.describe()

# %%
# number of unique entries across all features
for col in data.columns:
  print(col, '-', data[col].nunique())

# %%
# value counts for experiment, yes and no columns
for col  in ['yes', 'no', 'experiment']:
  print(col, ': \n',data[col].value_counts())

# %%
# hour distribution
f, ax = plt.subplots(figsize=(7, 6))
sns.distplot(data['hour'], bins = 20, color = 'Magenta')
ax.set(ylabel="Frequency")
ax.set(xlabel="hour")
ax.set(title="hour distribution")

# %%
# yes distribution
f, ax = plt.subplots(figsize=(7, 6))
sns.distplot(data['yes'], bins = 20, color = 'black')
ax.set(ylabel="Frequency")
ax.set(xlabel="yes")
ax.set(title="yes distribution")

# %%
# no distribution
f, ax = plt.subplots(figsize=(7, 6))
sns.distplot(data['no'], bins = 20, color = 'green')
ax.set(ylabel="Frequency")
ax.set(xlabel="no")
ax.set(title="no distribution")

# %%
# platform_os distribution
f, ax = plt.subplots(figsize=(7, 6))
sns.distplot(data['platform_os'], bins = 20, color = 'blue')
ax.set(ylabel="Frequency")
ax.set(xlabel="platform_os")
ax.set(title="platform_os distribution")

# %%
# yes
sns.countplot(data['yes'])
plt.title('a countplot indicating unique value counts of the yes column')

# %%
# no
sns.countplot(data['no'])
plt.title('a countplot indicating unique value counts of the no column')

# %%
# experiment
sns.countplot(data['experiment'])
plt.title('a countplot indicating unique value counts of the experiment column')

# %%
# platform_os
sns.countplot(data['platform_os'])
plt.title('a countplot indicating unique value counts of the platform_os column')

# %%
# device_make (top 10 most popular)
a = data.device_make.value_counts(ascending=False).head(5)
a = pd.DataFrame(a)
plt.figure(figsize = (6,5))
sns.barplot(x = a.index, y = a.device_make)
plt.title('a plot indicating value counts of the most popular device makes')
plt.xticks(rotation = 45)

# %%
# browser
plt.figure(figsize = (12,10))
sns.countplot(data['browser'])
plt.title('a countplot indicating unique value counts of the browser column')
plt.xticks(rotation = 45)

# %%
# hour
plt.figure(figsize = (12,10))
sns.countplot(data['hour'])
plt.title('a countplot indicating unique value counts of the hour column')
plt.xticks(rotation = 45)

# %%
corr = data.corr()
plt.figure(figsize = (8,6))
sns.heatmap(corr, cmap="YlGnBu")
plt.title('a heatmap indicating correlation between the variables')

# %%
# yes vs no
plt.figure(figsize=(6, 4))
sns.scatterplot(data = data, x='yes', y='no')
plt.title('Scatter plot showing relationship between yes and no')

# %%
# encode the categorical features in order to be able to plot them
df = data.copy()   # make a copy to void tempering with the original one
columns = [1, 3, 4]
for col in columns:
    x = df.iloc[:, col].values
    x = x.reshape(-1,1)
    encoder = LabelEncoder()
    encoder = encoder.fit(x)
    x = encoder.transform(x)
    df.iloc[:, col] = x

# %%
# device vs browser
plt.figure(figsize=(6, 4))
sns.scatterplot(data = df, x='device_make', y='browser')
plt.title('Scatter plot showing relationship between device_make and browser')

# %%
# device  vs os
plt.figure(figsize=(6, 4))
sns.scatterplot(data = df, x='device_make', y='platform_os')
plt.title('Scatter plot showing relationship between device_make and os')

# %%
# os vs browser
plt.figure(figsize=(6, 4))
sns.scatterplot(data = df, x='browser', y='platform_os')
plt.title('Scatter plot showing relationship between browser and os')

# %%
# hour vs yes
plt.figure(figsize=(6, 4))
sns.scatterplot(data = df, x='hour', y='yes')
plt.title('Scatter plot showing relationship between hour and yes')

# %%
# hour vs no
plt.figure(figsize=(6, 4))
sns.scatterplot(data = df, x='hour', y='no')
plt.title('Scatter plot showing relationship between hour and no')
