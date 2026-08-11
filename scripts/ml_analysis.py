"""Executable analysis script generated from the corresponding notebook.

Run from the repository root after placing ABAdRecall.csv in data/.
"""

# %%
# data processing and Linear Algebra
import pandas as pd
import numpy as np

# plotting 
import seaborn as sns
import matplotlib.pyplot as plt


# maths and statistics
from scipy import stats
from scipy.stats import skew, norm
import math

# ML models 
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# House keeping (data preparation and model evaluation)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, log_loss


# others
import datetime as dt

# ignore warnings
import warnings
warnings.filterwarnings(action="ignore")

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
# Null Values
data.isna().any()

# %%
categorical = []
numerical = []
for col in data.columns:
  if data[col].dtype == object:
    categorical.append(col)
  elif data[col].dtype in ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']:
    numerical.append(col)


features = categorical + numerical 
df = data[features]
df.head()

# %%
# check if they exist
def iqr_outlier_test(data, col):
  Q1 = np.percentile(data[col], 25, interpolation = 'midpoint')  
  Q2 = np.percentile(data[col], 50, interpolation = 'midpoint')  
  Q3 = np.percentile(data[col], 75, interpolation = 'midpoint')  
  IQR = stats.iqr(data[col], interpolation = 'midpoint') 
  o = (data[col] < (Q1 - 1.5 * IQR)) |(data[col] > (Q3 + 1.5 * IQR))
  m = o.unique()
  return m

for col in df[numerical].columns:
  print(col, '-', iqr_outlier_test(df, col))

# %%
# #treat them
# def treat_outliers(data, col):
#   data[col] = data[col].clip(lower=data[col].quantile(0.10), upper= data[col].quantile(0.90))

# for col in df[numerical].columns:
#   treat_outliers(df, col)

# #check again
# for col in df[numerical].columns:
#   print(col, '-', iqr_outlier_test(df, col))

# %%
# Find skewed numerical features
skew_features = df[numerical].apply(lambda x: skew(x)).sort_values(ascending=False)

high_skew = skew_features[skew_features > 0.5]
skew_index = high_skew.index

print("There are {} numerical features with Skew > 0.5 :".format(high_skew.shape[0]))
skewness = pd.DataFrame({'Skew' :high_skew})
skew_features

# %%
# f, ax = plt.subplots(figsize=(7, 6))
# sns.distplot(df['platform_os'], bins = 20, color = 'blue')
# ax.set(ylabel="Frequency")
# ax.set(xlabel="platform_os")
# ax.set(title="platform_os distribution")

# print(df.platform_os.nunique())

# %%
# f, ax = plt.subplots(figsize=(7, 6))
# sns.distplot(df['hour'], bins = 20, color = 'Magenta')
# ax.set(ylabel="Frequency")
# ax.set(xlabel="hour")
# ax.set(title="hour distribution")

# %%
# df.hour.value_counts()

# %%
def correlation_map(f_data, f_feature, f_number):
    f_most_correlated = f_data.corr().nlargest(f_number,f_feature)[f_feature].index
    f_correlation = f_data[f_most_correlated].corr()
    
    f_mask = np.zeros_like(f_correlation)
    f_mask[np.triu_indices_from(f_mask)] = True
    with sns.axes_style("white"):
        f_fig, f_ax = plt.subplots(figsize=(8, 6))
        f_ax = sns.heatmap(f_correlation, mask=f_mask, vmin=0, vmax=1, square=True,
                           annot=True, annot_kws={"size": 10}, cmap="BuPu")

    plt.show()

correlation_map(df, 'yes', 4)

# %%
# Feature generation
df['date'] = pd.to_datetime(df['date'],format='%Y-%m-%d')  # convert date to datetime object
# df['day']=df['date'].dt.day                       #extract the day
df['dayofweek_num']=df['date'].dt.dayofweek       # extract the day of the week

# features reduction
df = df.drop(['date'], axis = 1)  #drop  the date col
df = df.drop(['auction_id'], axis = 1)  #drop  the auction_id col
df.tail(5)

# %%
#check the datatypes
print(df.shape)
df.dtypes

# %%
# get the location of the 3 categorical columns
features = df.copy()
indices = []
for col in ['browser', 'experiment', 'device_make']:
    k = features.columns.get_loc(col)
    indices.append(k)
    
indices

# %%
# Encoding categorical variables using Label Encoder
columns = indices
for col in columns:
    x = features.iloc[:, col].values
    x = x.reshape(-1,1)
    encoder = LabelEncoder()
    encoder = encoder.fit(x)
    x = encoder.transform(x)
    features.iloc[:, col] = x 

# features = pd.get_dummies(df)
print(features.shape)
features.head()

# %%
# create the target variable from the yes/no cols then drop yes/no cols

# the 1s in yes remain the same, the 1s in no become 2s, the entries with 0s in both cols remain as 0s.
features['target'] = 0
features.loc[features['yes'] ==1, 'target'] = 1
features.loc[features['no'] ==1, 'target'] = 2
features = features.drop(['yes', 'no'], axis = 1)
# features = features[features.target != 0]
# features.loc[features['target'] ==2, 'target'] = 0
print(features.shape)
features.target.value_counts()

# %%
features.head()

# %%
# dependent and independent variables
x = features.drop(['target'], axis = 1)
y = features[['target']]

# split dataset to train and test sets (90:10)
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = .1, random_state = 0)
print('x train', x_train.shape)
print('y train', y_train.shape)
print('x test', x_test.shape)
print('y test', y_test.shape)

# %%
# get the validation set from the train set (70:20)

# the % changes to 22 to be representative of the 20 expected originally
x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size = .22, random_state = 0)
print('x train', x_train.shape)
print('y train', y_train.shape)
print('x validation', x_val.shape)
print('y validation', y_val.shape)
print('x test', x_test.shape)
print('y test', y_test.shape)

# %%
# create the regressor
regressor = LogisticRegression(solver = 'lbfgs', random_state=42)
regressor.fit(x_train, y_train)

scores = cross_val_score(estimator = regressor, X = x_train, y = y_train, cv = 5)
print(scores)
print("mean Logistic regression score : ", scores.mean())

# %%
# feature importance
feat_imp_dict = dict(zip(x_train.columns, regressor.coef_[0]))
feat_imp = pd.DataFrame.from_dict(feat_imp_dict, orient='index')
feat_imp.rename(columns = {0:'FeatureImportance'}, inplace = True)
feat_imp.sort_values(by=['FeatureImportance'], ascending=False)

# %%
# feature weights for every class
coef_0=regressor.coef_[0]
coef_1=regressor.coef_[1]
coef_2=regressor.coef_[2]
print(coef_0)
print(coef_1)
print(coef_2)

# %%
### XGB
xgb = XGBClassifier(random_state=42, )
xgb.fit(x_train, y_train)

scores = cross_val_score(estimator = xgb, X = x_train, y = y_train, cv = 5)
print(scores)
print("mean xgb score : ", scores.mean())

# %%
# feature importance
feat_imp_dict = dict(zip(x_train.columns, xgb.feature_importances_))
feat_imp_2 = pd.DataFrame.from_dict(feat_imp_dict, orient='index')
feat_imp_2.rename(columns = {0:'FeatureImportance'}, inplace = True)
feat_imp_2.sort_values(by=['FeatureImportance'], ascending=False).head()

# %%
### dt
tree = DecisionTreeClassifier(random_state=42)
tree.fit(x_train, y_train)

scores = cross_val_score(estimator = tree, X = x_train, y = y_train, cv = 5)
print(scores)
print("mean decision trees score : ", scores.mean())

# %%
# feature importance
feat_importance = tree.tree_.compute_feature_importances(normalize=False)
feat_imp_dict = dict(zip(x_train.columns, tree.feature_importances_))
feat_imp_3 = pd.DataFrame.from_dict(feat_imp_dict, orient='index')
feat_imp_3.rename(columns = {0:'FeatureImportance'}, inplace = True)
feat_imp_3.sort_values(by=['FeatureImportance'], ascending=False).head()

# %%
# create accuracies df then plot
data = {'accuracy': [0.8403598319455925 * 100,  0.8403598319455925 * 100,  0.7856773934443837 * 100], 
        'model': ['Logistic Regression' , 'XGB', 'Decision Trees']}
df = pd.DataFrame(data, columns = ['accuracy', 'model'])
# plot
plt.figure(figsize = (6,4))
sns.barplot(y = df.accuracy, x = df.model)
plt.title('barplot indicating model performances')

# %%
# log loss for logistic regression
probabilities = regressor.predict_proba(x_val)
# calculate log loss
loss = log_loss(y_val, probabilities)
loss

# %%
# log loss for xgb
probabilities = xgb.predict_proba(x_val)
# calculate log loss
loss = log_loss(y_val, probabilities)
loss

# %%
# log loss for dt
probabilities = tree.predict_proba(x_val)
# calculate log loss
loss = log_loss(y_val, probabilities)
loss

# %%
# create accuracies df then plot
data = {'loss': [0.519512717164833,  0.5132259370622342], 
        'model': ['Logistic Regression' , 'XGB']}
df = pd.DataFrame(data, columns = ['loss', 'model'])
# plot
plt.figure(figsize = (6,4))
sns.barplot(y = df.loss, x = df.model)
plt.title('barplot indicating loss functions for different models')

# %%
# LR
plt.figure(figsize = (6,4))
sns.barplot(y = feat_imp.FeatureImportance, x = feat_imp.index)
plt.title('Feature Importances in Logistic Regression')
plt.xticks(rotation = 45)

# %%
# XGB
plt.figure(figsize = (6,4))
sns.barplot(y = feat_imp_2.FeatureImportance, x = feat_imp_2.index)
plt.title('Feature Importances in XGB')
plt.xticks(rotation = 45)

# %%
# DT
plt.figure(figsize = (6,4))
sns.barplot(y = feat_imp_3.FeatureImportance, x = feat_imp_3.index)
plt.title('Feature Importances in Decision Trees')
plt.xticks(rotation = 45)

# %%
# using Decision Tree to run predictions on x_test
y_pred = tree.predict(x_test)
a = pd.DataFrame(y_pred)
a.columns = ['pred']
a.pred.value_counts()
