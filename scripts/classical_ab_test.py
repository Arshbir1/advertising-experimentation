"""Executable analysis script generated from the corresponding notebook.

Run from the repository root after placing ABAdRecall.csv in data/.
"""

# %%
# data preprocessing and linear algebra
import pandas as pd
import numpy as np

# plotting libraries
import seaborn as sns
import matplotlib.pyplot as plt

# statistics and math
from scipy import stats
from scipy.stats import skew, norm
import math


# others
import datetime

# ignore warnings
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
# Device makes in each group 
users1 = data.loc[data.experiment == 'exposed']
users2 = data.loc[data.experiment == 'control']
dev_e = users1.device_make.nunique()
dev_c = users2.device_make.nunique()
print('number of unique devices in exposed group:', dev_e)
print('number of unique devices in control group:', dev_c)


# os_platforms in each group 
users1 = data.loc[data.experiment == 'exposed']
users2 = data.loc[data.experiment == 'control']
os_e = users1.platform_os.nunique()
os_c = users2.platform_os.nunique()
print('number of unique os platforms in exposed group:', os_e)
print('number of unique os platforms in control group:', os_c)


# browser types in each group 
users1 = data.loc[data.experiment == 'exposed']
users2 = data.loc[data.experiment == 'control']
b_e = users1.browser.nunique()
b_c = users2.browser.nunique()
print('number of unique browser in exposed group:', b_e)
print('number of unique browser in control group:', b_c)

# total number of users in each  group
users1 = data.loc[data.experiment == 'exposed']
users2 = data.loc[data.experiment == 'control']
u_e = users1.auction_id.nunique()
u_c = users2.auction_id.nunique()
print('total number of users in exposed group:', u_e)
print('total number of users in control group:', u_c)

# lets get the daily rates
# Average total number of users on a daily basis (N)
data.date = pd.to_datetime(data["date"])  #convert date to datetime object
data['day'] = data.date.dt.day    #extract the days
users = data[['auction_id', 'day']]
a = users.groupby('day').count()   #aggregate users per day
N = a.auction_id.mean()
print('Average number of total users per day:', N)  #average users per day


# Average number of users shown the smartad per day(E)
users = data[['auction_id', 'day', 'experiment']]
users = users.loc[users.experiment == 'exposed']
b = users.groupby('day').count()   #aggregate users per day
E = b.auction_id.mean()
print('Average number of users shown the ad per day:', E)  #average per day (we're expecting around half of the total users value)


# gross conversion (exposed/total users)
print('the gross conversion rate is:', E/N)

# Retention (proportion of Users who recall the ad (yes = 1) after being shown)
users = data[['auction_id', 'yes', 'experiment', 'day']]
users = users.loc[users.experiment == 'exposed']
users = users.loc[users.yes == 1]
c = users.groupby('day').count() 
d = c.auction_id.mean()
R = d/E
print('proportion of Average daily users who recall the ad after being shown the ad (Retention):', R)


# Net conversion (proportion of Users who recal the ad to total users generally)
print('the net conversion rate:',d/N )

# %%
# storing the values in a dictionary
baseline = {"users":1010,"exposed":501,"GConversion":0.49598,
           "Retention":0.07688,"NConversion":0.038133}

# %%
# scaling baseline counts
baseline["users"] = 200
baseline["exposed"]=int(baseline["exposed"]*(200/1010))
baseline

# %%
# calculating the sd
# GCOnversion
GC={}
GC["d_min"]=0.01
GC["p"]=baseline["GConversion"]
GC["n"]=baseline["users"]
GC["sd"]=round(math.sqrt((GC["p"]*(1-GC["p"]))/GC["n"]),5)
GC["sd"]

# %%
# calculating the sd
# Retention
R={}
R["d_min"]=0.01
R["p"]=baseline["Retention"]
R["n"]=baseline["exposed"]
R["sd"]=round(math.sqrt((R["p"]*(1-R["p"]))/R["n"]),5)
R["sd"]

# %%
# calculating the sd
# Net COnversion
NC={}
NC["d_min"]=0.01
NC["p"]=baseline["NConversion"]
NC["n"]=baseline["users"]
NC["sd"]=round(math.sqrt((NC["p"]*(1-NC["p"]))/NC["n"]),5)
NC["sd"]

# %%
#z_score, critical value and standard deviation

#Inputs: alpha value.
#Returns: z-score for given alpha
def get_z_score(alpha):
    return norm.ppf(alpha)

# Inputs: p-baseline conversion rate which is our estimated p and d-minimum
# Returns: standard deviations list
def get_sds(p,d):
    sd1=math.sqrt(2*p*(1-p))
    sd2=math.sqrt(p*(1-p)+(p+d)*(1-(p+d)))
    sds=[sd1,sd2]
    return sds

# Inputs:sd1-sd for the baseline,sd2-sd for the expected change,alpha,beta,d-d_min,p-baseline estimate p
# Returns: the minimum sample size required per group according to metric denominator
def get_sampSize(sds,alpha,beta,d):
    n=pow((get_z_score(1-alpha/2)*sds[0]+get_z_score(1-beta)*sds[1]),2)/pow(d,2)
    return n

# %%
#sample size per metric

#GConversion
# dmin
GC["d"]=0.01

GC["SampSize"]=round(get_sampSize(get_sds(GC["p"],GC["d"]),0.05,0.1,GC["d"]))
GC["SampSize"] * 2   #for the 2 groups

# %%
# Retention

R["d"]=0.01

R["SampSize"]=round(get_sampSize(get_sds(R["p"],R["d"]),0.05,0.1,R["d"]))
R["SampSize"] * 0.49598 * 2  # convert to total users then get the number for the 2 groups

# %%
# Net Conversion
NC["d"]=0.01

NC["SampSize"]=round(get_sampSize(get_sds(NC["p"],NC["d"]),0.05,0.1,NC["d"]))
NC["SampSize"] * 2  #for the 2 groups

# %%
# print out the first 2 rows
data.head(2)

# %%
# split data into control/exposed groups then agg by day

exposed = data.loc[data.experiment == 'exposed']
control = data.loc[data.experiment == 'control']

df_e = exposed.groupby('day').agg({'auction_id':'count', 'device_make':'count', 'platform_os':'count', 'browser':'count', 'yes':'sum', 'no':'sum'})
df_c = control.groupby('day').agg({'auction_id':'count', 'device_make':'count', 'platform_os':'count', 'browser':'count', 'yes':'sum', 'no':'sum'})

print(df_e)
print(df_c)

# %%
# total number of users in both groups (checking for a significant difference)
print('exposed users', df_e.auction_id.sum())
print('control users', df_c.auction_id.sum())

# %%
# the difference is slight
# making sure the difference is not significant and is random

a = df_c.auction_id.sum()
b = df_e.auction_id.sum()
c = a + b

p=0.5  #taking equal possibilities
alpha=0.05   #significance level
p_hat=round(a/(c),4)
sd=math.sqrt(p*(1-p)/(c))
ME=round(get_z_score(1-(alpha/2))*sd,4)
print("The confidence interval is between",p-ME,"and",p+ME,"; Is",p_hat,"inside this range?")

# %%
# Average total number of users on a daily basis (N)
print('Average number of total users per day (both groups):', N)  #average per day
e = df_e.auction_id.mean()
c = df_c.auction_id.mean()
print('Average number of total users per day in the exposed roup:', e)
print('Average number of total users per day in the control group:', c)

# %%
# making sure the difference is not significant and is random

a = e
b = c
d = a + b

p=0.5  #taking equal possibilities
alpha=0.05   #significance level
p_hat=round(a/(d),4)
sd=math.sqrt(p*(1-p)/(d))
ME=round(get_z_score(1-(alpha/2))*sd,4)
print("The confidence interval is between",p-ME,"and",p+ME,"; Is",p_hat,"inside this range?")

# %%
print(dev_c)
print(dev_e)

# %%
# making sure the difference is not significant and is random

a = dev_c
b = dev_e
c = a + b

p=0.5  #taking equal possibilities
alpha=0.05   #significance level
p_hat=round(a/(c),4)
sd=math.sqrt(p*(1-p)/(c))
ME=round(get_z_score(1-(alpha/2))*sd,4)
print("The confidence interval is between",p-ME,"and",p+ME,"; Is",p_hat,"inside this range?")

# %%
print(os_c)
print(os_e)

# %%
# making sure the difference is not significant and is random

a = os_c
b = os_e
c = a + b

p=0.5  #taking equal possibilities
alpha=0.05   #significance level
p_hat=round(a/(c),4)
sd=math.sqrt(p*(1-p)/(c))
ME=round(get_z_score(1-(alpha/2))*sd,4)
print("The confidence interval is between",p-ME,"and",p+ME,"; Is",p_hat,"inside this range?")

# %%
print(b_c)
print(b_e)

# %%
# making sure the difference is not significant and is random

a = b_c
b = b_e
c = a + b

p=0.5  #taking equal possibilities
alpha=0.05   #significance level
p_hat=round(a/(c),4)
sd=math.sqrt(p*(1-p)/(c))
ME=round(get_z_score(1-(alpha/2))*sd,4)
print("The confidence interval is between",p-ME,"and",p+ME,"; Is",p_hat,"inside this range?")

# %%
#Net Conversion - number of users who recall the ad divided by total users
recall_c=df_c["yes"].sum()
recall_e=df_e["yes"].sum()

a = df_c.auction_id.sum()
b = df_e.auction_id.sum()
c = a + b

NC_cont=recall_c/a
NC_exp=recall_e/b
NC_pooled=(recall_c+recall_e)/(a+b)
NC_sd_pooled=math.sqrt(NC_pooled*(1-NC_pooled)*(1/a+1/b))
NC_ME=round(get_z_score(1-alpha/2)*NC_sd_pooled,4)
NC_diff=round(NC_exp-NC_cont,4)
print("The change due to the experiment is",NC_diff*100,"%")
print("Confidence Interval: [",NC_diff-NC_ME,",",NC_diff+NC_ME,"]")
print ("The change is statistically significant if the CI doesn't include 0. In that case, it is practically significant if",-NC["d_min"],"is not in the CI as well.")
