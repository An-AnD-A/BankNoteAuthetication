# -*- coding: utf-8 -*-
"""
Created on Sun Aug 15 17:28:25 2021

@author: anand
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats  as stats
from sklearn import preprocessing

data = pd.read_csv("C:/analytics/datasets/Bank Note Authetication dataset/Banknote_authentication_dataset_red.csv")

data_ful = pd.read_csv("C:/analytics/datasets/Bank Note Authetication dataset/Banknote_authentication_dataset.csv")

data_ful.drop(["V3","V4"], axis =1,inplace = True)
data_ful["Class"] = data_ful["Class"].map({1:0,2:1})

#%% Exploring the data

print(data.info())
print(data.head())
print(data.describe())
print(data.isnull().sum())

sns.heatmap(data.corr(),annot = True)
plt.show()

plt.scatter(data["V1"],data["V2"],s = 3)
plt.show()

sns.kdeplot(data["V1"])
plt.show()
sns.kdeplot(data["V2"])
plt.show()
#%% Checking for normal distribution

sm.qqplot(data["V1"],line = "45")
sm.qqplot(data["V2"],line = "45")

stats.shapiro(data["V1"])
stats.shapiro(data["V2"])

# distribution is assumed non guassian as p value is less than 0.5

# normalize the data

norm_scalar = preprocessing.MinMaxScaler()
norm_data = norm_scalar.fit_transform(data)

plt.scatter(norm_data[:,0],norm_data[:,1],s = 3)
plt.show()

#%% Clustering

from sklearn.cluster import KMeans

fig,ax = plt.subplots(3,3, figsize = (16,16))
ax = np.ravel(ax)



for i in range(0,6):
    
    crct_label_km = 0
    
    Kmeans = KMeans(n_clusters = 2,init="random",n_init = 10)
    Kmeans.fit(data)
    
    cent_km = Kmeans.cluster_centers_
    
    data["Kmeans Labels"] = Kmeans.labels_
    
    label_km0 = data[data["Kmeans Labels"] == 0]
    label_km1 = data[data["Kmeans Labels"] == 1]
    
    label_km0.reset_index(drop = True,inplace = True)
    label_km1.reset_index(drop = True,inplace = True)
    
    ax[i].scatter(label_km0.V1,label_km0.V2,s = 10,c = "c")
    ax[i].scatter(label_km1.V1,label_km1.V2,s = 10)
    ax[i].scatter(cent_km[:,0], cent_km[:,1],c = "r",marker = "*", s = 70)
    ax[i].set_title("K means (run No. : {} )".format(i + 1))
    
    for j in range(0,len(data_ful)):
        if data["Kmeans Labels"][j] == data_ful["Class"][j]:
            crct_label_km +=1

    print("Correct Label for Run {} :".format(i) , crct_label_km)
    print(crct_label_km/len(data_ful))
    
    if i == 5:
        
        crct_label_kmp = 0
        
        data_kmp = pd.read_csv("C:/analytics/datasets/Bank Note Authetication dataset/Banknote_authentication_dataset_red.csv")
        
        Kmeans_kmp = KMeans(n_clusters = 2,init="k-means++")
        Kmeans_kmp.fit(data_kmp)

        cent_kmp = Kmeans_kmp.cluster_centers_

        data_kmp["Kmeans Labels"] = Kmeans_kmp.labels_

        label_kmp0 = data_kmp[data_kmp["Kmeans Labels"] == 0]
        label_kmp1 = data_kmp[data_kmp["Kmeans Labels"] == 1]

        label_kmp0.reset_index(drop = True,inplace = True)
        label_kmp1.reset_index(drop = True,inplace = True)
        
        ax[i+2].scatter(label_kmp0.V1,label_kmp0.V2,s = 10,c = "c")
        ax[i+2].scatter(label_kmp1.V1,label_kmp1.V2,s = 10)
        ax[i+2].scatter(cent_kmp[:,0], cent_kmp[:,1],c = "r",marker = "*", s = 70)
        ax[i+2].set_title("K-Means ++")
        
        ax[i+1].set_axis_off()
        ax[i+3].set_axis_off()
        
        for j in range(0,len(data_ful)):
            if data_kmp["Kmeans Labels"][j] == data_ful["Class"][j]:
                crct_label_kmp +=1


print("Correct Label for Run K_means++ : " , crct_label_kmp)
print(crct_label_kmp/len(data_ful))

plt.tight_layout()
plt.show()

#%% Accuracy

# The accuracy of the clustering was found to be at 65%.


#%% Clustering using custom centers.

# =============================================================================
# 
# V1_min = data[data["V1"] == data["V1"].min()].drop("Kmeans Labels", axis = 1)
# V1_max = data[data["V1"] == data["V1"].max()].drop("Kmeans Labels", axis = 1)
# 
# points = [V1_min,V1_max]
#  
# cust_cent = pd.concat(points)
# cust_cent.reset_index(drop = True)
# cust_cent = cust_cent.to_numpy()
# 
# data_kmc = pd.read_csv("C:/analytics/datasets/Bank Note Authetication dataset/Banknote_authentication_dataset_red.csv")
# crct_label_kmc = 0
#     
# Kmeans_kmc = KMeans(n_clusters = 2,init=cust_cent)
# Kmeans_kmc.fit(data_kmc)
# 
# print(Kmeans_kmc.cluster_centers_)
# cent_kmc = cust_cent
# 
# data_kmc["Kmeans Labels"] = Kmeans_kmc.labels_
# 
# label_kmc0 = data_kmc[data_kmc["Kmeans Labels"] == 0]
# label_kmc1 = data_kmc[data_kmc["Kmeans Labels"] == 1]
# 
# label_kmc0.reset_index(drop = True,inplace = True)
# label_kmc1.reset_index(drop = True,inplace = True)
# 
# plt.scatter(label_kmc0.V1,label_kmc0.V2,s = 10,c = "c")
# plt.scatter(label_kmc1.V1,label_kmc1.V2,s = 10)
# plt.scatter(cent_kmc[:,0], cent_kmc[:,1],c = "r",marker = "*", s = 70)
# plt.title("K means (custom centers)")
# plt.show()
# 
# =============================================================================
