#D:\Study_In_Office\Machine_Learning_Algorithm

'''
Algorithm Name - Kmeans Algorithm
Algorithm Learning Type - Unsupervised Learning(Means We Don't Know What is our Target Variable)

Mistakes - It's Calculate Distance Between Centroid And Target Variable So Our Point Should Be Closer
'''

from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

#Read The DataSets

df = pd.read_csv("D:/Study_In_Office/Machine_Learning_Algorithm/income.csv")
df.columns = ['Name','Age', 'Income']

df = df.drop('Name',axis='columns')

#Plot Visualization

plt.scatter(df.Age,df.Income)
plt.xlabel("Age")
plt.ylabel("Income")

#Clearly see three clusters in our Datasets but Uses Elbo Method

#---------------------------------------------Kmeans Starts From Here
km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age', 'Income']])

df['cluster'] = y_predicted

#Clusters Centroid Position

km.cluster_centers_

#Visualization Part

df1 = df[df.cluster ==0]
df2 = df[df.cluster ==1]
df3 = df[df.cluster ==2]

plt.scatter(df1.Age,df1.Income,color = 'green')
plt.scatter(df2.Age,df2.Income,color = 'red')
plt.scatter(df3.Age,df3.Income,color = 'Black')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color = 'Black',marker ='*',label = 'centroid')

plt.xlabel("Age")
plt.ylabel("Income")


#So Problem is We calculate Distance Then It Will So some Sort of Anamolies

scaler = MinMaxScaler()
scaler.fit(df[['Income']])
df['Income'] = scaler.transform(df[['Income']])

scaler.fit(df[['Age']])
df['Age'] = scaler.transform(df[['Age']])


#Apply Kmeans Algorithm Again


km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(df[['Age', 'Income']])

df['cluster'] = y_predicted

#Clusters Centroid Position

km.cluster_centers_

#Visualization Part

df1 = df[df.cluster ==0]
df2 = df[df.cluster ==1]
df3 = df[df.cluster ==2]

plt.scatter(df1.Age,df1.Income,color = 'green')
plt.scatter(df2.Age,df2.Income,color = 'red')
plt.scatter(df3.Age,df3.Income,color = 'Black')

plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color = 'Black',marker ='*',label = 'centroid')

plt.xlabel("Age")
plt.ylabel("Income")

#Beautiful You Resolve This Issue

#Elbow Tecniques for Determine Best Cluster

sse = []

k_rng = range(1,7)

for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit_predict(df[['Age', 'Income']])
    sse.append(km.inertia_)
    
#Visaulize The Results
    
plt.xlabel("K")
plt.ylabel("Sum of Sqaure Error")
plt.plot(k_rng,sse)

#So Best Value Of K Is 3























