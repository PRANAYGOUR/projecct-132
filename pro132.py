from cProfile import label
from math import dist
from tracemalloc import Snapshot
from turtle import color, distance
from cv2 import kmeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans
import seaborn as sns

df = pd.read_csv("final.csv")

X = []

mass = df["Mass"].to_list()
radius = df["Radius"].to_list()
distance = df["Distance"].to_list()
gravity = df["Gravity"].to_list()

mass.sort()
radius.sort()
distance.sort()
gravity.sort()

plt.plot(radius, mass)

plt.title("Radius and mass of the star")
plt.xlabel("Radius")
plt.ylabel("Mass")
plt.show()

plt.scatter(radius , mass)
plt.xlabel("Radius")
plt.ylabel("Mass")
plt.show()

kmeans = KMeans(n_clusters = 4, init = "k-means++" , random_state=42)
y_kmeans = kmeans.fit_predict(X)

print(y_kmeans)

cluster1_x = []
cluster1_y = []
cluster2_x = []
cluster2_y = []
cluster3_x = []
cluster3_y = []
cluster4_x = []
cluster4_y = []

for index , data in enumerate(X):
    if y_kmeans[index] == 0:
        cluster1_x.append(data[0])
        cluster1_y.append(data[1])
    elif y_kmeans[index] == 1:
        cluster2_x.append(data[0])
        cluster2_y.append(data[1])
    elif y_kmeans[index] == 2:
        cluster3_x.append(data[0])
        cluster4_y.append(data[1])
    elif y_kmeans[index] == 3:
        cluster4_x.append(data[0])
        cluster4_y.append(data[1])


plt.figure(figsize=(15 , 7))
sns.scatterplot(cluster1_x , cluster1_y , color = "yellow" , label = "Cluster 1")
sns.scatterplot(cluster2_x , cluster2_y , color = "blue" , label = "Cluster 2")
sns.scatterplot(cluster3_x , cluster2_y , color = "red" , label = "Cluster 3")
sns.scatterplot(cluster4_x , cluster4_y , color = "green" , label= "Cluster 4")
plt.title("Cluster of planets")
plt.xlabel("Planet Radius")
plt.ylabel("Planet Mass")
plt.legend()
plt.gca().invert_yaxis()
plt.show()