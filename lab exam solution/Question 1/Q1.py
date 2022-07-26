import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt


#setting random seed to get the point fixed
np.random.seed(40)

#setting the k-mean value for getiing optimal answers
K = 3

columns = []
columns.append("label")
for i in range(1,14):
    columns.append(i)

#fitering data
c2 = ["label"]
df = pd.read_csv("wine.data",delimiter = ",",names = columns,header = None)


#splitting the data into training and testing modules according to given percentages 70-30
x_train, X_test, y_train, y_test = train_test_split(df[columns[1:]], df[columns[0]], test_size=0.30, random_state=42)

X = np.array(x_train[[1,2,3,4,5,6,7,8,9,10,11,12,13]])





def euclidian_distance(query,X):
        difference = np.array(X) - np.array(query)
        sqrd_diff = np.square(difference)
        sum_sqrd_diff = np.sum(sqrd_diff, axis = 1)
        distance = np.sqrt(sum_sqrd_diff)
        # print("D: ",distance)
        return distance




#setting the random cent
centroidIndex = np.random.randint(0,125,(K,))
# print(centroidIndex.shape)
cent = X[centroidIndex]
# print(cent)
# print(cent)


for i in range(100):
    
    clust = [[],[],[],[]]

    for x in X:
        id = np.argmin(euclidian_distance(x,cent))
        
        clust[id].append(x)

    c = np.array(clust, dtype = object)

    for x in range(K):
        cent[x] = np.mean(c[x], axis = 0)
        
print("lenc1",len(clust[0]),"lenc2",len(clust[1]),"lenc3",len(clust[2]))
print("Model Trained")

predictions = []
for x in X_test.T:
    temp = euclidian_distance(x,cent)
    # print(temp)
    temp = np.argmin(temp)
    predictions.append(temp+1)


acc = accuracy_score(np.array(predictions), np.array(y_test))
print("Accuracy of the model trained : = ", acc)
