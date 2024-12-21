from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt

data, labels = make_blobs(n_samples=20, centers=2, n_features=2, random_state=0)
data = data.round(2)

print(list(labels))
points = [tuple(point) for point in data]
print(points)

x1_list = data[labels==0, 0]
x2_list = data[labels==1, 0]
y1_list = data[labels==0, 1]
y2_list = data[labels==1, 1]

plt.scatter(x=x1_list, y=y1_list, color='red')
plt.scatter(x=x2_list, y=y2_list, color='green')
plt.show()