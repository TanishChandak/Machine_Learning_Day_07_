import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import seaborn as sns

# the dataset is inbuilt in the sklearn:
digits = load_digits()
# having the element like target, data, images, target_names etc:
print(dir(digits))
# printing the first number for the dataset:
print(digits.data[0])

# printing the image as the number is given:
plt.gray()
# for i in range(5):
#     plt.matshow(digits.images[i])
#     plt.show()

# printing the target value for the digits dataset:
print(digits.target[0:5])

# training and testing the model:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)
# print(len(X_train))
# print(len(X_test))

# Logistic Regression:
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# training the data:
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(model.predict([digits.data[0:5]]))

# confusion matrix:
y_predicted = model.predict(X_test)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_predicted)
print(cm)

plt.figure(figsize=(10,7))
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()