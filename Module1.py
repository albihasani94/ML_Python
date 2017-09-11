import sklearn
from sklearn.datasets import load_digits
digits = load_digits();
print (type(digits))
print (digits.data)
print (digits.DESCR)
print(digits.target)
print(digits.target_names)
print(type(digits.data))
print(type(digits.target))
print(type(digits.target_names))
print(digits.data.shape)
print(digits.target.shape)
X = digits.data
y = digits.target
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100)
clf.fit(X,Y)
print('Prediction:'), clf.predict(digits.data[-1])
print('Actual:'), y[-1]