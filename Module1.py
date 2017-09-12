#import the dataset from sklearn and declare the dataset
from sklearn.datasets import load_digits
digits = load_digits();

#check out the type and data for digits
print (type(digits))
print (digits.data)

#check out the description of this dataset for more information
print (digits.DESCR)

#see the categories that classify each of the images by invoking the target field
print(digits.target)

#print out the target_names, so we can find out what the data is categorized as
print(digits.target_names)

#the data is stored as a numpy datatype
print(type(digits.data))
print(type(digits.target))
print(type(digits.target_names))

#shape of the data
print(digits.data.shape)
print(digits.target.shape)

#declare variables for the data and target which will be used to fit (train) the machine
X = digits.data
y = digits.target

#import svm and declare a variable called clf with gamma and C attributes. Now we can fit our model and predict the last digit, which should be 8
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100)
clf.fit(X,y)
print('Prediction:'), clf.predict(digits.data[-1])
print('Actual:'), y[-1]