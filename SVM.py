import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn.externals import joblib

#Loading data from the dataset
cancer = datasets.load_breast_cancer()

#Splitting the data into input and output
x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)
#Using SVM with a linear kernel 
clf=svm.SVC(kernel="linear", 

#training the model
clf.fit(x_train,y_train)

#using the model
y_pred=clf.predict(x_test)

#checking the accuracy
acc=metrics.accuracy_score(y_test,y_pred)

#Saving the trained model in an external file

joblib.dump(clf, 'trained_model.joblib')

