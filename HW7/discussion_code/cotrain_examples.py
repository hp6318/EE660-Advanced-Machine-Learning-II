'''
EE660
Adapted from https://github.com/jjrob13/sklearn_cotraining
'''
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.datasets import make_classification
from cotrain_classifiers import CoTrainingClassifier

def example1():
	N_SAMPLES = 25000
	N_FEATURES = 1000
	X, y = make_classification(n_samples=N_SAMPLES, n_features=N_FEATURES)

	y[:N_SAMPLES//2] = -1

	X_test = X[-N_SAMPLES//4:]
	y_test = y[-N_SAMPLES//4:]

	X_labeled = X[N_SAMPLES//2:-N_SAMPLES//4]
	y_labeled = y[N_SAMPLES//2:-N_SAMPLES//4]

	y = y[:-N_SAMPLES//4]
	X = X[:-N_SAMPLES//4]


	X1 = X[:,:N_FEATURES // 2]
	X2 = X[:, N_FEATURES // 2:]

	print("\nSynthetic data set instance")
	print("Feature dimension: ",N_FEATURES)
	print("Number of labeled patterns: ", X_labeled.shape[0])
	print("Number of unlabeled patterns: ", N_SAMPLES//2)
	print("Number of test patterns: ", X_test.shape[0])

	print ('Logistic')
	base_lr = LogisticRegression()
	base_lr.fit(X_labeled, y_labeled)
	y_pred = base_lr.predict(X_test)
	print (classification_report(y_test, y_pred))

	print ('Logistic CoTraining')
	lg_co_clf = CoTrainingClassifier(LogisticRegression())
	lg_co_clf.fit(X1, X2, y)
	y_pred = lg_co_clf.predict(X_test[:, :N_FEATURES // 2], X_test[:, N_FEATURES // 2:])
	print (classification_report(y_test, y_pred))
	return

import numpy as np
import time
import random
import examples
def example2():
	my_random_generator = random.Random()
	my_random_generator.seed(0)
	# dense moons data set
	X_train_l, L_train_l, X_train_u, X_test, L_test = examples.get_moons_data(my_random_generator)
	X_test = np.array(X_test)
	L_train_l = (np.array(L_train_l)+1)//2
	L_test = (np.array(L_test)+1)//2
	print("Unique labels: ",np.unique(L_test))

	X_train = np.concatenate((X_train_l,X_train_u),axis=0)
	y_train = np.hstack((L_train_l,-np.ones(len(X_train_u))))
	X1 = X_train[:,0].reshape((-1,1))
	X2 = X_train[:,1].reshape((-1,1))

	print ('Logistic')
	base_lr = LogisticRegression()
	base_lr.fit(X_train_l, L_train_l)
	#base_lr.fit(X_test, L_test)
	y_pred = base_lr.predict(X_test)
	print (classification_report(L_test, y_pred))

	print ('Logistic CoTraining')
	t_start = time.time()
	lg_co_clf = CoTrainingClassifier(LogisticRegression())
	lg_co_clf.fit(X1, X2, y_train)
	t_end = time.time()
	y_pred_train = lg_co_clf.predict(X_train[:,0].reshape((-1,1)), X_train[:,1].reshape((-1,1)))
	y_pred = lg_co_clf.predict(X_test[:,0].reshape((-1,1)), X_test[:,1].reshape((-1,1)))
	print (classification_report(L_test, y_pred))
	return

if __name__ == '__main__':
	example2()	
	example1()
	
	
