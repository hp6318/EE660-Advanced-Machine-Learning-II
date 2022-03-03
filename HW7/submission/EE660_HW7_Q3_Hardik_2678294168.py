import warnings
warnings.filterwarnings('ignore', category=PendingDeprecationWarning)
from qns3vm import QN_S3VM
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import random
import warnings
warnings.filterwarnings('ignore', category=PendingDeprecationWarning)

#grab the training and testing data.
train_data=pd.read_csv("..\\data\\SSL\\ssl_train_data.csv")
test_data=pd.read_csv("..\\data\\SSL\\test_data.csv")

#change the class label from (0,1) to (-1,1) in order to match the requirements of S3VM
train_data["class"].replace(to_replace=0, value=-1, inplace=True)
test_data["class"].replace(to_replace=0, value=-1, inplace=True)

#extract features and labels from training and testing data
Xtrain=train_data.iloc[:,:10]
Ytrain=train_data.iloc[:,-1]
Xtest=test_data.iloc[:,:10]
Ytest=test_data.iloc[:,-1]

#Q3.a: Train SVC on Xtrain, Ytrain and compute accuracy score of predictions on test data
model_a=SVC(C=1.0, kernel='linear')
model_a=model_a.fit(Xtrain,Ytrain)
Ypred_a=model_a.predict(Xtest)
score_a=accuracy_score(Ytest,Ypred_a)
print("--xxx--Q3.a: Best case, SVM on entire train data--xxx--")
print("Accuracy score: {}".format(score_a))

#Q3.b: Train SVM classifier for the first 2*L training samples by splicing the df and then
#storing the accuracy scores of predictions on test data in 'score_b' array.
L=np.arange(1,11)
score_b=[]
for l in L:
    X_train=Xtrain.iloc[:2*l,:]
    Y_train=Ytrain.iloc[:2*l]
    model_b=SVC(C=1.0, kernel='linear')
    model_b=model_b.fit(X_train,Y_train)
    Ypred_b=model_b.predict(Xtest)
    score=accuracy_score(Ytest,Ypred_b)
    score_b.append(score)
print("--xxx--Q3.b: SVM on 2*L train data--xxx--")
print("Accuracy score: {}".format(score_b))

#Q3.c: Train S3VM classifier for the first 2*L training samples as labeled and rest as unlabeled training
#samples. Then storing the accuracy scores of predictions on test data in 'score_c' array.
my_random_generator = random.Random()
my_random_generator.seed(0)
score_c=[]
for l in L:
    X_train_l=Xtrain.iloc[:2*l,:].values.tolist()
    X_train_u=Xtrain.iloc[2*l:,:].values.tolist()
    Y_train_l=Ytrain.iloc[:2*l].values.tolist()
    model_c=QN_S3VM(X_train_l,Y_train_l,X_train_u,my_random_generator,kernel_type="Linear",lam=1.0)
    model_c.train()
    Ypred_c=model_c.getPredictions(Xtest)
    score=accuracy_score(Ytest,Ypred_c)
    score_c.append(score)
print("--xxx--Q3.c: S3VM on 2*L train data--xxx--")
print("Accuracy score: {}".format(score_c))

#plotting the bar chart for comparing SVM and S3VM
X_axis=np.arange(len(2*L))
plt.bar(X_axis - 0.2, score_b, 0.4, label = 'SVM')
plt.bar(X_axis + 0.2, score_c, 0.4, label = 'S3VM')
plt.xticks(X_axis, 2*L)
plt.xlabel('N_L')
plt.ylabel('accuracy')
plt.legend()
plt.show()

#plotting the curves of accuracy scores of SVM and S3VM
plt.plot(2*L,score_b,'r',label='SVM')
plt.plot(2*L,score_c,'g',label='S3VM')
plt.xticks(2*L)
plt.xlabel('N_L')
plt.ylabel('accuracy')
plt.legend()
plt.show()

#Q3.e: Interpret the results of d
print("\n--xxx--Q3.e.a: Expectation matching")
print("\nIt was expected that S3VM would perform better that SVM in general. This can be reasoned as "
      "making use of unlabeled data to train the model would improve generalization as the model can"
      "learn the patterns better from more data which is drawn from the same density.")
print("\n--xxx--Q3.e.a: Expectation difference")
print("\nIt was expected that the curves would be non-decreasing but that is not the case. It seems like the "
      "quality of labeled data matters, i.e if we have any outliers in labeled data, then generalization"
      "might be affected."
      "Also, SVM wasn't expected to perform better with only 20 labeled data in comparision with S3VM. This"
      "again can be due to the fact that the synthesized data is so arranged that first 20 data points give"
      "richness of data and more outliers in rest unlabeled data which does not help S3VM in generalizing better ")
