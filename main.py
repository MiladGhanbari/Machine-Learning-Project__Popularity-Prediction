#################################
##Importing necessary libraries##
import json 
import numpy as np
import Preprocessing as pr
import LinearRegression as lin
from time import time
################################
##Task 1: Dataset Construction##

with open("proj1_data.json") as fp:
    data = json.load(fp)

Training_data = data[0:10000]
Validation_data = data[10000:11000]
Test_data = data[11000:12000]

list_word, wordtext, allwords = pr.TextProcess(Training_data, 60)

with open("words.text" , 'w') as txtfile:
    for i in range(len(wordtext)):
        txtfile.write(wordtext[i][0]+ " " + str(wordtext[i][1]) + "\n")


x_train, y_train = pr.DatasetConstruct(Training_data, list_word , 68)
x_val, y_val = pr.DatasetConstruct(Validation_data, list_word , 68)
x_test, y_test = pr.DatasetConstruct(Test_data, list_word , 68)
#########################################
##Task 2: Linear Regression Forms##
##Gradient Descent
t = time()
net_g, error_t = lin.GradientDescent(x_train, y_train, eps = 0.00001,
                                    eta=0.00000001, beta= 0.000001)
run = time() - t
print("Runtime: {}".format(run))
y_predtr = np.matmul(x_train,net_g)
y_predv = np.matmul(x_val,net_g)
y_predte = np.matmul(x_test,net_g)

acc_t = lin.mse(y_train,y_predtr)
acc_v = lin.mse(y_val,y_predv)
acc_te = lin.mse(y_test,y_predte)

print("Training MSE of gradient descent is: {}".format(acc_t))
print("Validation MSE of gradient descent is: {}".format(acc_v))
print("Test MSE of gradient descent is: {}".format(acc_te))   
###################################
##Closed Form
t = time()
net = lin.ClosedForm(x_train,y_train)
run = time() - t
print()
print("Runtime: {}".format(run))
y_predtr = np.matmul(x_train,net)
y_predv = np.matmul(x_val,net)
y_predte = np.matmul(x_test,net)

acc_t = lin.mse(y_train,y_predtr)
acc_v = lin.mse(y_val,y_predv)
acc_te = lin.mse(y_test,y_predte)

print("Training MSE of closed form is: {}".format(acc_t))
print("Validation MSE of closed form is: {}".format(acc_v))
print("Test MSE of closed form is: {}".format(acc_te))
#################################