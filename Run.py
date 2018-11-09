import numpy as np
import pandas as pd
from numpy.linalg import inv
import math
import matplotlib.pyplot as plt

''' Function declarations '''
def standardize2( b ): #function for normalization. min-max method
    b = b.transpose()
    a = np.empty( b.shape )
    i = 0
    for x in b:
        max = np.amax(x)
        min = np.amin(x)
        range = max - min
        if range == 0:
            a[i] = np.full( b[i].shape , 1 )
            i+=1
            continue
        a[i] = (b[i]-min)/range
        i+=1
    a = a.transpose()
    return a

def standardize( b ): #function for normalization. mean-variance method
    b = b.transpose()
    a = np.empty( b.shape )
    i = 0
    for x in b:
        mean = np.mean(b[i])
        std = np.std(b[i])
        a[i] = (b[i]-mean)/std
        i+=1
    a = a.transpose()
    return a


def error(X, Y, w): #error function calculator
    Error = 0.0
    i=0
    prediction = np.matmul(X,w)
    diff = prediction - Y
    Error = 0.5 * np.matmul(diff.transpose(), diff) / X.shape[0]
    return Error


def ridgeRegression(X, Y, lam, X_cv, Y_cv):
    w = np.random.rand(5,1)

    #hard coded parameters
    numOfIters = 100000
    alpha = 0.01
    #gradient decent
    i = 0
    while i<numOfIters:
        #vectorized to run in one loop
        grad = np.random.rand(5,1)
        grad = np.matmul(X.transpose() , (np.matmul(X , w) - Y ) ) + 2*lam*w
        grad = grad / x_train.shape[0]
        w = w - alpha * grad
        i+=1

    wridge = w
    Error = error(X_cv , Y_cv ,w)
    return Error


def lassoRegression(X, Y, lam, X_cv, Y_cv):
    w = np.random.rand(5,1)

    #hard coded parameters
    numOfIters = 100000
    alpha = 0.01
    #gradient decent
    i = 0
    while i<numOfIters:
        #vectorized to run in one loop
        grad = np.random.rand(5,1)
        grad = np.matmul(X.transpose() , (np.matmul(X , w) - Y ) ) + lam * np.divide(w, np.abs(w))
        grad = grad / x_train.shape[0]
        w = w - alpha * grad
        i+=1

    wlasso = w
    Error = error(X_cv , Y_cv ,w)
    return Error

def PartA(): #Part A of assignment.
    print('*****Normal Equations method*****')
    w = np.matmul( np.matmul( inv( np.matmul( x_train.transpose(), x_train ) ) , x_train.transpose() ) , y_train ) #performing w = inv(xTx)xy

    Error = error(x_test, y_test, w)

    print('The weight value is: \n' + str(w) )
    print('Error is: ' + str(Error) )

def PartB(): #Part B of assignment
    print('*****Gradient Descent method*****')
    #random initialization of weights in [0,1)
    w = np.random.rand(5,1)

    #hard coded parameters
    numOfIters = 100000
    alpha = 0.01

    #gradient decent
    i = 0
    while i<numOfIters:
        #vectorized to run in one loop
        grad = np.random.rand(5,1)
        grad = np.matmul(  x_train .transpose() , (np.matmul(x_train , w) - y_train ) )
        grad = grad/x_train .shape[0]
        w = w - alpha * grad
        #print(error(x_test , y_test ,w))
        i+=1

    print('The weight value is: \n' + str(w) )
    print('Error is: ' + str(error(x_test , y_test ,w)) )

def PartC(): #Part C of the assignment
    '''preprocess data again because of new cv data'''
    data_np1 = standardize(data_np)

    train_data = data_np1[:int(data_np.shape[0]*0.6)]
    cv_data = data_np1[int(data_np.shape[0]*0.6)+1:int(data_np.shape[0]*0.8)]
    test_data = data_np1[int(data_np.shape[0]*0.8)+1:]

    x_train = train_data[:,:-1]
    y_train = train_data[:,4]
    y_train = np.reshape(y_train , (y_train.shape[0], 1))

    x_cv = cv_data[:,:-1]
    y_cv = cv_data[:,4]
    y_cv = np.reshape(y_cv , (y_cv.shape[0], 1))

    x_test = test_data[:,:-1]
    y_test = test_data[:,4]
    y_test = np.reshape(y_test , (y_test.shape[0], 1))

    x_train = np.concatenate( ( np.full( (x_train .shape[0],1) , 1) ,x_train ) ,axis = 1)
    x_cv = np.concatenate( ( np.full( (x_cv .shape[0],1), 1) ,x_cv ) ,axis = 1)
    x_test = np.concatenate( ( np.full( (x_test .shape[0],1), 1) ,x_test ) ,axis = 1)

    '''loop over all the lambda values'''
    print('***** Ridge and lasso Regression ******')
    loglambda = np.array([-5, -4, -3, -2, -1, 0, 1, 2])
    ridgeErrors = np.zeros((loglambda.shape[0],1))
    lassoErrors = np.zeros((loglambda.shape[0],1))
    lam = 0.0
    i=0
    #iterate over all teh log lambda values to see which one is the best
    for loglam in loglambda:
        lam = math.pow(10, loglam)
        #perform ridge regression
        e = ridgeRegression(x_train, y_train, lam, x_cv, y_cv)
        ridgeErrors[i][0] = e
        #perform lasso regression
        e = lassoRegression(x_train, y_train, lam, x_cv, y_cv)
        lassoErrors[i][0] = e
        i+=1


    #plot graphs for each
    plt.plot(loglambda, ridgeErrors, 'r--', label = 'ridge regression errors')#, lassoErrors, 'bs')
    plt.plot(loglambda, lassoErrors, 'b--', label = 'lasso regression errors')
    plt.legend()
    plt.xlabel('log lambda')
    plt.ylabel('cross-validation error')
    plt.title('CV Errors with different lambda values')

    #save the graph
    plt.savefig('Errors_lambda.png', format = "png")

    #get the best lambda and report test errors
    i=0
    minimum = float("inf")
    optimumLambda = 0
    while i<loglambda.shape[0]:
        if ridgeErrors[i][0]<minimum:
            minimum = ridgeErrors[i][0]
            optimumLambda = math.pow(10, loglambda[i])
        i+=1

    print("The optimum value of lambda is: " + str(optimumLambda))
    #perform ridge regression
    e = ridgeRegression(x_train, y_train, lam, x_test, y_test)
    print('The weight value is: \n' + str(wridge) )
    print("The test error in ridge regression is: " +  str(e))
    #perform lasso regression
    e = lassoRegression(x_train, y_train, lam, x_test, y_test)
    print('The weight value is: \n' + str(wlasso) )
    print("The test error in lasso regression is: " +  str(e))


    #show the graph in the end
    plt.show()

'''Run part'''


''' Data Pre-processing part '''
#data = np.loadtxt( 'CCPP/Folds5x2_pp.xlsx', delimiter=',') #tried to use numpy. Didnt work because of unicode encoding issues.

#get the excel file to pandas dataframe
data = pd.read_excel('CCPP/Folds5x2_pp.xlsx')

#convert it into a numpy array
data_np = data.as_matrix()

#train-test split
train_data = data_np[:int(data_np.shape[0]*0.8)]
test_data = data_np[int(data_np.shape[0]*0.8)+1:]

#x-y split
x_train = train_data[:,:-1]
y_train = train_data[:,4]
y_train = np.reshape(y_train , (y_train.shape[0], 1))

x_test = test_data[:,:-1]
y_test = test_data[:,4]
y_test = np.reshape(y_test , (y_test.shape[0], 1))

#standardize the data and concatenate x0
x_train  = standardize(x_train)
x_test  = standardize(x_test)
y_train  = standardize(y_train)
y_test  = standardize(y_test)

x_train = np.concatenate( ( np.full( (x_train .shape[0],1) , 1) ,x_train ) ,axis = 1)
x_test = np.concatenate( ( np.full( (x_test .shape[0],1), 1) ,x_test ) ,axis = 1)

#weights results of ridge and lasso regression
wridge = np.random.rand(5,1)
wlasso = np.random.rand(5,1)

#run all parts of the assignment
PartA()
PartB()
PartC()
