import numpy as np
import pandas as pd
from numpy.linalg import inv

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
        #mean = np.mean(b[i])
        #var = np.std(b[i])**2
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
        var = np.std(b[i])**2
        a[i] = (b[i]-mean)/var
        i+=1
    a = a.transpose()
    return a


def error(X, Y, w): #error function calculator
    Error = 0.0
    i=0
    prediction = np.matmul(X,w)
    diff = prediction - Y
    Error = 0.5 * np.matmul(diff.transpose(), diff) / X.shape[0]
    '''
    while i < X.shape[0]:
        x = X[i]
        y = Y[i]
        prediction = np.matmul( w.transpose() , x )
        Error += 0.5 * (prediction - y)**2
        i+=1
    Error = Error/X.shape[0]
    '''
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
        #w_new = np.random.rand(w.shape[0],1) #intermnediate storage of new weights -- useless in vectorized implementation
        '''
        j=0
        #loop over all the weight dimensions
        while j < w.shape[0]:
            grad = 0.0
            grad = np.matmul( (np.matmul(x_train , w) - y_train ).transpose(), x_train .transpose()[j])
            grad = grad/x_train .shape[0]
            w_new[j][0] = w[j][0] - alpha * grad
            j+=1
        w = w_new
        '''
        #vectorized to run in one loop
        grad = np.random.rand(5,1)
        grad = np.matmul(  x_train .transpose() , (np.matmul(x_train , w) - y_train ) )
        grad = grad/x_train .shape[0]
        w = w - alpha * grad
        #print(error(x_test , y_test ,w))
        i+=1

    print('The weight value is: \n' + str(w) )
    print('Error is: ' + str(error(x_test , y_test ,w)) )


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


PartA()
PartB()
