import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
 
class Perceptron :
 
    def __init__(self, max_iterations=1000, learning_rate=0.2) :
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
 
    def fit(self, X, y) :
        self.w = np.zeros(len(X[0]))
        converged = False
        iterations = 0
        while (not converged and iterations < self.max_iterations) :
            converged = True
            for i in range(len(X)) :
                if y[i] * self.discriminant(X[i]) <= 0 :
                    self.w = self.w + y[i] * self.learning_rate * X[i]
                    converged = False
            iterations += 1
        self.converged = converged
        if converged :
            print 'converged in %d iterations ' % iterations
        else:
            print 'not converged'
 
    def discriminant(self, x) :
        return np.dot(self.w, x)

    """In sample error which is calculated as the no. of mismatches divided by the total no of training examples"""
    
    def insample_error(self,X,y,w):
        N = len(X)
        mismatch = 0
        for i in range(len(X)) :
            if int(np.sign(np.dot(w,X[i]))) != int(np.sign(y[i])):
                mismatch += 1
        in_error = mismatch / float(N)
        return in_error
        
    """Out sample error is the in sample error calculation performed on the test data"""
    
    def outsample_error(self,X,y,w):
        err=[]
        err.append(p.insample_error(X,y,w))
        return np.mean(err)        

"""Extract the original and scaled data test and training sets from the heart data"""

def get_data (path):
    data=np.genfromtxt(path, delimiter=",", comments="#")
    I = np.arange(270)
    np.random.shuffle(I)
    data = data[I]
    y_train = data[0:170,1]
    y_test = data[170:270,1]
    X_train_temp = data[0:170,2::]
    X_test_temp = data[170:270,2::]
    X_test = np.c_[np.ones(len(X_test_temp[:,0])),X_test_temp]
    X_train = np.c_[np.ones(len(X_train_temp[:,0])),X_train_temp]
    X_train_temp_tr = X_train_temp.T
    X_test_temp_tr = X_test_temp.T
    X_train_scale_tr = np.zeros(shape=(len(X_train_temp_tr),len(X_train_temp_tr.T)))
    X_test_scale_tr = np.zeros(shape=(len(X_test_temp_tr),len(X_test_temp_tr.T)))    
    std_scale = MinMaxScaler((-1,1),copy=True)
    for i in range(len(X_train_temp_tr)):
        scale_arr_a = []        
        scale_arr_a = (std_scale.fit_transform(X_train_temp_tr[i]))
        X_train_scale_tr[i] = scale_arr_a 
    for i in range(len(X_test_temp_tr)):
        scale_arr_b = []        
        scale_arr_b = (std_scale.fit_transform(X_test_temp_tr[i]))
        X_test_scale_tr[i] = scale_arr_b 
    X_train_scale = X_train_scale_tr.T
    X_test_scale = X_test_scale_tr.T
    X_train_scale = np.c_[np.ones(len(X_train_scale[:,0])),X_train_scale]
    X_test_scale = np.c_[np.ones(len(X_test_scale[:,0])),X_test_scale]
    return X_test, X_test_scale, X_train, X_train_scale, y_test, y_train
 
if __name__=='__main__' :
    in_sample_err_without_scaling = []
    out_sample_err_without_scaling = []
    in_sample_err_with_scaling = []
    out_sample_err_with_scaling = []
    for i in range (1,11):
        X_test, X_test_scale, X_train, X_train_scale, y_test, y_train = get_data ("/data/Work_CSU/Course_Work/Dropbox/Sneha_share/Latest_assgn/Final/Data_Normalization/heart.data")
        p = Perceptron()
        p.fit(X_train,y_train)
        in_sample_err_without_scaling.append(p.insample_error(X_train,y_train,p.w))
        out_sample_err_without_scaling.append(p.outsample_error(X_test,y_test,p.w))
        p.fit(X_train_scale,y_train)
        in_sample_err_with_scaling.append(p.insample_error(X_train_scale,y_train,p.w))
        out_sample_err_with_scaling.append(p.outsample_error(X_test_scale,y_test,p.w))
    fig = plt.figure(figsize=(5,5))
    plt.subplot(2,1,1)
    plt.title('Ein with and without scaling')
    plt.plot(in_sample_err_without_scaling, marker='o', label='Ein without scaling',color='r') 
    plt.plot(in_sample_err_with_scaling, marker='*', label='Ein with scaling',color='b')  
    plt.ylabel('Ein')
    plt.subplot(2,1,2)
    plt.plot(out_sample_err_without_scaling, marker='h', label='Eout without scaling',color='g')  
    plt.plot(out_sample_err_with_scaling, marker='d', label='Eout with scaling',color='m') 
    plt.title('Eout with and without scaling')
    plt.ylabel('Ein')
    plt.xlabel('Iterations')   
    plt.show()