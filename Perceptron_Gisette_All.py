import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
 
class Perceptron :
 
    def __init__(self, max_iterations=100, learning_rate=0.2) :
 
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
 
    def fit(self, X, y) :
        self.w = np.zeros(len(X[0]))
        self.w_pocket = np.zeros(len(X[0]))
        converged = False
        iterations = 0
        while (not converged and iterations < self.max_iterations) :
            converged = True
            for i in range(len(X)) :
                if y[i] * self.discriminant(X[i]) <= 0 :
                    self.w = self.w + y[i] * self.learning_rate * X[i]
                    E_in_w = self.insample_error(X,y,self.w)
                    E_in_w_pocket = self.insample_error(X,y,self.w_pocket)
                    if E_in_w < E_in_w_pocket:
                        self.w_pocket = self.w
                    converged = False
            iterations += 1
        self.converged = converged
        if converged :
            print 'converged in %d iterations ' % iterations
        else:
            print 'Not converged'  
            
    def fit_pocket(self, X, y) :
        self.w = np.zeros(len(X[0]))
        self.w_pocket = np.zeros(len(X[0]))
        converged = False
        iterations = 0
        while (not converged and iterations < self.max_iterations) :
            converged = True
            for i in range(len(X)) :
                if y[i] * self.discriminant(X[i]) <= 0 :
                    self.w = self.w + y[i] * self.learning_rate * X[i]
                    E_in_w = self.insample_error(X,y,self.w)
                    E_in_w_pocket = self.insample_error(X,y,self.w_pocket)
                    if E_in_w < E_in_w_pocket:
                        self.w_pocket = self.w
                    converged = False
            iterations += 1
        self.converged = converged
        if converged :
            print 'converged in %d iterations ' % iterations
        else:
            print 'Not converged'
            
    def fit_modified(self, X, y) :
        self.w=[]
        for i in range(len(X[0])):
            self.w.append(np.random.uniform(-1,1))
        c = np.random.rand()  
        converged = False
        iterations = 0  
        while (not converged and iterations < self.max_iterations) :    
            self.w = preprocessing.normalize(self.w,norm='l2')
            Lambda= np.zeros(len(X))
            flag = 1
            index=0
            converged = True
            max_lambda=-10000000000
            for i in range(len(X)) :
                Lambda[i] = y[i] * self.discriminant(X[i])
                if Lambda[i] < c:
                    if Lambda[i] > max_lambda:
                        max_lambda = Lambda[i]
                        index=i                   
                    flag = 0
                    converged = False
            if flag == 0:
                self.w = self.w + y[index] * self.learning_rate * X[index]
            iterations += 1
        self.converged = converged
        if converged :
            print 'Converged in %d iterations ' % iterations
        else :
            print 'Not converged'

 
    def discriminant(self, x) :
        return np.dot(self.w, x)
    
    def insample_error(self,X,y,w):
        N = len(X)
        mismatch = 0
        for i in range(len(X)) :
            if int(np.sign(np.dot(w,X[i]))) != int(np.sign(y[i])):
                mismatch += 1
        in_error = mismatch / float(N)
        return in_error
        
def get_data (data,labels):
    I = np.arange(len(data))
    np.random.shuffle(I)
    data = data[I]
    labels = labels[I]  
    y_train = labels[0:(len(labels)-1500)]
    y_test = labels[(len(labels)-1500):len(labels)]
    X_train = data[0:(len(data)-1500),:]
    X_test = data[(len(data)-1500):len(data),:]
    X_train_bias = np.c_[np.ones(len(data[0:(len(data)-1500),0])),data[0:(len(data)-1500),:]]
    X_test_bias = np.c_[np.ones(len(data[(len(data)-1500):len(data),0])),data[(len(data)-1500):len(data),:]]   
    return X_test, X_test_bias, X_train, X_train_bias, y_test, y_train
 
if __name__=='__main__' :
    i=0
    no_of_iterations=5
    err_in_no_bias=[]
    err_out_no_bias=[]
    err_in_with_bias=[]
    err_out_with_bias=[]
    err_in_pocket=[]
    err_out_pocket=[]
    err_in_modified=[]
    err_out_modified=[]
    for i in range(no_of_iterations):
#    for i in range(0,1):
        data = np.genfromtxt("/data/Work_CSU/Course_Work/Dropbox/Sneha_share/Latest_assgn/Final/Gisette_Final/gisette_train.data") 
        labels = np.genfromtxt("/data/Work_CSU/Course_Work/Dropbox/Sneha_share/Latest_assgn/Final/Gisette_Final/gisette_train.labels") 
        X_test, X_test_bias, X_train, X_train_bias, y_test, y_train = get_data (data,labels)     
        p = Perceptron()
        p.fit(X_train,y_train)
        err_in_no_bias.append(p.insample_error(X_train,y_train,p.w))
        err_out_no_bias.append(p.insample_error(X_test,y_test,p.w))
        p.fit(X_train_bias,y_train)    
        err_in_with_bias.append(p.insample_error(X_train_bias,y_train,p.w))
        err_out_with_bias.append(p.insample_error(X_test_bias,y_test,p.w))
        p.fit_pocket(X_train_bias,y_train)
        err_in_pocket.append(p.insample_error(X_train_bias,y_train,p.w_pocket))
        err_out_pocket.append(p.insample_error(X_test_bias,y_test,p.w_pocket))
        p.fit_modified(X_train,y_train)
        err_in_modified.append(p.insample_error(X_train,y_train,p.w))
        err_out_modified.append(p.insample_error(X_test,y_test,p.w))        
    fig = plt.figure(figsize=(6,6))
    plt.subplot(2,1,1)
    plt.title('Comparison of Ein')
    plt.plot(err_in_no_bias,marker='o', label = 'err_in_no_bias',color='r')
    plt.plot(err_in_with_bias,marker='s', label='err_in_with_bias',color='b')
    plt.plot(err_in_pocket,marker='*', label=' err_in_pocket', color='m')
    plt.plot(err_in_modified,marker='d',label='err_in_modified',color='c')
    plt.ylabel('Ein')
    plt.subplot(2,1,2)
    plt.title('Comparison of Eout')
    plt.plot(err_out_no_bias,marker='o',label = 'err_out_no_bias',color='r')
    plt.plot(err_out_with_bias,marker='s',label = 'err_out_with_bias',color='b')
    plt.plot(err_out_pocket,marker='*',label = 'err_in_pocket',color='m')
    plt.plot(err_out_modified,marker='d',label = 'err_out_modified',color='c')
    plt.xlabel('Iteration')
    plt.ylabel('Eout')   
    plt.show
    plt.savefig("All_Figure.jpg")
    
