import numpy as np
import matplotlib.pyplot as plt
 
class Perceptron :
 
    def __init__(self, max_iterations=100, learning_rate=0.2) :
 
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
            print 'No. of training samples taken are %d' % len(X)
            print 'The algorithm converged in %d iterations ' % iterations
        else :
            print 'not converged'
 
    """ Define the product w(transpose) and x"""
    
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

"""To generate the logarithmic scale on x-axis depending on the no. of training samples"""

def generate_x_axis(data):
    Num_ex = []
    initial_tick = 10
    log_base = 2
    Num_ex.append(initial_tick)
    while (initial_tick < len(train_data)):
        for k in range (1,4):
            if (initial_tick*(log_base**k)) > len(train_data):
                break
            Num_ex.append(initial_tick*(log_base**k))
        initial_tick = initial_tick*10 
    return Num_ex            
        
if __name__=='__main__' :
    p = Perceptron() 
    data = np.genfromtxt("/data/Work_CSU/Course_Work/Dropbox/Sneha_share/Latest_assgn/Final/gisette_train.data") 
    labels = np.genfromtxt("/data/Work_CSU/Course_Work/Dropbox/Sneha_share/Latest_assgn/Final/gisette_train.labels")    
    I = np.arange(len(data))
    np.random.shuffle(I)
    data = data[I]
    labels = labels[I]
    train_data = data[0:3000,:]
    test_data = np.c_[np.ones(len(data[3000:len(data),0])),data[3000:len(data),:]]
    train_labels = labels[0:3000]
    test_labels = labels[3000:len(labels)]
    Num_ex = generate_x_axis(data)
    Error = []
    for i in Num_ex :
        if i <= len(train_data):        
            X_train = np.c_[np.ones(len(train_data[0:i,0])),train_data[0:i,:]]
            y_train = train_labels[0:i]
            p.fit(X_train,y_train)
            Error.append(p.outsample_error(test_data,test_labels,p.w))
    fig = plt.figure(figsize=(5,5))
    plt.plot(Num_ex, Error, marker='o')
    plt.xscale('log')
    plt.xticks(Num_ex,('10','20','40','80','200','400','800','2000'))
    plt.xlabel('Number of Test Samples')
    plt.ylabel('In Sample Error')
    plt.title('Learning Curve')
    plt.show()
    plt.savefig("LearningCurve.jpg")
    
    
    
    
    
    