#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def rbf_kernel(x1,x2,l=1):
    return np.exp(-((x1-x2)/l)**2/2)

def k_matrix(xs,sigma=1e-12,kernel=rbf_kernel):
    K = [[kernel(x1,x2) for x1 in xs] for x2 in xs]
    K = np.array(K)
    K += np.eye(len(K))*sigma #Regularisation
    return K

def q1a(n=3,d=50):
    xs = np.linspace(-5,5,d)
    K = k_matrix(xs)

    gp_samples = np.random.multivariate_normal(np.zeros((d,)),K,size=n)

    plt.plot(xs,gp_samples.T)
    plt.show()

def q1b(n=10,train_size = 5, test_size = 95):
    xs = np.linspace(-5,5,train_size + test_size)
    np.random.shuffle(xs)
    K = k_matrix(xs)

    ys = np.cos(xs**2/5)+np.cos(xs)+0.1*xs**2

    x_train,x_test,y_train,y_test = train_test_split(xs,ys,test_size=test_size,shuffle=False)
    
    mu = K[:,:train_size] @ np.linalg.solve(K[:train_size,:train_size], y_train)
    sigma = K - K[:,:train_size] @ np.linalg.solve(K[:train_size,:train_size],K[:train_size,:])
    print(mu.shape)
    print(sigma.shape)

    gp_samples = np.random.multivariate_normal(mu,sigma,size=n)
    
    plt.plot(*zip(*sorted(zip(xs,ys))),linestyle='dashed',marker='_')
    plt.vlines(x_train,ymin=min(min(ys),min(gp_samples.reshape(-1))),ymax=max(max(ys),max(gp_samples.reshape(-1))))
    plt.plot(*zip(*sorted(zip(xs,gp_samples.T))))
    plt.show()


#q3
"""
mu=0 is obvious.

Applying the general conditioning formula,
sigma(x,y) = K(x,y) - K(x,1) @ K(1,1)^{-1} @ K(1,y) = min(x,y) - x * 1 * y = min(x,y)-xy.
"""
def bridge_kernel(x,y):
    return min(x,y) - x*y
def q3(n=30,d=50):
    xs = np.linspace(0,1,d)
    K = k_matrix(xs,kernel=bridge_kernel)
    
    gp_samples = np.random.multivariate_normal(np.zeros((d,)),K,size=n)

    plt.plot(xs,gp_samples.T)
    plt.show()


def explicit_basis():
    xs = np.linspace(-5,5,train_size + test_size)
    np.random.shuffle(xs)
    K = k_matrix(xs)

    ys = np.cos(xs**2/5)+np.cos(xs)+0.1*xs**2

    x_train,x_test,y_train,y_test = train_test_split(xs,ys,test_size=test_size,shuffle=False)
    
    mu = K[:,:train_size] @ np.linalg.solve(K[:train_size,:train_size], y_train)
    sigma = K - K[:,:train_size] @ np.linalg.solve(K[:train_size,:train_size],K[:train_size,:])
    print(mu.shape)
    print(sigma.shape)

    gp_samples = np.random.multivariate_normal(mu,sigma,size=n)
    
    plt.plot(*zip(*sorted(zip(xs,ys))),linestyle='dashed',marker='_')
    plt.vlines(x_train,ymin=min(min(ys),min(gp_samples.reshape(-1))),ymax=max(max(ys),max(gp_samples.reshape(-1))))
    plt.plot(*zip(*sorted(zip(xs,gp_samples.T))))
    plt.show()

if __name__=="__main__":
    q1a()
    q1b()
    q3()
