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

def radial_basis_linear_combination(d = 1000, sigma = 1, l=0.1, n=5):
    x = np.linspace(0,1,d)
    basis = np.array([np.exp(-0.5*(x-c)**2/l**2) / (l * np.sqrt(np.pi)) for c in x]).T
    for _ in range(n):
        weights = (sigma**2 / d) * np.random.normal(size=(d))
        plt.plot(x,basis @ weights)
    plt.show()

def multidimensional_brownian_motion(d = 40, l=1):
    axis = np.linspace(-1,1,d)
    points = np.array(np.meshgrid(axis,axis)).reshape(2,-1).T
    #k = np.array([[np.linalg.norm(np.array(list(map(lambda t: abs(max(min(t),0)+max(-max(t),0)),[(x1,x2),(y1,y2)]))),ord=1) for x1,y1 in points] for x2,y2 in points])
    k = np.array([[np.exp(-np.linalg.norm([x1-x2,y1-y2])/l) for x1,y1 in points] for x2,y2 in points])
    vals = np.random.multivariate_normal(np.zeros(len(points)),k).reshape(d,d)
    #plt.plot(vals)
    #plt.show()
    plt.imshow(vals)
    plt.show()

if __name__=="__main__":
    #radial_basis_linear_combination()
    multidimensional_brownian_motion()
