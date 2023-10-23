#!/usr/bin/env python3
import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def read_data():
    df = pd.read_csv('../data/co2_mm_mlo.csv',comment='#')
    df['year_month'] = df.year+(df.month-1)/12
    return df

def kernel(x,y,params):
    k1 = params[0]**2*jnp.exp(-(x-y)**2/(2*params[1]**2))
    k2 = params[2]**2*jnp.exp(-(x-y)**2/(2*params[3]**2)-2*jnp.sin(jnp.pi*(x-y))**2/(params[4]**2))
    k3 = params[5]**2*(1+(x-y)**2/(2*params[7]*params[6]**2))**(-params[7])
    k4 = params[8]**2*jnp.exp(-(x-y)**2/(2*params[9]**2))+params[10]**2*(x==y)
    return k1+k2+k3+k4

def gram_matrix(xs,params):
    return jnp.array([[kernel(x1,x1,params) for x1 in xs] for x2 in xs])

def surprisal(xs,ys,params):
    cov = gram_matrix(xs,params)
    return 0.5 * (ys.T @ jnp.linalg.solve(cov,ys) + jnp.log(jnp.linalg.det(cov)))

def plot(df):
    train = df.year_month<2004
    test = ~train
    plt.scatter(df.year_month[train],df.average[train])
    plt.scatter(df.year_month[test],df.average[test])
    plt.show()

if __name__=="__main__":
    print("Reading data")
    df = read_data()
    #plot(df)
    train = df.year_month<2004
    test = ~train
    x_train, y_train = df.year_month[train].values, df.average[train].values
    x_test,  y_test  = df.year_month[test].values,  df.average[test].values

    params = np.random.normal(size=(5,11))*100
    print("Training")
    while True:
        """
        gradients = np.zeros(params.shape)
        for x1,y1 in tqdm([*zip(x_train,y_train)]):
            for x2,y2 in zip(x_train,y_train):
                gradient_function = jax.jit(jax.grad(lambda p : surprisal(np.array([x1,x2]),np.array([y1,y2]),p)))
                gradients += np.array(list(map(gradient_function,params)))
        """
        gradient_function = jax.jit(jax.grad(lambda p : surprisal(x_train,y_train,p)))
        print("Computing gradients")
        gradients = np.array(list(map(gradient_function,params)))
        print(*map(lambda p : surprisal(x_train,y_train,p),params))
        params -= gradients
        print(params)
