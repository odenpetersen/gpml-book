# Linear regression
Assume no intercept for simplicity.
## (Incorrect) weight-space view
$X_{test} (X_{train}^T X_{train})^{-1} X_{train}^T$ gives the weights for each test case. Note that this is effectively the inner product with respect to $(X_{train}^T X_{train})^{-1}$. Right-multiplying the expression by $y_{train}$ gives the prediction.
### Do the "weights" add to one?
```
X_train = np.random.normal(size=(100,10))
X_test  = np.random.normal(size=(5,10))
weights = X_test @ np.linalg.solve(X_train.T @ X_train,X_train.T)
weights.sum(axis=1)
```
```
array([ 0.27793818,  0.02144581, -0.39510935, -0.20132966,  0.02201635])
```
Answer: no, doesn't look like it.

## Correct weight-space view
$\hat y_{test}=(K(X_train,X_train)^{-1} y_train)^T K(X_train,X_test)$
```
K = X_train @ X_train.T + 0.01*np.eye(X_train.shape[0]) #Numerical regularisation
K_train_test = X_train @ X_test.T

beta = np.random.normal(size=10)
y_train = X_train @ beta
beta_hat = np.linalg.solve(X_train.T @ X_train, X_train.T @ y_train)
alpha = np.linalg.solve(K,y_train)
alpha.T @ K_train_test, X_test @ beta
```

Alternatively (Algorithm 2.1 in the book):
```
L = np.linalg.cholesky(K)
alpha = np.linalg.solve(L.T,np.linalg.solve(L,y_train))
alpha.T @ K_train_test
```

# SARCOS Robotics Example

```
import scipy.io
mat = scipy.io.loadmat('../data/sarcos_inv.mat')
```

Should take z-scores of everything before using MSE loss.

Models to test:
- Linear regression
- GP subset of regresors
