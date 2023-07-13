import numpy as np

T = 3
d = 400

# Activation Function
def sigmoid_fn(xi):
  return 1/(1+np.exp(-xi))

# Derivative of Activation Function
def sigmoid_fn_derivative(xi):
  return sigmoid_fn(xi)*(1-sigmoid_fn(xi))

# Inference: xtp = f(xt,ut)
def inference_dynamics(xt,ut):
  xtp = np.zeros(d)

  for ell in range(d):
    xtp[ell] = sigmoid_fn(xt@ut[ell,1:] + ut[ell,0])

  return xtp

# Forward Propagation
def forward_pass(uu,x0):
  xx = np.zeros((T,d))
  xx[0] = x0

  for t in range(T-1):
    xx[t+1] = inference_dynamics(xx[t],uu[t])

  return xx

# Adjoint dynamics: 
def adjoint_dynamics(ltp,xt,ut):
  df_dx = np.zeros((d,d))
  df_du = np.zeros(((d+1)*d,d))
  
  dim = np.tile([d+1],d)
  cs_idx = np.append(0,np.cumsum(dim))
  
  for ell in range(d):
    temp = xt@ut[ell,1:] + ut[ell,0]
    disgma_ell = sigmoid_fn_derivative(temp)
    df_dx[:,ell] = disgma_ell*ut[ell,1:]

    df_du[ cs_idx[ell]:cs_idx[ell+1] , ell] = disgma_ell*np.hstack([1,xt])

  lt = df_dx@ltp
  Delta_ut_vec = df_du@ltp 
  Delta_ut = np.reshape(Delta_ut_vec,(d,d+1))

  return lt, Delta_ut

# Backward Propagation
def backward_pass(xx,uu,llambdaT):
  llambda = np.zeros((T,d))
  Delta_u = np.zeros((T-1,d,d+1))

  llambda[-1] = llambdaT

  for t in reversed(range(T-1)):
    llambda[t], Delta_u[t] = adjoint_dynamics(llambda[t+1],xx[t],uu[t])

  return llambda,Delta_u

def binary_cross_entropy(y_tilde, y):
  loss = y * np.log(y_tilde + 1e-6) + (1 - y) * (np.log(1 - y_tilde + 1e-6))
  grad_loss = - (y / (y_tilde + 1e-6)) + (1 - y) / (1 - y_tilde + 1e-6)
  return -loss, grad_loss
