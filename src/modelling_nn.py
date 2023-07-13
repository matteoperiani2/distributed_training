import numpy as np
import matplotlib.pyplot as plt
from collections import deque

T = 3 # number of layers
input_shape = 784
layer_neurons = [input_shape, 32, 1]                                                                                                                                                                         

# Activation Function
def sigmoid_fn(xi):
  return 1/(1+np.exp(-xi))

# Derivative of Activation Function
def sigmoid_fn_derivative(xi):
  return sigmoid_fn(xi)*(1-sigmoid_fn(xi))

# Inference: xtp = f(xt,ut)
def inference_dynamics(xt,ut):
  return [sigmoid_fn(xt@ut[ell,1:] + ut[ell,0]) for ell in range(ut.shape[0])]

# Forward Propagation
def forward_pass(uu,x0):
  xx = deque(maxlen=T)
  xx.append(x0) #400

  for t in range(T-1):
    xx.append(inference_dynamics(xx[t],uu[t]))

  return xx

# Adjoint dynamics: 
def adjoint_dynamics(ltp,xt,ut):
  df_dx = np.zeros((len(xt), ut.shape[0]))
  df_du = np.zeros((ut.shape[1]*ut.shape[0], ut.shape[0]))

  dim = np.tile(ut.shape[1],ut.shape[0])
  cs_idx = np.append(0,np.cumsum(dim))
  
  for ell in range(ut.shape[0]):
    disgma_ell = sigmoid_fn_derivative(xt@ut[ell,1:] + ut[ell,0])
    df_dx[:,ell] = disgma_ell*ut[ell,1:]
    df_du[ cs_idx[ell]:cs_idx[ell+1] , ell] = disgma_ell*np.hstack([1,xt])

  lt = df_dx@ltp
  Delta_ut_vec = df_du@ltp 
  Delta_ut = np.reshape(Delta_ut_vec,(ut.shape[0],ut.shape[1]))

  return lt, Delta_ut

# Backward Propagation
def backward_pass(xx,uu,llambdaT):
  # llambda = np.zeros((T,d))
  llambda = deque(maxlen=T)
  # Delta_u = np.zeros((T-1,d,d+1))
  Delta_u = deque(maxlen=T-1)

  # llambda[-1] = llambdaT
  llambda.append(np.array([llambdaT]))

  for t in reversed(range(T-1)):
    ll_t, du_t = adjoint_dynamics(llambda[0], xx[t],uu[t])
    Delta_u.appendleft(du_t)
    llambda.appendleft(ll_t)

  # for t in reversed(range(T-1)):
  #   llambda[t], Delta_u[t] = adjoint_dynamics(llambda[t+1],xx[t],uu[t])

  return llambda, Delta_u

def binary_cross_entropy(y_tilde, y):
  loss = y * np.log(y_tilde + 1e-6) + (1 - y) * (np.log(1 - y_tilde + 1e-6))
  grad_loss = - (y / (y_tilde + 1e-6)) + (1 - y) / (1 - y_tilde + 1e-6)
  return -loss, grad_loss
