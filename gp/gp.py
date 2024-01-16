## Pyro GP tutorial used as starting point:
## https://pyro.ai/examples/gp.html

import matplotlib.pyplot as plt
import numpy as np
import torch
import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist

# Partition observations
X = np.asarray([x / 29 for x in range(1, 31)])
np.random.shuffle(X)
Y = 6 * np.square(X) - np.square(np.sin(6 * np.pi * X)) - 5 * np.power(X, 4) + 3 / 2 + np.random.normal(0.0, 0.1, 30)
Xtrain, Xtest, Ytrain, Ytest = torch.tensor(X[10:]), torch.tensor(X[:10]), torch.tensor(Y[10:]), torch.tensor(Y[:10])

# Define model and log-likelihood
def model(xs, ys, theta):
    kernel = gp.kernels.Matern32(input_dim=1, variance=theta[0], lengthscale=theta[1])
    return gp.models.GPRegression(xs, ys, kernel, noise=torch.tensor(0.01))

# Computes log-likelihood
def logLikelihood(xs, ys, theta):
    # solution based on lecture notes section 6.3
    kernel = gp.kernels.Matern32(input_dim=1, variance=theta[0], lengthscale=theta[1])
    t1 = 0.5 * torch.transpose(ys, 0, 0) * torch.linalg.inv(kernel.forward(xs)) * ys
    t2 = 0.5 * torch.log(torch.linalg.det(kernel.forward(xs)))
    t3 = 15.0 * torch.log(2 * torch.tensor(np.pi))
    return - t1 - t2 - t3

# Select prior and find posterior estimate
prior = dist.MultivariateNormal(torch.tensor([1.5,1]), torch.eye(2))
gpr = model(Xtrain, Ytrain, prior.sample())
llh = logLikelihood(Xtrain, Ytrain, prior.sample())
