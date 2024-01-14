## Pyro GP tutorial used as starting point:
## https://pyro.ai/examples/gp.html

import matplotlib.pyplot as plt
import numpy as np
import torch
import pyro
import pyro.contrib.gp as gp

# Partition observations
X = np.asarray([x / 29 for x in range(1, 31)])
np.random.shuffle(X)
Y = 6 * np.square(X) - np.square(np.sin(6 * np.pi * X)) - 5 * np.power(X, 4) + 3 / 2 + np.random.normal(0.0, 0.1, 30)
Xtrain, Xtest, Ytrain, Ytest = X[10:], X[:10], Y[10:], Y[:10]

# Define model and log-likelihood
def model(xs, ys, var, lscale):
    kernel = gp.kernels.Matern32(input_dim=1, variance=torch.tensor(var), lengthscale=torch.tensor(lscale))
    return gp.models.GPRegression(torch.tensor(xs), torch.tensor(ys), kernel, noise=torch.tensor(0.01))

def logLikelihood(params):
    # TODO
    return 1

# Select prior and find posterior estimate
gpr = model(Xtrain, Ytrain, 2, 0.5)


# Use NUTS to sample from posterior
