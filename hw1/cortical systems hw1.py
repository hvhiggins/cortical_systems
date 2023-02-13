import numpy as np
import matplotlib.pyplot as plt

trials = 500

g = np.random.default_rng(123)

var_y = []
for n in range(1,50):
    xi = g.poisson(lam=100,size=n*trials)
    xi.resize((n,trials))
    xc = g.poisson(lam=5, size=trials)
    # print(np.var(xi+xc))
    var_y.append(np.var(xi+xc))

plt.plot(var_y)