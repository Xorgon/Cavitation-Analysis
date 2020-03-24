import numpy as np
import itertools
import matplotlib.pyplot as plt

from numerical.models.corner.peters_corner import get_peters_corner_jet_dir

n = 100
xs = np.linspace(0.05, 1, n)
ys = np.linspace(0.05, 1, n)
X, Y = np.meshgrid(xs, ys)
U = np.empty(X.shape)
V = np.empty(X.shape)

for i, j in itertools.product(range(len(xs)), range(len(ys))):
    vel = get_peters_corner_jet_dir([xs[i], ys[j], 0], 2)
    U[i, j], V[i, j], _ = vel

S = np.linalg.norm([U, V], axis=0)
U = U / S
V = V / S

fig = plt.figure()
fig.gca().set_aspect('equal', 'box')
plt.contourf(X, Y, S, levels=32)
plt.quiver(X, Y, U, V, scale=50)
plt.xlabel("$p$")
plt.ylabel("$q$")
plt.colorbar(label="$|v|$")
plt.plot([0, 0, max(xs)], [max(ys), 0, 0], 'k')
plt.show()
