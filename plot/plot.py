import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_trj_3_wind(x_trj, wind_field_fn):
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    x, y, z = np.meshgrid(
        # (-min, max, step_length)
        np.arange(-10, 10, 7),
        np.arange(-10, 10, 7),
        np.arange(0, 10, 3),
    )
    u, v, w = wind_field_fn(x, y, z)
    ax.quiver(x, y, z, u, v, w, length=1, linewidth=1)

    ax.plot(
        x_trj[:, 0],
        x_trj[:, 1],
        x_trj[:, 2],
        label="Flight path",
        color="red",
        linewidth=1,
    )
    ax.scatter(x_trj[0, 0], x_trj[0, 1], x_trj[0, 2])
    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()

    return
