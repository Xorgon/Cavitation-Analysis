from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import numpy as np


def load_cna_from_mesh(filename):
    stl_mesh = mesh.Mesh.from_file(filename)
    centroids = np.mean([stl_mesh.v0, stl_mesh.v1, stl_mesh.v2], axis=0)

    # TODO: figure out how to determine normals appropriately
    normals = np.array(stl_mesh.get_unit_normals())  # Normals generated from Solidworks STLs are conventionally opposite our normals unless they aren't
    areas = np.array(stl_mesh.areas).reshape((len(stl_mesh.areas)))

    return centroids, normals, areas


if __name__ == "__main__":
    # Create a new plot
    figure = pyplot.figure()
    axes = mplot3d.Axes3D(figure, auto_add_to_figure=False)
    figure.add_axes(axes)

    # TODO: Coming in matplotlib 3.6
    # vertical_axis = 'z'
    # elev, azim, roll = 30, 0, 0
    # axes.view_init(elev, azim, roll, vertical_axis=vertical_axis)

    # Load the STL files and add the vectors to the plot
    your_mesh = mesh.Mesh.from_file('slot.STL')
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors, alpha=0.9))

    centroids = np.mean([your_mesh.v0, your_mesh.v1, your_mesh.v2], axis=0)
    axes.scatter3D(centroids[:, 0], centroids[:, 1], centroids[:, 2], c="C1")

    ns = your_mesh.get_unit_normals()
    axes.quiver3D(centroids[:, 0], centroids[:, 1], centroids[:, 2], ns[:, 0], ns[:, 1], ns[:, 2], color="C2")

    # Auto scale to the mesh size
    scale = your_mesh.points.flatten()
    axes.auto_scale_xyz(scale, scale, scale)

    # Show the plot to the screen
    pyplot.show()
