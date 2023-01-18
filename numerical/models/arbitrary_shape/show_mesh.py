from stl import mesh
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import numpy as np

# Create a new plot
figure = pyplot.figure()
axes = mplot3d.Axes3D(figure, auto_add_to_figure=False)
figure.add_axes(axes)

# TODO: Coming in matplotlib 3.6
# vertical_axis = 'z'
# elev, azim, roll = 30, 0, 0
# axes.view_init(elev, azim, roll, vertical_axis=vertical_axis)

# Load the STL files and add the vectors to the plot
# your_mesh = mesh.Mesh.from_file(
#     'C:/Users/eda1g15/OneDrive - University of Southampton/Research/Corners/CAD/n4-corner/n4-corner-surface.STL')
# your_mesh = mesh.Mesh.from_file('C:/Users/eda1g15/OneDrive - University of Southampton/Research/Abritrary Shape/CAD/Shape 1/surface.STL')
your_mesh = mesh.Mesh.from_file("C:/Users/eda1g15/OneDrive - University of Southampton/Research/Porous Materials/CAD/Manufacturing/50mm plates/Square plates/w20vf24/w20vf24 - simple.STL")

your_mesh.translate([-0.025, -0.001, -0.025])

centroids = np.mean([your_mesh.v0, your_mesh.v1, your_mesh.v2], axis=0)
ns = your_mesh.get_unit_normals()  # TODO: Figure out a way to determine exactly which to use (+ve or -ve)

axes.add_collection3d(mplot3d.art3d.Poly3DCollection(your_mesh.vectors, alpha=0.9, color=np.abs(ns)))
# axes.scatter3D(centroids[:, 0], centroids[:, 1], centroids[:, 2], c="C1")
# axes.quiver3D(centroids[:, 0], centroids[:, 1], centroids[:, 2],
#               0.0005 * ns[:, 0], 0.0005 * ns[:, 1], 0.0005 * ns[:, 2], color="C2")

# Auto scale to the mesh size
scale = your_mesh.points.flatten()
# axes.set_xlim3d((0, 0.01))
# axes.set_ylim3d((0, 0.01))
# axes.set_zlim3d((0, 0.01))
axes.auto_scale_xyz(scale, scale, scale)

# Show the plot to the screen
pyplot.show()
