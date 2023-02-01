import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import warnings
from matplotlib.pyplot import Axes
from matplotlib.axis import Axis
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import StrMethodFormatter, LogLocator, AutoLocator, AutoMinorLocator
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np


def fig_width():
    return 5.31445


def plot_frame(frame, pos=None, show_immediately=True):
    plt.figure()
    plt.imshow(frame, cmap=plt.cm.gray)
    if pos is not None:
        plt.plot(pos[0], pos[1], 'rx')
    plt.xticks([])
    plt.yticks([])
    if show_immediately:
        plt.show()


def initialize_plt(font_size=10, line_scale=1, capsize=3, latex=True, dpi=300, colourblind=True):
    warnings.filterwarnings("ignore", "cmr10")  # STOP TELLING ME THIS PLEASE I KNOW WHAT FONT I WANT
    if colourblind:
        sns.set_palette("colorblind")
        # plt.style.use('seaborn-colorblind')
    if (latex):
        plt.rc('text', usetex=True)
        plt.rc('text.latex',
               preamble=r'\usepackage{amsmath}\usepackage{amsfonts}\usepackage{amssymb}\DeclareMathAlphabet\mathsfbi{OT1}{cmss}{m}{sl}')
        font = {'family': 'serif', 'size': font_size, 'serif': ['cmr10']}
        plt.rc('font', **font)
    plt.rc('lines', linewidth=line_scale, markersize=3 * line_scale)
    plt.rc('xtick.major', width=line_scale / 2)
    plt.rc('xtick.minor', width=line_scale / 2)
    plt.rc('ytick.major', width=line_scale / 2)
    plt.rc('ytick.minor', width=line_scale / 2)
    plt.rc('axes', linewidth=0.5 * line_scale)
    plt.rc('patch', linewidth=0.5 * line_scale)
    plt.rc('figure', dpi=dpi)
    plt.rc('errorbar', capsize=capsize)


def label_subplot(ax, label, color='k', loc='tl'):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    font_size = plt.rcParams['font.size']
    x_off = 0.33 * font_size / 72  # inches
    y_off = 0.33 * font_size / 72

    ax_width = ax.get_window_extent().width / ax.figure.dpi  # inches
    ax_height = ax.get_window_extent().height / ax.figure.dpi  # inches

    if loc == 'out':
        height_frac = (ax_height - y_off) / ax_height
        width_frac = - x_off / ax_width
        valign = 'top'
        halign = 'right'
    else:
        if loc[0] == 't':
            height_frac = (ax_height - y_off) / ax_height
            valign = 'top'
        elif loc[0] == 'b':
            height_frac = y_off / ax_height
            valign = 'bottom'
        else:
            raise ValueError("loc must be tl, bl, tr, or br")

        if loc[1] == 'l':
            width_frac = x_off / ax_width
            halign = 'left'
        elif loc[1] == 'r':
            width_frac = (ax_width - x_off) / ax_width
            halign = 'right'
        else:
            raise ValueError("loc must be tl, bl, tr, or br")

    ax.annotate(label,
                xy=(np.mean(xlim), np.mean(ylim)),  # ensure it shows up
                xytext=(width_frac, height_frac),  # points = inches * 72
                textcoords='axes fraction', color=color,
                horizontalalignment=halign, verticalalignment=valign)


def format_axis_ticks_decimal(axis: Axis, decimal_places=2):
    y_formatter = StrMethodFormatter("{x:.%0df}" % decimal_places)
    axis.set_major_formatter(y_formatter)
    axis.set_minor_formatter(y_formatter)

    axis.set_major_locator(AutoLocator())
    axis.set_minor_locator(AutoLocator())


color_dict = {'red': ((0.0, 0.0, 0.0),
                      (0.9, 0.5, 1.0),
                      (1.0, 1.0, 1.0)),
              'green': ((0.0, 0.0, 0.0),
                        (0.5, 0.0, 0.0),
                        (1.0, 0.0, 0.0)),
              'blue': ((0.0, 0.0, 0.0),
                       (0.5, 0.0, 0.0),
                       (1.0, 0.0, 0.0))}

heatmap_cm = LinearSegmentedColormap('heatmap', color_dict)


def set_axes_radius(ax, origin, radius):
    """ https://stackoverflow.com/a/50664367/5270376 """
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def set_axes_equal(ax):
    """
    https://stackoverflow.com/a/50664367/5270376

    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)


def plot_3d_points(points, c=None, cmap_name="RdBu", center_cmap=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # type: Axes3D
    points = np.array(points)
    if c is not None:
        if center_cmap:
            scat = ax.scatter3D(points[:, 0], points[:, 2], points[:, 1], c=c, cmap=cm.get_cmap(cmap_name),
                                vmin=-np.max(np.abs(c)), vmax=np.max(np.abs(c)))
        else:
            scat = ax.scatter3D(points[:, 0], points[:, 2], points[:, 1], c=c, cmap=cm.get_cmap(cmap_name))
        fig.colorbar(scat)
    else:
        ax.scatter3D(points[:, 0], points[:, 2], points[:, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')

    lim = [min([ax.get_xlim()[0], ax.get_ylim()[0], ax.get_zlim()[0]]),
           max([ax.get_xlim()[1], ax.get_ylim()[1], ax.get_zlim()[1]])]
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_zlim(lim)
    plt.show()


def plot_2d_point_sets(point_sets):
    colors = ["r", "b", "g", "orange", "k", "yellow"]
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    for i in range(len(point_sets)):
        points = np.array(point_sets[i])
        ax.scatter(points[:, 0], points[:, 1], c=colors[i])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    lim = [min([ax.get_xlim()[0], ax.get_ylim()[0]]),
           max([ax.get_xlim()[1], ax.get_ylim()[1]])]
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    plt.show()


def plot_3d_point_sets(point_sets, colors=("r", "b", "g", "orange", "k", "yellow"), colorbar=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # type: Axes3D
    for i in range(len(point_sets)):
        points = np.array(point_sets[i])
        scat = ax.scatter3D(points[:, 0], points[:, 2], points[:, 1], c=colors[i])
        if colorbar:
            plt.colorbar(scat)
    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')

    # lim = [min([ax.get_xlim()[0], ax.get_ylim()[0], ax.get_zlim()[0]]),
    #        max([ax.get_xlim()[1], ax.get_ylim()[1], ax.get_zlim()[1]])]
    # ax.set_xlim(lim)
    # ax.set_ylim(lim)
    # ax.set_zlim(lim)
    set_axes_equal(ax)
    plt.show()


def plot_heatmap(points, x_bins=100, mode='smooth', filename=None, cmap_name='viridis'):
    """
    Generate a heatmap plot of a number of points.
    :param points: The points to use in the heatmap.
    :param x_bins: The number of bins along the x-axis.
    :param mode: 'smooth' for smooth bin edges, 'sharp' for sharp bin edges.
    :param filename: The filename (and path) to save the plot, does not save if None.
    """
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]

    x_dim = abs(np.ptp(x))
    y_dim = abs(np.ptp(y))

    # Generate heatmap
    y_bins = int(round(x_bins * y_dim / x_dim))
    if y_bins == 0:
        y_bins = 1
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=(x_bins, y_bins))
    extent = [xedges[0] - 1, xedges[-1] + 1, yedges[0] - 1, yedges[-1] + 1]

    heatmap = heatmap.T

    fig = plt.figure()
    if abs(yedges[-1] - yedges[0]) < 0.025:
        yedges[0] = -0.025  # Ensure that the data is actually visible.

    if mode == 'smooth':
        ax = fig.add_subplot(111, aspect='equal', xlim=xedges[[0, -1]], ylim=yedges[[0, -1]])
        im = mpl.image.NonUniformImage(ax, interpolation='bilinear')
        xcenters = (xedges[:-1] + xedges[1:]) / 2
        ycenters = (yedges[:-1] + yedges[1:]) / 2
        im.set_data(xcenters, ycenters, heatmap)
        ax.images.append(im)
    elif mode == 'sharp':
        ax = fig.add_subplot(111, aspect='equal')
        X, Y = np.meshgrid(xedges, yedges)
        cmap = None
        if cmap_name == "heatmap":
            cmap = heatmap_cm
        else:
            cmap = cm.get_cmap(cmap_name)
        ax.pcolormesh(X, Y, heatmap, cmap=cmap)
    else:
        raise (ValueError("{0} is not a valid mode.".format(mode)))
    if filename is not None:
        plt.savefig(filename)
    plt.show()


if __name__ == "__main__":
    this_points = [[0, 1], [0, 2], [2, 3], [1, 3.5]]
    plot_heatmap(this_points, x_bins=5, mode='sharp')
