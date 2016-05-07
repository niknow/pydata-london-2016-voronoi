import numpy as np
import math
import scipy
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch


def add_sphere(ax, center=np.array([0,0,0]), radius=1, color='y'):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)

    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    x += center[0]
    y += center[1]
    z += center[2]
    ax.plot_surface(
        x, y, z,
        rstride=5,
        cstride=5,
        color=color,
        alpha=0.1
    )


# generate random points on the unit sphere
def generate_random_array_spherical_generators(
        num_generators,
        sphere_radius,prng_object):
    """
    Recoded using standard uniform selector over theta and acos phi,
    http://mathworld.wolfram.com/SpherePointPicking.html
    Same as in iPython notebook version
    """

    u = prng_object.uniform(low=0,high=1,size=num_generators)
    v = prng_object.uniform(low=0,high=1,size=num_generators)
    theta_array = 2 * math.pi * u
    phi_array = np.arccos((2*v - 1.0))
    r_array = sphere_radius * np.ones((num_generators,))
    spherical_polar_data = np.column_stack((r_array,theta_array, phi_array))
    cartesian_random_points = convert_spherical_array_to_cartesian_array(spherical_polar_data)
    return cartesian_random_points


def convert_spherical_array_to_cartesian_array(
        spherical_coord_array,
        angle_measure='radians'):
    """
    Take shape (N,3) spherical_coord_array (r,theta,phi) and return an array of
    the same shape in cartesian coordinate form (x,y,z). Based on the
    equations provided at:
    http://en.wikipedia.org/wiki/List_of_common_coordinate_transformations#From_spherical_coordinates
    use radians for the angles by default, degrees if angle_measure == 'degrees'
    """

    cartesian_coord_array = np.zeros(spherical_coord_array.shape)
    # convert to radians if degrees are used in input
    if angle_measure == 'degrees':
        spherical_coord_array[...,1] = np.deg2rad(spherical_coord_array[...,1])
        spherical_coord_array[...,2] = np.deg2rad(spherical_coord_array[...,2])
    # now the conversion to Cartesian coords
    cartesian_coord_array[...,0] = spherical_coord_array[...,0] * np.cos(spherical_coord_array[...,1]) * np.sin(spherical_coord_array[...,2])
    cartesian_coord_array[...,1] = spherical_coord_array[...,0] * np.sin(spherical_coord_array[...,1]) * np.sin(spherical_coord_array[...,2])
    cartesian_coord_array[...,2] = spherical_coord_array[...,0] * np.cos(spherical_coord_array[...,2])
    return cartesian_coord_array


# horrobly ineffective function to plot a simplex
def plot_simplex(vertices, ax):
    def plot_face(face):
        face = np.append(face, [face[0]], axis=0)
        ax.plot(face[:, 0], face[:, 1], face[:, 2], 'k-')

    plot_face(np.array([vertices[0], vertices[1], vertices[2]]))
    plot_face(np.array([vertices[1], vertices[2], vertices[3]]))
    plot_face(np.array([vertices[2], vertices[3], vertices[0]]))
    plot_face(np.array([vertices[0], vertices[1], vertices[3]]))


def calc_circumcenters(tetrahedrons):
    num = tetrahedrons.shape[0]
    a = np.concatenate((tetrahedrons, np.ones((num, 4, 1))), axis=2)

    sums = np.sum(tetrahedrons ** 2, axis=2)
    d = np.concatenate((sums[:, :, np.newaxis], a), axis=2)

    dx = np.delete(d, 1, axis=2)
    dy = np.delete(d, 2, axis=2)
    dz = np.delete(d, 3, axis=2)

    dx = np.linalg.det(dx)
    dy = -np.linalg.det(dy)
    dz = np.linalg.det(dz)
    a = np.linalg.det(a)

    nominator = np.vstack((dx, dy, dz))
    denominator = 2*a
    return (nominator / denominator).T


def project_to_sphere(points, center, radius):

    lengths = scipy.spatial.distance.cdist(points, np.array([center]))
    return (points - center) / lengths * radius + center
