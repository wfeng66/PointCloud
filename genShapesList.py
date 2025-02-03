import math
import numpy as np
from random import random, sample, randint, shuffle
from pathlib import Path


def make_point_clouds(n_samples_per_shape: int, n_points: int, noise: float):
    """Make point clouds for circles, spheres, and tori with random noise.
    """
    circle_point_clouds = [
        np.asarray(
            [
                [np.sin(t) + noise * (np.random.rand(1)[0] - 0.5), np.cos(t) + noise * (np.random.rand(1)[0] - 0.5), 0]
                for t in range((n_points ** 2))
            ]
        )
        for kk in range(n_samples_per_shape)
    ]
    # label circles with 0
    circle_labels = np.zeros(n_samples_per_shape)

    sphere_point_clouds = [
        np.asarray(
            [
                [
                    np.cos(s) * np.cos(t) + noise * (np.random.rand(1)[0] - 0.5),
                    np.cos(s) * np.sin(t) + noise * (np.random.rand(1)[0] - 0.5),
                    np.sin(s) + noise * (np.random.rand(1)[0] - 0.5),
                ]
                for t in range(n_points)
                for s in range(n_points)
            ]
        )
        for kk in range(n_samples_per_shape)
    ]
    # label spheres with 1
    sphere_labels = np.ones(n_samples_per_shape)

    torus_point_clouds = [
        np.asarray(
            [
                [
                    (2 + np.cos(s)) * np.cos(t) + noise * (np.random.rand(1)[0] - 0.5),
                    (2 + np.cos(s)) * np.sin(t) + noise * (np.random.rand(1)[0] - 0.5),
                    np.sin(s) + noise * (np.random.rand(1)[0] - 0.5),
                ]
                for t in range(n_points)
                for s in range(n_points)
            ]
        )
        for kk in range(n_samples_per_shape)
    ]
    # label tori with 2
    torus_labels = 2 * np.ones(n_samples_per_shape)

    point_clouds = np.concatenate((circle_point_clouds, sphere_point_clouds, torus_point_clouds))
    labels = np.concatenate((circle_labels, sphere_labels, torus_labels))

    return point_clouds, labels


def make_gravitational_waves(
    path_to_data: Path,
    n_signals: int = 30,
    downsample_factor: int = 2,
    r_min: float = 0.075,
    r_max: float = 0.65,
    n_snr_values: int = 10,
        ):
    def padrand(V, n, kr):
        cut = np.random.randint(n)
        rand1 = np.random.randn(cut)
        rand2 = np.random.randn(n - cut)
        out = np.concatenate((rand1 * kr, V, rand2 * kr))
        return out

    Rcoef = np.linspace(r_min, r_max, n_snr_values)
    Npad = 500  # number of padding points on either side of the vector
    gw = np.load(path_to_data / "gravitational_wave_signals.npy")
    Norig = len(gw["data"][0])
    Ndat = len(gw["signal_present"])
    N = int(Norig / downsample_factor)

    ncoeff = []
    Rcoeflist = []

    for j in range(n_signals):
        ncoeff.append(10 ** (-19) * (1 / Rcoef[j % n_snr_values]))
        Rcoeflist.append(Rcoef[j % n_snr_values])

    noisy_signals = []
    gw_signals = []
    k = 0
    labels = np.zeros(n_signals)

    for j in range(n_signals):
        signal = gw["data"][j % Ndat][range(0, Norig, downsample_factor)]
        sigp = int((np.random.randn() < 0))
        noise = ncoeff[j] * np.random.randn(N)
        labels[j] = sigp
        if sigp == 1:
            rawsig = padrand(signal + noise, Npad, ncoeff[j])
            if k == 0:
                k = 1
        else:
            rawsig = padrand(noise, Npad, ncoeff[j])
        noisy_signals.append(rawsig.copy())
        gw_signals.append(signal)

    return noisy_signals, gw_signals, labels


# create cubes with various size and number of points
def create_cubes(cube_sizes, num_points):
    # Create a list to store the cubes
    cubes = []

    # Create the cubes and append them to the list
    for size, n_points in zip(cube_sizes, num_points):
        x = np.linspace(-size/2, size/2, int(np.cbrt(n_points)))
        y = np.linspace(-size/2, size/2, int(np.cbrt(n_points)))
        z = np.linspace(-size/2, size/2, int(np.cbrt(n_points)))
        xx, yy, zz = np.meshgrid(x, y, z)
        cube_points = np.column_stack((xx.flatten(), yy.flatten(), zz.flatten()))
        remain = []
        for _ in range(n_points - cube_points.shape[0]):
            remain.append((size*random()-size/2, size*random()-size/2, size*random()-size/2))
        cube_points = np.concatenate((cube_points, np.array(remain)), axis=0)
        cubes.append(cube_points)

    return np.array(cubes)


# def rotate_point_cloud(point_cloud, angles):
#     rotation_matrix = np.array([[math.cos(angles[0]), -math.sin(angles[0]), 0],
#                                 [math.sin(angles[0]), math.cos(angles[0]), 0],
#                                 [0, 0, 1]])
#
#     rotation_matrix = np.dot(rotation_matrix, np.array([[1, 0, 0],
#                                                         [0, math.cos(angles[1]), -math.sin(angles[1])],
#                                                         [0, math.sin(angles[1]), math.cos(angles[1])]]))
#
#     rotation_matrix = np.dot(rotation_matrix, np.array([[math.cos(angles[2]), 0, -math.sin(angles[2])],
#                                                         [0, 1, 0],
#                                                         [math.sin(angles[2]), 0, math.cos(angles[2])]]))
#
#     rotated_points = np.dot(point_cloud, rotation_matrix)
#     return rotated_points

def rotate_point_cloud(point_cloud, angles):
    rotation_matrix = np.array([[math.cos(angles[0]), 0, math.sin(angles[0])],
                                [0, 1, 0],
                                [-math.sin(angles[0]), 0, math.cos(angles[0])]])

    rotation_matrix = np.dot(rotation_matrix, np.array([[1, 0, 0],
                                                        [0, math.cos(angles[1]), -math.sin(angles[1])],
                                                        [0, math.sin(angles[1]), math.cos(angles[1])]]))

    rotation_matrix = np.dot(rotation_matrix, np.array([[math.cos(angles[2]), -math.sin(angles[2]), 0],
                                                        [math.sin(angles[2]), math.cos(angles[2]), 0],
                                                        [0, 0, 1]]))

    rotated_points = np.dot(point_cloud, rotation_matrix)
    return rotated_points


def create_shape_point_cloud(shape_params):
    # this function create point clouds according to the parameter set, shape_params
    # shape_params: a set whose keys are the shapes
    # return: a list of point clouds with various shapes
    angles = (0, math.pi / 2, 0)
    shape_points = []
    for shape, params in shape_params.items():
        subparams = zip(*params.values())
        for subparam in subparams:
            if shape == "cube":
                edge_lengths = subparam[0]
                num_points = subparam[1]
                high_density_points = create_cube_point_cloud(edge_lengths, 50000)  # Generate high density of points
                sampled_points = sample_points(high_density_points, num_points)  # Sample required number of points
                shape_points.append(sampled_points)
            elif shape == "sphere":
                radius = subparam[0]
                num_points = subparam[1]
                high_density_points = create_sphere_point_cloud(radius, 50000)  # Generate high density of points
                sampled_points = sample_points(high_density_points, num_points)  # Sample required number of points
                shape_points.append(sampled_points)
            elif shape == "cuboid":
                edge_lengths = subparam[0]
                num_points = subparam[1]
                high_density_points = create_cuboid_point_cloud(edge_lengths, 50000)  # Generate high density of points
                sampled_points = sample_points(high_density_points, num_points)  # Sample required number of points
                shape_points.append(sampled_points)
            elif shape == "cylinder":
                radius = subparam[0]
                height = subparam[1]
                num_points = subparam[2]
                high_density_points = create_cylinder_point_cloud(radius, height,
                                                                  50000)  # Generate high density of points
                sampled_points = sample_points(high_density_points, num_points)  # Sample required number of points
                shape_points.append(sampled_points)
            elif shape == "hexagonal_prism":
                edge_length = subparam[0]
                height = subparam[1]
                num_points = subparam[2]
                high_density_points = create_hexagonal_prism_point_cloud(edge_length, height,
                                                                         50000)  # Generate high density of points
                sampled_points = sample_points(high_density_points, num_points)  # Sample required number of points
                shape_points.append(sampled_points)
            elif shape == "torus":
                major_radius = subparam[0]
                minor_radius = subparam[1]
                num_points = subparam[2]
                high_density_points = create_torus_point_cloud(major_radius, minor_radius,
                                                               50000)  # Generate high density of points
                sampled_points = sample_points(high_density_points, num_points)  # Sample required number of points
                shape_points.append(sampled_points)
            elif shape == "ellipsoid":
                radii = subparam[0]
                num_points = subparam[1]
                high_density_points = create_ellipsoid_point_cloud(radii, 50000)  # Generate high density of points
                sampled_points = sample_points(high_density_points, num_points)  # Sample required number of points
                shape_points.append(sampled_points)
            elif shape == "horizontal_cuboid":
                edge_lengths = subparam[0]
                num_points = subparam[1]
                high_density_points = create_cuboid_point_cloud(edge_lengths, 50000)  # Generate high density of points
                vertical_point_cloud = sample_points(high_density_points, num_points)  # Sample required number of points
                horizontal_point_cloud = rotate_point_cloud(vertical_point_cloud, angles)
                shape_points.append(horizontal_point_cloud)
            elif shape == "horizontal_cylinder":
                radius = subparam[0]
                height = subparam[1]
                num_points = subparam[2]
                high_density_points = create_cylinder_point_cloud(radius, height, 50000)  # Generate high density of points
                vertical_point_cloud = sample_points(high_density_points, num_points)  # Sample required number of points
                horizontal_point_cloud = rotate_point_cloud(vertical_point_cloud, angles)
                shape_points.append(horizontal_point_cloud)
            elif shape == "horizontal_hexagonal_prism":
                edge_length = subparam[0]
                height = subparam[1]
                num_points = subparam[2]
                high_density_points = create_hexagonal_prism_point_cloud(edge_length, height,
                                                                         50000)  # Generate high density of points
                vertical_point_cloud = sample_points(high_density_points, num_points)  # Sample required number of points
                horizontal_point_cloud = rotate_point_cloud(vertical_point_cloud, angles)
                shape_points.append(horizontal_point_cloud)
    # shape_points = np.array(shape_points)

    return shape_points


def sample_points(point_cloud, num_points):
    if len(point_cloud) <= num_points:
        return point_cloud

    indices = np.random.choice(len(point_cloud), size=num_points, replace=False)
    sampled_points = point_cloud[indices]

    return sampled_points


# Helper functions for generating each shape

def create_cube_point_cloud(edge_length, num_points):
    # Generate random coordinates on the surface of the cube
    points = []

    # Generate points on each face of the cube
    for i in range(num_points):
        # Randomly select a face of the cube
        face = np.random.choice(['front', 'back', 'left', 'right', 'top', 'bottom'])

        if face == 'front':
            x = np.random.uniform(-edge_length / 2, edge_length / 2)
            y = np.random.uniform(-edge_length / 2, edge_length / 2)
            z = edge_length / 2
        elif face == 'back':
            x = np.random.uniform(-edge_length / 2, edge_length / 2)
            y = np.random.uniform(-edge_length / 2, edge_length / 2)
            z = -edge_length / 2
        elif face == 'left':
            x = -edge_length / 2
            y = np.random.uniform(-edge_length / 2, edge_length / 2)
            z = np.random.uniform(-edge_length / 2, edge_length / 2)
        elif face == 'right':
            x = edge_length / 2
            y = np.random.uniform(-edge_length / 2, edge_length / 2)
            z = np.random.uniform(-edge_length / 2, edge_length / 2)
        elif face == 'top':
            x = np.random.uniform(-edge_length / 2, edge_length / 2)
            y = edge_length / 2
            z = np.random.uniform(-edge_length / 2, edge_length / 2)
        elif face == 'bottom':
            x = np.random.uniform(-edge_length / 2, edge_length / 2)
            y = -edge_length / 2
            z = np.random.uniform(-edge_length / 2, edge_length / 2)

        points.append([x, y, z])

    # Convert the list of points into a NumPy array
    point_cloud = np.array(points)

    return point_cloud


def create_cuboid_point_cloud(edge_lengths, num_points):
    # Extract edge lengths
    length_x, length_y, length_z = edge_lengths

    # Generate random coordinates on the surface of the cuboid
    points = []

    # Generate points on each face of the cuboid
    for i in range(num_points // 6):
        # Randomly select a face of the cuboid
        face = np.random.choice(['front', 'back', 'left', 'right', 'top', 'bottom'])

        if face == 'front':
            x = np.random.uniform(-length_x / 2, length_x / 2)
            y = np.random.uniform(-length_y / 2, length_y / 2)
            z = length_z / 2
        elif face == 'back':
            x = np.random.uniform(-length_x / 2, length_x / 2)
            y = np.random.uniform(-length_y / 2, length_y / 2)
            z = -length_z / 2
        elif face == 'left':
            x = -length_x / 2
            y = np.random.uniform(-length_y / 2, length_y / 2)
            z = np.random.uniform(-length_z / 2, length_z / 2)
        elif face == 'right':
            x = length_x / 2
            y = np.random.uniform(-length_y / 2, length_y / 2)
            z = np.random.uniform(-length_z / 2, length_z / 2)
        elif face == 'top':
            x = np.random.uniform(-length_x / 2, length_x / 2)
            y = length_y / 2
            z = np.random.uniform(-length_z / 2, length_z / 2)
        elif face == 'bottom':
            x = np.random.uniform(-length_x / 2, length_x / 2)
            y = -length_y / 2
            z = np.random.uniform(-length_z / 2, length_z / 2)

        points.append([x, y, z])

    # Convert the list of points into a NumPy array
    point_cloud = np.array(points)

    return point_cloud


def create_sphere_point_cloud(radius, num_points):
    # Generate random spherical coordinates
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, np.pi, num_points)

    # Convert spherical coordinates to Cartesian coordinates
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    # Combine coordinates into a point cloud array
    point_cloud = np.column_stack((x, y, z))

    return point_cloud


def create_cylinder_point_cloud(radius, height, num_points):
    # Generate random cylindrical coordinates
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    z = np.random.uniform(0, height, num_points)

    # Convert cylindrical coordinates to Cartesian coordinates
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Combine coordinates into a point cloud array
    point_cloud = np.column_stack((x, y, z))

    return point_cloud


def create_hexagonal_prism_point_cloud(edge_length, height, num_points):
    # Generate random coordinates within the hexagonal prism range
    x = np.random.uniform(-edge_length / 2, edge_length / 2, num_points)
    y = np.random.uniform(-edge_length / 2, edge_length / 2, num_points)
    z = np.random.uniform(0, height, num_points)

    # Apply a mask to keep points inside the hexagonal prism
    mask = np.abs(x) + np.abs(y) <= edge_length / 2
    x = x[mask]
    y = y[mask]
    z = z[mask]

    # Combine coordinates into a point cloud array
    point_cloud = np.column_stack((x, y, z))

    return point_cloud


def create_torus_point_cloud(major_radius, minor_radius, num_points):
    # Generate random toroidal coordinates
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, 2 * np.pi, num_points)

    # Convert toroidal coordinates to Cartesian coordinates
    x = (major_radius + minor_radius * np.cos(theta)) * np.cos(phi)
    y = (major_radius + minor_radius * np.cos(theta)) * np.sin(phi)
    z = minor_radius * np.sin(theta)

    # Combine coordinates into a point cloud array
    point_cloud = np.column_stack((x, y, z))

    return point_cloud


def create_ellipsoid_point_cloud(radii, num_points):
    # Generate random ellipsoidal coordinates
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, np.pi, num_points)

    # Convert ellipsoidal coordinates to Cartesian coordinates
    x = radii[0] * np.sin(phi) * np.cos(theta)
    y = radii[1] * np.sin(phi) * np.sin(theta)
    z = radii[2] * np.cos(phi)

    # Combine coordinates into a point cloud array
    point_cloud = np.column_stack((x, y, z))

    return point_cloud


def remove_points(point_cloud, percentage):
    for i in range(point_cloud.shape[0]):
        point_cloud[i] = [point_cloud[i, 0] * (1 + percentage * random()),
                          point_cloud[i, 1] * (1 + percentage * random()) \
            , point_cloud[i, 2] * (1 + percentage * random())]

    return point_cloud

# randomly cut pc in random direction
def destroy(point_cloud, height, variation):
    pts = []
    dim = int(np.round(random() * 2, 0))  # randomly select the diamention to cut points

    direction = int(np.round(random(), 0))  # randomly select the direction to cut points
    threshold = min(point_cloud[:, dim]) + height * (1 + variation * random()) if direction == 0 else \
        max(point_cloud[:, dim]) - height * (1 + variation * random())

    for pt in point_cloud:
        if direction == 0:
            if pt[dim] < threshold:
                pts.append(pt)
        else:
            if pt[dim] > threshold:
                pts.append(pt)
    return np.array(pts)


# collapse point clouds
def destroy(point_cloud, height_percentage, variation):
    collapsed_points = point_cloud.copy()
    
    # Calculate the distance from each point to the center (x, y) = (0, 0)
    distances_from_center = np.linalg.norm(collapsed_points[:, :2], axis=1)
    max_distance = np.max(distances_from_center)
    
    # Determine the original height of the point cloud
    original_height = np.max(point_cloud[:, 2]) - np.min(point_cloud[:, 2])
    
    # Calculate the maximum height based on the percentage
    max_height = original_height * height_percentage
    
    # Make points closer to the center thicker
    collapsed_points[:, 2] = (1 - distances_from_center / max_distance) * max_height * (1 + variation * random())
    
    return collapsed_points




def create_hole(combined_points, distance_threshold, Y, Z, axe='X'):
    filtered_points = []
    for point in combined_points:
        dist_th = distance_threshold * (1 + random() * 0.1)
        if axe == 'X':
            dist_to_tunnel = np.linalg.norm(point - (point[0], Y, Z))
        elif axe == 'Y':
            dist_to_tunnel = np.linalg.norm(point - (Y, point[1], Z))
        else:
            dist_to_tunnel = np.linalg.norm(point - (Y, Z, point[2]))
        if dist_to_tunnel > dist_th:
            filtered_points.append(point)
    return np.array(filtered_points)


def create_point_clouds(pcs):
    # this function create the post point clouds and labels
    # pcs: the point clouds list without holes
    # return: pre_pcs, post_pcs, post_labels, lists
    post_pcs = []  # the post- structures
    post_labels = []
    dirs = ['X', 'Y', 'Z']

    # create 3 holes on the pre- point clouds
    pre_pcs = []
    for i in range(len(pcs)):  # create some holes on the perfect objects' surface
        temp_obj = np.copy(pcs[i])
        # decide the holes to be created
        n_hole = randint(1, 3)
        for _ in range(n_hole):
            # Define a distance threshold to determine points inside the hole
            distance_threshold = float(randint(1, 3)) / 10.0
            # randomly create the coordinates of the hole
            Y = float(randint(1, 8)) / 10.0
            Z = float(randint(1, 8)) / 10.0
            # randomly select the direction of the hole
            hole_dir = dirs[randint(0, 2)]
            temp_obj = create_hole(temp_obj, distance_threshold, Y, Z, hole_dir)
        pre_pcs.append(temp_obj)

    # pre_pcs = np.array(pre_pcs)

    # create no damage post- structures
    e = 0.10  # random error factor
    for i in range(len(pre_pcs)):
        no_damages = remove_points(pre_pcs[i], e)
        post_pcs.append(no_damages)
        post_labels.append(1)

    # create totally damage post- structures
    height = 0.2  # the height to be keepped
    variation = 0.1
    for i in range(len(pre_pcs)):
        total_damages = destroy(pre_pcs[i], height, variation)
        post_pcs.append(total_damages)
        post_labels.append(4)

    # create minor damaged post - structures
    for i in range(len(pre_pcs)):
        minor_damaged = np.copy(pre_pcs[i])
        # decide the holes to be created
        n_hole = randint(1, 3)
        for _ in range(n_hole):
            # Define a distance threshold to determine points inside the hole
            distance_threshold = float(randint(1, 3)) / 10.0
            # randomly create the coordinates of the hole
            Y = float(randint(1, 8)) / 10.0
            Z = float(randint(1, 8)) / 10.0
            # randomly select the direction of the hole
            hole_dir = dirs[randint(0, 2)]
            minor_damaged = create_hole(minor_damaged, distance_threshold, Y, Z, hole_dir)
        post_pcs.append(minor_damaged)
        post_labels.append(2)

    # create major damaged post - structures
    for i in range(len(pre_pcs)):
        major_damaged = np.copy(pre_pcs[i])
        # decide the holes to be created
        n_hole = randint(5, 8)
        for _ in range(n_hole):
            # Define a distance threshold to determine points inside the hole
            distance_threshold = float(randint(2, 5)) / 10.0
            # randomly create the coordinates of the hole
            Y = float(randint(1, 10)) / 10.0
            Z = float(randint(1, 10)) / 10.0
            # randomly select the direction of the hole
            hole_dir = dirs[randint(0, 2)]
            major_damaged = create_hole(major_damaged, distance_threshold, Y, Z, hole_dir)
        post_pcs.append(major_damaged)
        post_labels.append(3)

    return pre_pcs, post_pcs, post_labels


def create_data(pre_pcs, post_pcs, post_labels):
    # create data set
    # combine pre- and post- together
    n_pre_pcs = len(pre_pcs)
    # data = [[pre_pcs[i % n_pre_pcs], post_pcs[i]] for i in range(len(post_pcs))]
    # data = np.array(data)
    # labels = np.array(post_labels)


    data = []
    for i in range(len(post_pcs)):
        pcs = [pre_pcs[i % n_pre_pcs], post_pcs[i]]
        data.append(pcs)

    # # Disrupt the order
    # idx = np.random.permutation(len(post_labels))
    # data = data[idx]
    # labels = np.array(post_labels)[idx]
    idx = list(range(len(post_pcs)))
    shuffle(idx)
    # for i in idx:
    #     shuffled_data = []
    #     print(data[i][0].shape, data[i][1].shape)
    #     shuffled_data.append([data[i][0], data[i][1]])
    #     print(shuffled_data[-1][0].shape, shuffled_data[-1][1].shape)

    data = [[data[i][0], data[i][1]] for i in idx]

    post_labels = [post_labels[i] for i in idx]

    return data, post_labels


def genshapes(n_total_shapes, n_pts, shapes_lst='all'):
    # this function create appropriate parameter set for creating point clouds
    # n_total_shapes: the total number of point clouds to be created
    # n_pts: the least number of points in each point cloud
    # shapes_lst: a list of the shape names to be created. They should be one of the keys in shapes_para.
    n_shapes = 10 if shapes_lst == 'all' else len(shapes_lst)
    n_per_shape = n_total_shapes//(4*n_shapes*2)
    shapes_para = {
        "cube": {
            "edge_lengths ": list(np.linspace(1.0, 3.0, n_per_shape*2)),
            "num_points": [n_pts] * n_per_shape + [n_pts*2] * n_per_shape
        },
        "sphere": {
            "radii ": list(np.linspace(1.0, 3.0, n_per_shape*2)),
            "num_points":  [n_pts] * n_per_shape + [n_pts*2] * n_per_shape
        },
        "cuboid": {
            "edge_lengths ": [[x * float(t) / 10, y * float(t) / 10, z * float(t) / 10] for x, y, z in
                              zip([1.0], [2.0], [0.8]) for t in range(10, 10+(n_per_shape*2))],
            "num_points": [n_pts] * n_per_shape + [n_pts*2] * n_per_shape
        },
        "cylinder": {
            "radius ": list(np.linspace(1.0, 3.0, n_per_shape*2)),
            "height": list(np.linspace(2.0, 6.0, n_per_shape*2)),
            "num_points": [n_pts] * n_per_shape + [n_pts*2] * n_per_shape
        },
        "hexagonal_prism": {
            "edge_length ": list(np.linspace(1.0, 3.0, n_per_shape*2)),
            "height": list(np.linspace(2.0, 6.0, n_per_shape*2)),
            "num_points": [n_pts] * n_per_shape + [n_pts*2] * n_per_shape
        },
        "torus": {
            "major_radius": list(np.linspace(2.0, 6.0, n_per_shape*2)),
            "minor_radius": list(np.linspace(1.0, 3.0, n_per_shape*2)),
            "num_points": [n_pts] * n_per_shape + [n_pts*2] * n_per_shape
        },
        "ellipsoid": {
            "radii": [[x * float(t) / 10, y * float(t) / 10, z * float(t) / 10] for x, y, z in zip([1.0], [2.0], [0.8])
                      for t in range(10, 10+(n_per_shape*2))],
            "num_points": [n_pts] * n_per_shape + [n_pts*2] * n_per_shape
        },
        "horizontal_cuboid": {
            "edge_lengths ": [[x * float(t) / 10, y * float(t) / 10, z * float(t) / 10] for x, y, z in
                              zip([1.0], [2.0], [0.8]) for t in range(10, 10+(n_per_shape*2))],
            "num_points": [n_pts] * n_per_shape + [n_pts*2] * n_per_shape
        },
        "horizontal_cylinder": {
            "radius ": list(np.linspace(1.0, 2.0, n_per_shape*2)),
            "height": list(np.linspace(2.0, 6.0, n_per_shape*2)),
            "num_points": [n_pts] * n_per_shape + [n_pts*2] * n_per_shape
        },
        "horizontal_hexagonal_prism": {
            "edge_length ": list(np.linspace(1.0, 3.0, n_per_shape*2)),
            "height": list(np.linspace(2.0, 6.0, n_per_shape*2)),
            "num_points": [n_pts] * n_per_shape + [n_pts*2] * n_per_shape
        }
    }

    if shapes_lst == 'all':
        para = shapes_para
    else:
        para = {key: shapes_para[key] for key in shapes_lst}
    pcs = create_shape_point_cloud(para)
    pre_pcs, post_pcs, post_labels = create_point_clouds(pcs)
    data, labels = create_data(pre_pcs, post_pcs, post_labels)

    return data, labels


