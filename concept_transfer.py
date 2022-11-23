#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
import pdb
import torch
import time
from multiprocessing import Pool, Manager, get_context
from contextlib import closing

from vapory import *
from reasoning.pytorch_net.util import init_args, to_np_array, plot_matrices, set_seed
from reasoning.util import color_dict, visualize_matrices


# ## 0. Helper Functions

# In[ ]:


WHITE = 1
BLACK = 0
DEGREE_OFFSET =  16.8 # Corresponds with 40 degree camera angle
# DEFAULT_NORMAL = Normal('bumps', 0.75, 'scale', 0.0125)
DEFAULT_NORMAL = Normal('bumps', 0, 'scale', 0)
DEFAULT_PHONG = Finish('phong', 0.05)
COLOR_THRESH = 0.33
COLOR_DICT = {
    0: [0, 0, 0],
    1: [0.2, 0.9, 0.2],
    2: [0.9, 0.1, 0.9],
    3: [0.3, 0.3, .9],
    4: [0.9, 0.2, 0.2],
    5: [.5, .55, .5],
    6: [.5, 0.1, .5],
    7: [1, .64, 0.3],
    8: [0.2, 0.9, 0.9],
    9: [1, 0.3, 1],
}

def crop(image, r, c, height, width):
    return image[r:r+height, c:c+width]


def moore_neighbor_tracing(image):
    original_height, original_width = image.shape
    image = np.pad(image, ((1, 1), (1, 1)), 'constant', constant_values=(BLACK, BLACK))
    height, width = image.shape
    contour_pixels = []
    p = (0, 0)
    c = (0, 0)
    s = (0, 0)
    previous = (0, 0)
    found = False

    # Find the first point
    for i in range(height):
        for j in range(width):
            if image[i, j] == WHITE and not (i == 0 and j == 0):
                s = (i, j)
                # contour_pixels.append(s)
                contour_pixels.append((s[0]-1, s[1]-1))
                p = s
                found = True
                break
            if not found:
                previous = (i, j)
        if found:
            break

    # If the pixel is isolated i don't do anything
    isolated = True
    m = moore_neighbor(p)
    for r, c in m:
        if image[r, c] == WHITE:
            isolated = False

    if not isolated:
        tmp = c
        # Backtrack and next clockwise M(p)
        c = next_neighbor(s, previous)
        previous = tmp
        while c != s:
            if image[c] == WHITE:
                previous_contour = contour_pixels[len(contour_pixels) - 1]

                # contour_pixels.append(c)
                contour_pixels.append((c[0]-1, c[1]-1))
                p = c
                c = previous

                # HERE is where i have to start checking for lines
                # i get the previous contour pixel
                current_contour = p[0] - 1, p[1] - 1

            else:
                previous = c
                c = next_neighbor(p, c)
        image = crop(image, 1, 1, original_height, original_width)
    return contour_pixels


def moore_neighbor(pixel):
    row, col = pixel
    return ((row - 1, col - 1), (row - 1, col), (row - 1, col + 1),
            (row, col + 1), (row + 1, col + 1),
            (row + 1, col), (row + 1, col - 1),
            (row, col - 1))


# Function that "deletes" an object using the information about its contours
def delete_object(image, contoured, contours):
    # With the edge pixel i also delete its moore neighborhood because otherwise if and edge is 2 pixel thick
    # because i find only the external contour i wouldn't delete the contour completely
    height, width = image.shape
    for x, y in contours:
        image[x, y] = BLACK
        image[np.clip(x - 1, 0, height - 1), np.clip(y - 1, 0, width - 1)] = BLACK
        image[np.clip(x - 1, 0, height - 1), y] = BLACK
        image[np.clip(x - 1, 0, height - 1), np.clip(y + 1, 0, width - 1)] = BLACK
        image[x, np.clip(y - 1, 0, width - 1)] = BLACK
        image[x, y] = BLACK
        image[x, np.clip(y + 1, 0, width - 1)] = BLACK
        image[np.clip(x + 1, 0, height - 1), np.clip(y - 1, 0, width - 1)] = BLACK
        image[np.clip(x + 1, 0, height - 1), y] = BLACK
        image[np.clip(x + 1, 0, height - 1), np.clip(y + 1, 0, width - 1)] = BLACK

    return image


def next_neighbor(central, neighbor):
    neighbors = moore_neighbor(central)
    index = np.where((np.array(neighbors) == neighbor).all(axis=1))[0][0]
    index += 1
    index = index % 8

    # Problem operating like this:
    # if the object of which i want to detect contours starts at the edges of the image there's the possibility
    # of going out of bounds
    return neighbors[index]

def iter_moore(image):
    contour_pixels = np.zeros(image.shape, np.uint8)
    contour_all = []
    while np.any(image == WHITE):
        contoured = np.zeros(image.shape, np.uint8)
        contours = moore_neighbor_tracing(image)
        for x, y in contours:
            contoured[x, y] = WHITE
            contour_pixels[x, y] = WHITE
        image = delete_object(image, contoured, contours)
        contour_all.append(contours)
    return contour_all, contour_pixels


def get_neigh(coord, all_bound):
    # Don't include diagonals
    row, col = coord
    candidates = [(row - 1, col), (row + 1, col),
            (row, col - 1), (row, col + 1)]
    neighbors = []
    for neigh in candidates:
        if neigh in all_bound:
            neighbors.append(neigh)
    return neighbors


def get_surf_points(coord, all_bound, use_only_outer=False):
    neighbors = get_neigh(coord, all_bound)
    if len(neighbors) < 3:
        if len(neighbors) == 1:
            neigh = neighbors[0]
            rows = set([neigh[0], coord[0]])
            cols = set([neigh[1], coord[1]])
            if len(rows) > 1:
                # neigh is either above or below coord
                if coord[0] > neigh[0]:
                    # Use coord's bottom edge
                    return [(coord[0]+1, coord[1]), (coord[0]+1, coord[1]+1)]
                return [(coord[0], coord[1]), (coord[0], coord[1]+1)]
            else:
                # neigh is either to the left or right of coord
                if coord[1] > neigh[1]:
                    # Use coord's right edge
                    return [(coord[0], coord[1]+1), (coord[0]+1, coord[1]+1)]
                return [(coord[0], coord[1]), (coord[0]+1, coord[1])]
        else:
            # L-shape:
            top = True if neighbors[0][0] < coord[0] or neighbors[1][0] < coord[0] else False
            bottom = True if neighbors[0][0] > coord[0] or neighbors[1][0] > coord[0] else False
            left = True if neighbors[0][1] < coord[1] or neighbors[1][1] < coord[1] else False
            right = True if neighbors[0][1] > coord[1] or neighbors[1][1] > coord[1] else False
            if not use_only_outer:
                if (top and right) or (bottom and left):
                    # Get coord's top right and bottom left corners
                    return [(coord[0], coord[1]+1), (coord[0]+1, coord[1])]
                elif (top and left) or (bottom and right):
                    # Get coord's top left and bottom right corners
                    return [(coord[0], coord[1]), (coord[0]+1, coord[1]+1)]
                else:
                    # Not an L-shape
                    return []
            else:
                # In the case of RectSolid
                if top and right:
                    return [(coord[0]+1, coord[1])]
                elif top and left:
                    return [(coord[0]+1, coord[1]+1)]
                elif bottom and right:
                    return [(coord[0], coord[1])]
                elif bottom and left:
                    return [(coord[0], coord[1]+1)]
                else:
                    return []
    else:
        # T-shape:
        first = neighbors[0]
        # adj is the protruding pixel of the T-shape
        adj = None
        if neighbors[1][0] == first[0] or neighbors[1][1] == first[1]:
            adj = neighbors[2]
        elif neighbors[2][0] == first[0] or neighbors[2][1] == first[1]:
            adj = neighbors[1]
        if adj is None:
            adj = first
        rows = set([coord[0], adj[0]])
        if len(rows) == 1:
            # adj is either on the left or right of coord
            if adj[1] > coord[1]:
                # Use the left edge of adj, i.e. the edge that is shared with coord
                return [(adj[0], adj[1]), (adj[0]+1, adj[1])]
            # Use right edge of adj
            return [(adj[0], adj[1] + 1), (adj[0]+1, adj[1] + 1)]
        else: 
            if adj[0] > coord[0]:
                # Use the top edge of adj, i.e. the edge that is shared with coord
                return [(adj[0], adj[1]), (adj[0], adj[1]+1)]
            # Use bottom edge of adj
            return [(adj[0]+1, adj[1]), (adj[0]+1, adj[1]+1)]
        
        
def reorder(surf_points, add_thick_surf):
    """Reorder points such that consecutive points don't form a line that crosses 
    the surface. 
    
    add_thick: [min, max) values by which to expand the thickness of the surface.
    This is done by moving surface points in the opposite direction of the inside of
    the surface.
    """
    set_points = set(surf_points)
    ordered_points = []
    sampled_add = np.random.uniform(*add_thick_surf)
    while len(set_points) > 0:
        curr_points = sorted(list(set_points)) # Need to sort to get a point in a corner
        curr = curr_points[0]
        prev_dir = None
        while curr is not None:
            set_points.remove(curr)
            ordered_points.append(curr)
            # Get the next point
            shared_x = sorted(list(filter(lambda coord: coord[0] == curr[0], set_points)), key=lambda coord: abs(coord[1] - curr[1]))
            shared_y = sorted(list(filter(lambda coord: coord[1] == curr[1], set_points)), key=lambda coord: abs(coord[0] - curr[0]))
            # Set curr and currdir
            curr = None
            curr_dir = None
            # 1 is vertical direction, 0 is horizontal
            if len(shared_x) != 0 and prev_dir != "1":
                curr = shared_x[0]
                curr_dir = "1"
            if len(shared_y) != 0 and prev_dir != "0":
                curr = shared_y[0]
                curr_dir = "0"

            prev_dir = curr_dir
        ordered_points.append(curr_points[0]) # Make sure to close the surface
    assert set(ordered_points) == set(surf_points)
    return ordered_points
        

def reorder(surf_points, add_thick_surf, canvas_size):
    """Reorder points such that consecutive points don't form a line that crosses 
    the surface. 
    
    add_thick: [min, max) values by which to expand the thickness of the surface.
    This is done by moving surface points in the opposite direction of the inside of
    the surface.
    
    TODO: does not check for overlapping edges which may occur when max >= 1
    """
    assert add_thick_surf[1] <= 1, "Thickness >= 1 may result in edge overlapping"
    set_points = set(surf_points)
    ordered_points = []
    sampled_add = np.random.uniform(*add_thick_surf)
    drawn = np.pad(np.zeros((canvas_size, canvas_size)), ((1, 1), (1, 1)), 'constant', constant_values=(BLACK, BLACK))
    while len(set_points) > 0:
        curr_points = sorted(list(set_points)) # Need to sort to get a point in a corner
        curr = curr_points[0]
        # Always go counter clockwise
        prev_dir = None
        while curr is not None:
            set_points.remove(curr)
            ordered_points.append(curr)
            # Get the next point
            shared_x = sorted(list(filter(lambda coord: coord[0] == curr[0], set_points)),                               key=lambda coord: abs(coord[1] - curr[1]))
            shared_y = sorted(list(filter(lambda coord: coord[1] == curr[1], set_points)),                               key=lambda coord: abs(coord[0] - curr[0]))
            # Set curr and currdir
            prev_curr = curr
            curr = None
            curr_dir = None
            # 1 is vertical direction, 0 is horizontal
            # l is left, r is right, u is up, d is down
            if len(shared_x) != 0 and prev_dir != "1":
                curr = shared_x[0]
                iter_dir = -1 if curr[1] - prev_curr[1] < 0 else 1
                for y_coord in range(prev_curr[1], curr[1] + iter_dir, iter_dir):
                    drawn[curr[0], y_coord] = WHITE
                curr_dir = "1"
            if len(shared_y) != 0 and prev_dir != "0":
                curr = shared_y[0]
                iter_dir = -1 if curr[0] - prev_curr[0] < 0 else 1
                for x_coord in range(prev_curr[0], curr[0] + iter_dir, iter_dir):
                    drawn[x_coord, curr[1]] = WHITE
                curr_dir = "0"
                
            prev_dir = curr_dir
        ordered_points.append(curr_points[0]) # Make sure to close the surface
    # Iterate through ordered points and shift them diagonally depending on neighbors
    shifted_points = []
    for point in ordered_points:
        px, py = point[0], point[1]
        # Left, top, right, bottom neighbors
        adj = np.array([drawn[px, py-1], drawn[px-1, py], 
                        drawn[px, py+1], drawn[px+1, py]])
        shifted = None
        if np.sum(adj) == 2:
            if adj[0]:
                if adj[1]:
                    # Neighbors are left and top, so shift in bottom right
                    shifted = (px + sampled_add, py + sampled_add)
                elif adj[3]:
                    # Neighbors are left and bottom, so shift in top right
                    shifted = (px - sampled_add, py + sampled_add)
            elif adj[2]:
                if adj[1]:
                    # Neighbors are right and top, so shift in bottom left
                    shifted = (px + sampled_add, py - sampled_add)
                elif adj[3]:
                    # Neighbors are right and bottom, so shift in top left
                    shifted = (px - sampled_add, py - sampled_add)
        elif np.sum(adj) == 4:
            # Top left, top right, bottom right, bottom left
            diag_adj = np.array([drawn[px-1, py-1], drawn[px-1, py+1], 
                        drawn[px+1, py+1], drawn[px+1, py-1]])
            if np.sum(diag_adj) == 3:
                min_idx = np.argmin(diag_adj)
                # Shift in direction where there is no diag neighbor
                if min_idx == 0:
                    shifted = (px - sampled_add, py - sampled_add)
                elif min_idx == 1:
                    shifted = (px - sampled_add, py + sampled_add)
                elif min_idx == 2:
                    shifted = (px + sampled_add, py + sampled_add)
                else:
                    shifted = (px + sampled_add, py - sampled_add)
        if shifted is None:
            # If a point has three neighbors, then the shape is too small to add thickness
            shifted_points = ordered_points
            break
        shifted_points.append(shifted)
    return shifted_points


def get_all_surf_points(all_bound, use_only_outer=False):
    all_surf_points = []
    for pixel in all_bound:
        surf_points = get_surf_points(pixel, all_bound, use_only_outer=use_only_outer)
        all_surf_points.extend(surf_points)
    return all_surf_points

            
def povray_transform(all_surf_points, canvas_size):
    transformed = []
    for point in all_surf_points:
        transformed.append((point[1], canvas_size-point[0]))
    return transformed


def get_povray_points(obj_mask, obj_type, add_thick_surf=(0, 0)):
    # Shape mask is the mask of one shape
    obj = to_np_array(obj_mask)
    canvas_size = obj.shape[0]
    obj_contours, contour_pixels = iter_moore(deepcopy(obj))
    all_bound = set(obj_contours[0])
    all_surf_points = get_all_surf_points(all_bound, 
                                          use_only_outer = (True if obj_type == "rectangleSolid" else False))
    ordered = povray_transform(reorder(all_surf_points, add_thick_surf, canvas_size), canvas_size)
    return ordered


def get_rand_color():
    rgb_array = np.random.rand(3)
    if not np.any(rgb_array >= COLOR_THRESH):
        idx = np.random.randint(0, 3)
        rgb_array[idx] = COLOR_THRESH
    return rgb_array


def convert_example(example, args, tempfile):
    # Converts a single example in a dataset
    center = example[0].shape[1] / 2
    camera = Camera( 'location', [center, center * 2.5, center - DEGREE_OFFSET], 
                    'look_at', [center, 0, center] )
    light = LightSource( [center, center, center], 'color', [1,1,1] )
    info = example[3]
    id_object_mask = info['id_object_mask']
    node_id_map = info['node_id_map']
    
    example_3d = list(deepcopy(example))
    example_3d[1] = list(example_3d[1])
    objects_lst = [light]
    # Go through the object spec
    for obj in info['obj_spec']:
        obj_name, obj_desc = obj[0]
        obj_type = obj_desc.split("_")[0]
        obj_mask = id_object_mask[node_id_map[obj_name]].squeeze()
        # Map color to RGB
        obj_img = (obj_mask * example[0])
        if not args.color_map_3d == "random":
            # Get any pixel in the object
            pixel_coord = torch.nonzero(obj_img)[0]
            # The channel dim is the first one
            rgb_color = COLOR_DICT[pixel_coord[0].item()]
        else:
            rgb_color = get_rand_color()
        # Get the object in 3D space 
        ordered = get_povray_points(obj_mask, obj_type, add_thick_surf=args.add_thick_surf)
        prism_depth = 1 + np.random.uniform(*args.add_thick_depth)
        prism = Prism(
            'linear_sweep',
            'linear_spline',
            0, # sweep the following shape from here ...
            prism_depth, # ... up through here
            len(ordered), # The number of points making up the shape ...
            *ordered,
            Texture( Pigment( 'color', 'rgb', rgb_color),
                             DEFAULT_NORMAL,
                             DEFAULT_PHONG),
            'rotate', (0, 0, 0)
        )
        objects_lst.append(prism)
        scene = Scene( camera = camera , # a Camera object
                   objects= [light, prism], # POV-Ray objects (items, lights)
                   included = ["colors.inc"]) # headers that POV-Ray may need
        # Permute from [H, W, C] to [C, H, W]
        rendered = scene.render(width=args.image_size_3d[0], height=args.image_size_3d[1], tempfile=tempfile)
        obj_3d = torch.tensor(rendered, dtype=torch.float32).permute(2, 0, 1)
        obj_3d_mask = torch.any(obj_3d != 0, dim=0).float()
        # Reduce the channel dimension
        example_3d[3]['id_object_mask'][example_3d[3]['node_id_map'][obj_name]] = obj_3d_mask
        for i, gt_mask in enumerate(example[1]):
            if torch.all(torch.eq(obj_mask, gt_mask)):
                example_3d[1][i] = obj_3d_mask.unsqueeze(0)
    # Render the image separately to avoid occlusion issues 
    scene = Scene( camera = camera , # a Camera object
                   objects= objects_lst, # POV-Ray objects (items, lights)
                   included = ["colors.inc"]) # headers that POV-Ray may need
    full_img = torch.tensor(scene.render(width=args.image_size_3d[0], height=args.image_size_3d[1], tempfile=tempfile),
                              dtype=torch.float32).permute(2, 0, 1)
    example_3d[0] = full_img / 255 # RGB values must be in [0, 1] range
    example_3d[1] = tuple(example_3d[1])
    return tuple(example_3d)

    
def iter_examples(seed, tup_dataset, args):
    set_seed(seed)
    dataset_3d = []
    for i, example in enumerate(tup_dataset):
        if i % 200 == 0:
            print(i)
        dataset_3d.append(convert_example(example, args, f"temp{os.getpid()}.pov"))
    return dataset_3d


def convert_babyarc(babyarc_dataset, args):
    if args.num_processes_3d == 1:
        return iter_examples(args.seed_3d, babyarc_dataset, args)
    else:
        assert args.num_processes_3d > 0
        # Create iterable of arguments 
        lst_indices = np.linspace(0, len(babyarc_dataset), num=args.num_processes_3d+1, dtype=np.int32)
        convert_args = zip([args.seed_3d + i for i in range(args.num_processes_3d)], 
                           [[babyarc_dataset[j] for j in range(first, second)] for first, second in zip(lst_indices, lst_indices[1:])],
                           [args] *  args.num_processes_3d)
        pool = Pool(processes=args.num_processes_3d)
        results = pool.starmap(iter_examples, convert_args)
        pool.close()
        pool.join()
        combined_results = []
        for res in results:
            combined_results.extend(res)
        return combined_results


# ## 1.1 Basic POVRay Testing

# In[ ]:


# if __name__ == "__main__":
#     camera = Camera( 'location', [0,2,-5], 'look_at', [0,1,2] )
#     light = LightSource( [2,4,-3], 'color', [1,1,1] )
#     sphere = Sphere( [0,1,2], 2, Texture( Pigment( 'color', [1,0,1] )))
#     scene = Scene( camera = camera , # a Camera object
#                objects= [light, sphere], # POV-Ray objects (items, lights)
#     #            atmospheric = [fog], # Light-interacting objects
#                included = ["colors.inc"]) # headers that POV-Ray may need
#     scene.render('3D/sphere', width=300, height=300)


# In[ ]:


# if __name__ == "__main__":
#     camera = Camera( 'location', [0,0,-20], 'look_at', [0, 0, 0] )
#     light = LightSource( [2,4,-3], 'color', [1,1,1] )
#     prism = Prism(
#         'linear_sweep',
#         'linear_spline',
#         0, # sweep the following shape from here ...
#         1, # ... up through here
#         7, # The number of points making up the shape ...
#         (3,5), (-3,5), (-5,0), (-3,-5), (3, -5), (5,0), (3,5),
#         Texture( Pigment( 'color', [0.1, 0, 0])),
#         'rotate', (90, 15, 0)
#     )
#     scene = Scene( camera = camera , # a Camera object
#                objects= [light, prism], # POV-Ray objects (items, lights)
#                included = ["colors.inc"]) # headers that POV-Ray may need
#     scene.render('3D/prism', width=300, height=300)


# In[ ]:


# if __name__ == "__main__":
#     camera = Camera( 'location', [8, 20, 8], 'look_at', [8, 0, 8] )
#     light = LightSource( [2,4,3], 'color', [1,1,1] )
#     prism1 = Prism(
#         'linear_sweep',
#         'linear_spline',
#         0, # sweep the following shape from here ...
#         1, # ... up through here
#         9, # The number of points making up the shape ...
#         *povray_transform([(3,5), (9,5), (9,10), (3, 10), (3, 5), (4,6), (8,6), (8,9), (4, 9)], 16),
#         Texture( Pigment( 'color', 'rgb', [0.8*i for i in [1.00,0.95,0.8]]),
#                          Normal('bumps', 0.75, 'scale', 0.0125),
#                          DEFAULT_PHONG),
#         'rotate', (0, 0, 0)
#     )
    
#     prism2 = Prism(
#         'linear_sweep',
#         'linear_spline',
#         0, # sweep the following shape from here ...
#         1, # ... up through here
#         9, # The number of points making up the shape ...
#         *povray_transform([(3,5), (9,5), (9,10), (3, 10), (3, 5), (4,6), (8,6), (8,9), (4, 9)], 16),
#         Texture( Pigment( 'color', 'rgb', [0.8*i for i in [1.00,0.95,0.8]]),
#                          Normal('bumps', 0.75, 'scale', 0.0125),
#                          DEFAULT_PHONG),
#         'rotate', (0, 0, 30)
#     )
    
#     prism3 = Prism(
#         'linear_sweep',
#         'linear_spline',
#         0, # sweep the following shape from here ...
#         1, # ... up through here
#         9, # The number of points making up the shape ...
#         *povray_transform([(3,5), (9,5), (9,10), (3, 10), (3, 5), (4,6), (8,6), (8,9), (4, 9)], 16),
#         Texture( Pigment( 'color', 'rgb', [0.8*i for i in [1.00,0.95,0.8]]),
#                          Normal('bumps', 0.75, 'scale', 0.0125),
#                          DEFAULT_PHONG),
#         'rotate', (30, 0, 0)
#     )
    
#     prism4 = Prism(
#         'linear_sweep',
#         'linear_spline',
#         0, # sweep the following shape from here ...
#         1, # ... up through here
#         9, # The number of points making up the shape ...
#         *povray_transform([(3,5), (9,5), (9,10), (3, 10), (3, 5), (4,6), (8,6), (8,9), (4, 9)], 16),
#         Texture( Pigment( 'color', 'rgb', [0.8*i for i in [1.00,0.95,0.8]]),
#                           DEFAULT_PHONG),
#         'rotate', (0, 0, 0)
#     )
    
#     scene = Scene( camera = camera , # a Camera object
#                objects= [light, prism1], # POV-Ray objects (items, lights)
#                included = ["colors.inc"]) # headers that POV-Ray may need
#     scene.render('3D/prism_hollow', width=300, height=500)
    
#     scene = Scene( camera = camera , # a Camera object
#                objects= [light, prism2], # POV-Ray objects (items, lights)
#                included = ["colors.inc"]) # headers that POV-Ray may need
#     scene.render('3D/prism_hollow2', width=300, height=500)
    
#     scene = Scene( camera = camera , # a Camera object
#                objects= [light, prism3], # POV-Ray objects (items, lights)
#                included = ["colors.inc"]) # headers that POV-Ray may need
#     scene.render('3D/prism_hollow3', width=300, height=500)
    
#     scene = Scene( camera = camera , # a Camera object
#                objects= [light, prism4], # POV-Ray objects (items, lights)
#                included = ["colors.inc"]) # headers that POV-Ray may need
#     scene.render('3D/prism_hollow4', width=300, height=500)


# In[ ]:


# if __name__ == "__main__":
#     camera = Camera( 'location', [8, 20, 8 - 16.8], 'look_at', [8, 0, 8] )
#     light = LightSource( [8,8,8], 'color', [1,1,1] )
#     prefix = "cam_"
#     prism1 = Prism(
#         'linear_sweep',
#         'linear_spline',
#         0, # sweep the following shape from here ...
#         1, # ... up through here
#         9, # The number of points making up the shape ...
#         *povray_transform([(3,5), (9,5), (9,10), (3, 10), (3, 5), (4,6), (8,6), (8,9), (4, 9)], 16),
#         Texture( Pigment( 'color', 'rgb', [0, 0, 1]),
#                          Normal('bumps', 0.75, 'scale', 0.0125),
#                          DEFAULT_PHONG),
#         'rotate', (0, 0, 0)
#     )
    
#     prism2 = Prism(
#         'linear_sweep',
#         'linear_spline',
#         0, # sweep the following shape from here ...
#         1, # ... up through here
#         9, # The number of points making up the shape ...
#         *povray_transform([(3,5), (9,5), (9,10), (3, 10), (3, 5), (4,6), (8,6), (8,9), (4, 9)], 16),
#         Texture( Pigment( 'color', 'rgb', [1, 0, 0]),
#                          Normal('bumps', 0.75, 'scale', 0.0125),
#                          DEFAULT_PHONG),
#         'rotate', (0, 0, 0)
#     )
    
#     prism3 = Prism(
#         'linear_sweep',
#         'linear_spline',
#         0, # sweep the following shape from here ...
#         1, # ... up through here
#         9, # The number of points making up the shape ...
#         *povray_transform([(9,5), (15,5), (15,10), (9, 10), (9, 5), (10,6), (14,6), (14,9), (10, 9)], 16),
#         Texture( Pigment( 'color', 'rgb', [1, 0, 0]),
#                          Normal('bumps', 0.75, 'scale', 0.0125),
#                          DEFAULT_PHONG),
#         'rotate', (0, 0, 0)
#     )
    
#     prism4 = Prism(
#         'linear_sweep',
#         'linear_spline',
#         0, # sweep the following shape from here ...
#         1, # ... up through here
#         9, # The number of points making up the shape ...
#         *povray_transform([(3,10), (9,10), (9,15), (3, 15), (3, 10), (4,11), (8,11), (8,14), (4, 14)], 16),
#         Texture( Pigment( 'color', 'rgb', [1, 0, 0]),
#                          Normal('bumps', 0.75, 'scale', 0.0125),
#                          DEFAULT_PHONG),
#         'rotate', (0, 0, 0)
#     )
    
#     scene = Scene( camera = camera , # a Camera object
#                objects= [light, prism1], # POV-Ray objects (items, lights)
#                included = ["colors.inc"]) # headers that POV-Ray may need
#     scene.render(f'3D/{prefix}prism_hollow', width=300, height=500)
    
#     scene = Scene( camera = camera , # a Camera object
#                objects= [light, prism2], # POV-Ray objects (items, lights)
#                included = ["colors.inc"]) # headers that POV-Ray may need
#     scene.render(f'3D/{prefix}prism_hollow2', width=300, height=500)
    
#     scene = Scene( camera = camera , # a Camera object
#                objects= [light, prism3], # POV-Ray objects (items, lights)
#                included = ["colors.inc"]) # headers that POV-Ray may need
#     scene.render(f'3D/{prefix}prism_hollow3', width=300, height=500)
    
#     scene = Scene( camera = camera , # a Camera object
#                objects= [light, prism4], # POV-Ray objects (items, lights)
#                included = ["colors.inc"]) # headers that POV-Ray may need
#     scene.render(f'3D/{prefix}prism_hollow4', width=300, height=500)


# ## 1.2 Test Letters

# In[ ]:


# from reasoning.experiments.concept_energy import get_dataset, ConceptDataset, ConceptCompositionDataset


# In[ ]:


# if __name__ == "__main__":
#     args = init_args({
#             "dataset": "c-Eshape+Cshape+Fshape+Tshape^Eshape",
#             "seed": 1,
#             "n_examples":10,
#             "canvas_size": 16,
#             "rainbow_prob": 0.,
#             "color_avail": "1,2",
#             "w_type": "mask",
#             "max_n_distractors": 2,
#         })
#     dataset, args = get_dataset(args, is_load=False)
#     print(dataset[0][3])
#     print(dataset[0][3]['obj_spec'])
#     print(dataset[1][3]['obj_spec'])
#     print(dataset[3][3]['obj_spec'])
#     print(dataset[4][3]['obj_spec'])
#     print(dataset[5][3]['obj_spec'])
#     print(dataset[9][3]['obj_spec'])
#     obj_dct = {}
#     obj_dct["E1"] = dataset[0][3]['id_object_mask'][dataset[0][3]['node_id_map']['obj_1']], 'Eshape'
#     obj_dct["F1"] = dataset[0][3]['id_object_mask'][dataset[0][3]['node_id_map']['obj_0']], 'Fshape'
#     obj_dct["E2"] = dataset[1][3]['id_object_mask'][0], 'Eshape'
#     obj_dct["C1"] = dataset[3][3]['id_object_mask'][dataset[3][3]['node_id_map']['obj_0']], 'Cshape'
#     obj_dct["F2"] = dataset[3][3]['id_object_mask'][dataset[3][3]['node_id_map']['obj_1']], 'Fshape'
#     obj_dct["E3"] = dataset[3][3]['id_object_mask'][dataset[3][3]['node_id_map']['obj_2']], 'Eshape'
#     obj_dct["T1"] = dataset[4][3]['id_object_mask'][dataset[4][3]['node_id_map']['obj_1']], 'Tshape'


# In[ ]:


# if __name__ == "__main__":
#     for obj_name, obj_tup in obj_dct.items():
#         print(obj_name)
#         print(obj_tup)
#         ordered = get_povray_points(*obj_tup)
#         camera = Camera( 'location', [8, 20, 8], 'look_at', [8, 0, 8] )
#         light = LightSource( [2,4,-3], 'color', [1,1,1] )
#         prism = Prism(
#             'linear_sweep',
#             'linear_spline',
#             0, # sweep the following shape from here ...
#             1, # ... up through here
#             len(ordered), # The number of points making up the shape ...
#             *ordered,
#             Texture( Pigment( 'color', 'rgb', [0.8*i for i in [1.00,0.95,0.8]]),
#                              Normal('bumps', 0.75, 'scale', 0.0125),
#                              Finish('phong', 0.1)),
#             'rotate', (0, 0, 0)
#         )
#         scene = Scene( camera = camera , # a Camera object
#                    objects= [light, prism], # POV-Ray objects (items, lights)
#                    included = ["colors.inc"]) # headers that POV-Ray may need
#         scene.render(f"3D/{obj_name}", width=500, height=500)
        
#         # Test thickness
#         ordered = get_povray_points(*obj_tup, (0.5, 0.5))
#         camera = Camera( 'location', [8, 20, 8], 'look_at', [8, 0, 8] )
#         light = LightSource( [2,4,-3], 'color', [1,1,1] )
#         prism = Prism(
#             'linear_sweep',
#             'linear_spline',
#             0, # sweep the following shape from here ...
#             1, # ... up through here
#             len(ordered), # The number of points making up the shape ...
#             *ordered,
#             Texture( Pigment( 'color', 'rgb', [0.8*i for i in [1.00,0.95,0.8]]),
#                              Normal('bumps', 0.75, 'scale', 0.0125),
#                              Finish('phong', 0.1)),
#             'rotate', (0, 0, 0)
#         )
#         scene = Scene( camera = camera , # a Camera object
#                    objects= [light, prism], # POV-Ray objects (items, lights)
#                    included = ["colors.inc"]) # headers that POV-Ray may need
#         scene.render(f"3D/thick_{obj_name}", width=500, height=500)


# In[ ]:


# if __name__ == "__main__":
#     args = init_args({
#         "dataset": "c-Ashape",
#         "seed": 1,
#         "n_examples": 10,
#         "canvas_size": 16,
#         "rainbow_prob": 0.,
#         "color_avail": "1,2",
#         "w_type": "image+mask",
#         "max_n_distractors": 2,
#     })
#     dataset, args = get_dataset(args, is_load=True)
#     print(dataset[0][3]['obj_spec'])
#     print(dataset[1][3]['obj_spec'])
#     print(dataset[3][3]['obj_spec'])
#     print(dataset[4][3]['obj_spec'])
#     print(dataset[5][3]['obj_spec'])
#     print(dataset[9][3]['obj_spec'])
#     obj_dct ={}
#     obj_dct["A1"] = dataset[0][3]['id_object_mask'][dataset[0][3]['node_id_map']['obj_0']], 'Ashape'
#     obj_dct["A2"] = dataset[1][3]['id_object_mask'][dataset[1][3]['node_id_map']['obj_1']], 'Ashape'


# In[ ]:


# if __name__ == "__main__":
#     for obj_name, obj_tup in obj_dct.items():
#         print(obj_name)
#         print(obj_tup)
#         ordered = get_povray_points(*obj_tup)
#         camera = Camera( 'location', [8, 20, 8], 'look_at', [8, 0, 8] )
#         light = LightSource( [2,4,-3], 'color', [1,1,1] )
#         prism = Prism(
#             'linear_sweep',
#             'linear_spline',
#             0, # sweep the following shape from here ...
#             1, # ... up through here
#             len(ordered), # The number of points making up the shape ...
#             *ordered,
#             Texture( Pigment( 'color', 'rgb', [0.8*i for i in [1.00,0.95,0.8]]),
#                              Normal('bumps', 0.75, 'scale', 0.0125),
#                              Finish('phong', 0.1)),
#             'rotate', (0, 0, 0)
#         )
#         scene = Scene( camera = camera , # a Camera object
#                    objects= [light, prism], # POV-Ray objects (items, lights)
#                    included = ["colors.inc"]) # headers that POV-Ray may need
#         scene.render(f"3D/{obj_name}", width=500, height=500)
        
#         # Test thickness
#         ordered = get_povray_points(*obj_tup, (0.5, 0.5))
#         camera = Camera( 'location', [8, 20, 8], 'look_at', [8, 0, 8] )
#         light = LightSource( [2,4,-3], 'color', [1,1,1] )
#         prism = Prism(
#             'linear_sweep',
#             'linear_spline',
#             0, # sweep the following shape from here ...
#             1, # ... up through here
#             len(ordered), # The number of points making up the shape ...
#             *ordered,
#             Texture( Pigment( 'color', 'rgb', [0.8*i for i in [1.00,0.95,0.8]]),
#                              Normal('bumps', 0.75, 'scale', 0.0125),
#                              Finish('phong', 0.1)),
#             'rotate', (0, 0, 0)
#         )
#         scene = Scene( camera = camera , # a Camera object
#                    objects= [light, prism], # POV-Ray objects (items, lights)
#                    included = ["colors.inc"]) # headers that POV-Ray may need
#         scene.render(f"3D/thick_{obj_name}", width=500, height=500)


# ## 1.3 Test ARC Shapes

# In[ ]:


# from reasoning.experiments.concept_energy import get_dataset, ConceptDataset, ConceptCompositionDataset


# In[ ]:


# if __name__ == "__main__":
#     args = init_args({
#         "dataset": "c-Line+Lshape+Rect+RectSolid",
#         "seed": 1,
#         "n_examples": 20,
#         "canvas_size": 16,
#         "rainbow_prob": 0.,
#         "color_avail": "1,2",
#         "w_type": "image+mask",
#         "max_n_distractors": 2,
#     })
#     dataset, args = get_dataset(args, is_load=True)
#     print(dataset[0][3]['obj_spec'])
#     print(dataset[1][3]['obj_spec'])
#     print(dataset[3][3]['obj_spec'])
#     print(dataset[4][3]['obj_spec'])
#     print(dataset[15][3]['obj_spec'])
#     print(dataset[16][3]['obj_spec'])
#     obj_dct = {}
#     obj_dct["rect_solid1"] = dataset[0][3]['id_object_mask'][dataset[0][3]['node_id_map']['obj_1']], "rectangleSolid"
#     obj_dct["rect1"] = dataset[0][3]['id_object_mask'][dataset[0][3]['node_id_map']['obj_0']], "rectangle"
#     obj_dct["LShape1"] = dataset[3][3]['id_object_mask'][dataset[3][3]['node_id_map']['obj_0']], "Lshape"
#     obj_dct["Line1"] = dataset[15][3]['id_object_mask'][dataset[15][3]['node_id_map']['obj_0']], "line"


# In[ ]:


# if __name__ == "__main__":
#     for obj_name, obj_tup in obj_dct.items():
#         print(obj_name)
#         print(obj_tup)
#         ordered = get_povray_points(*obj_tup, (0, 0))
#         camera = Camera( 'location', [8, 20, 8], 'look_at', [8, 0, 8] )
#         light = LightSource( [2,4,-3], 'color', [1,1,1] )
#         prism = Prism(
#             'linear_sweep',
#             'linear_spline',
#             0, # sweep the following shape from here ...
#             1, # ... up through here
#             len(ordered), # The number of points making up the shape ...
#             *ordered,
#             Texture( Pigment( 'color', 'rgb', [0.8*i for i in [1.00,0.95,0.8]]),
#                              Normal('bumps', 0.75, 'scale', 0.0125),
#                              Finish('phong', 0.1)),
#             'rotate', (0, 0, 0)
#         )
#         scene = Scene( camera = camera , # a Camera object
#                    objects= [light, prism], # POV-Ray objects (items, lights)
#                    included = ["colors.inc"]) # headers that POV-Ray may need
#         scene.render(f"3D/{obj_name}", width=500, height=500)
        
#         # Test thickness
#         ordered = get_povray_points(*obj_tup, (0.5, 0.5))
#         camera = Camera( 'location', [8, 20, 8], 'look_at', [8, 0, 8] )
#         light = LightSource( [2,4,-3], 'color', [1,1,1] )
#         prism = Prism(
#             'linear_sweep',
#             'linear_spline',
#             0, # sweep the following shape from here ...
#             1, # ... up through here
#             len(ordered), # The number of points making up the shape ...
#             *ordered,
#             Texture( Pigment( 'color', 'rgb', [0.8*i for i in [1.00,0.95,0.8]]),
#                              Normal('bumps', 0.75, 'scale', 0.0125),
#                              Finish('phong', 0.1)),
#             'rotate', (0, 0, 0)
#         )
#         scene = Scene( camera = camera , # a Camera object
#                    objects= [light, prism], # POV-Ray objects (items, lights)
#                    included = ["colors.inc"]) # headers that POV-Ray may need
#         scene.render(f"3D/thick_{obj_name}", width=500, height=500)


# In[ ]:


# if __name__ == "__main__":
#     args = init_args({
#         "dataset": "c-Line",
#         "seed": 1,
#         "n_examples": 10,
#         "canvas_size": 16,
#         "rainbow_prob": 0.,
#         "color_avail": "1,2",
#         "w_type": "image+mask",
#         "max_n_distractors": 2,
#     })
#     dataset, args = get_dataset(args, is_load=True)
#     print(dataset[0][3]['obj_spec'])
#     print(dataset[1][3]['obj_spec'])
#     print(dataset[3][3]['obj_spec'])
#     print(dataset[4][3]['obj_spec'])
#     obj_dct = {}
#     obj_dct["Line2"] = dataset[0][3]['id_object_mask'][dataset[0][3]['node_id_map']['obj_1']], "line"
#     obj_dct["Line3"] = dataset[0][3]['id_object_mask'][dataset[0][3]['node_id_map']['obj_0']], "line"


# In[ ]:


# if __name__ == "__main__":
#     for obj_name, obj_tup in obj_dct.items():
#         print(obj_name)
#         print(obj_tup)
#         ordered = get_povray_points(*obj_tup)
#         camera = Camera( 'location', [8, 20, 8], 'look_at', [8, 0, 8] )
#         light = LightSource( [2,4,-3], 'color', [1,1,1] )
#         prism = Prism(
#             'linear_sweep',
#             'linear_spline',
#             0, # sweep the following shape from here ...
#             1, # ... up through here
#             len(ordered), # The number of points making up the shape ...
#             *ordered,
#             Texture( Pigment( 'color', 'rgb', [0.8*i for i in [1.00,0.95,0.8]]),
#                              Normal('bumps', 0.75, 'scale', 0.0125),
#                              Finish('phong', 0.1)),
#             'rotate', (0, 0, 0)
#         )
#         scene = Scene( camera = camera , # a Camera object
#                    objects= [light, prism], # POV-Ray objects (items, lights)
#                    included = ["colors.inc"]) # headers that POV-Ray may need
#         scene.render(f"3D/{obj_name}", width=500, height=500)
        
#         # Test thickness
#         ordered = get_povray_points(*obj_tup, (0.5, 0.5))
#         camera = Camera( 'location', [8, 20, 8], 'look_at', [8, 0, 8] )
#         light = LightSource( [2,4,-3], 'color', [1,1,1] )
#         prism = Prism(
#             'linear_sweep',
#             'linear_spline',
#             0, # sweep the following shape from here ...
#             1, # ... up through here
#             len(ordered), # The number of points making up the shape ...
#             *ordered,
#             Texture( Pigment( 'color', 'rgb', [0.8*i for i in [1.00,0.95,0.8]]),
#                              Normal('bumps', 0.75, 'scale', 0.0125),
#                              Finish('phong', 0.1)),
#             'rotate', (0, 0, 0)
#         )
#         scene = Scene( camera = camera , # a Camera object
#                    objects= [light, prism], # POV-Ray objects (items, lights)
#                    included = ["colors.inc"]) # headers that POV-Ray may need
#         scene.render(f"3D/thick_{obj_name}", width=500, height=500)


# ## 1.4 Test Dataset Generation

# In[ ]:


# from reasoning.experiments.concept_energy import get_dataset, ConceptDataset, ConceptCompositionDataset


# ### 1.4.1 Concepts

# In[ ]:


# if __name__ == "__main__":
#     args = init_args({
#         "dataset": "c-Line",
#         "seed": 1,
#         "n_examples": 10,
#         "canvas_size": 16,
#         "rainbow_prob": 0.,
#         "color_avail": "1,2",
#         "w_type": "image+mask",
#         "max_n_distractors": 2,
#     })
#     dataset, args = get_dataset(args, is_load=True)
#     convert_args = init_args({
#         "image_size_3d": (128, 128),
#         "color_map_3d": "same",
#         "num_processes_3d": 10,
#     })
#     start_time = time.time()
#     dataset_dct = convert_babyarc(dataset, convert_args)
#     print(f"Time to generate: {time.time() - start_time}")


# In[ ]:


# print(dataset[0][0].argmax(0))
# print(dataset_dct[0][0].shape)
# visualize_matrices(dataset_dct[0][0].unsqueeze(0), use_color_dict=False, images_per_row=6)
# plot_matrices(torch.stack(list(dataset_dct[0][3]['id_object_mask'].values())), images_per_row=6)


# In[ ]:


# print(dataset[3][0].argmax(0))
# print(dataset_dct[3][0].shape)
# visualize_matrices(dataset_dct[3][0].unsqueeze(0), use_color_dict=False, images_per_row=6)
# plot_matrices(torch.stack(list(dataset_dct[3][3]['id_object_mask'].values())), images_per_row=6)


# In[ ]:


# if __name__ == "__main__":
#     args = init_args({
#         "dataset": "c-Eshape+Cshape+Fshape+Tshape+Ashape^Eshape",
#         "seed": 1,
#         "n_examples": 400,
#         "canvas_size": 16,
#         "rainbow_prob": 0.,
#         "color_avail": "1,2",
#         "w_type": "image+mask",
#         "max_n_distractors": 2,
#     })
#     dataset, args = get_dataset(args, is_load=True)
#     convert_args = init_args({
#         "image_size_3d": (256, 256),
#         "color_map_3d": "random",
#         "num_processes_3d": 10,
#         "add_thick_surf": (0, 0.5),
#         "add_thick_depth": (0.5, 0.5),
#         "seed_3d": 42,
#     })
#     start_time = time.time()
#     dataset_dct = convert_babyarc(dataset, convert_args)
#     print(f"Time to generate: {time.time() - start_time}")


# In[ ]:


# print(dataset[0][0].argmax(0))
# print(dataset_dct[0][0].shape)
# visualize_matrices(dataset_dct[0][0].unsqueeze(0), use_color_dict=False, images_per_row=6)
# plot_matrices(torch.stack(list(dataset_dct[0][3]['id_object_mask'].values())), images_per_row=6)


# In[ ]:


# print(dataset[2][0].argmax(0))
# print(dataset_dct[2][0].shape)
# visualize_matrices(dataset_dct[2][0].unsqueeze(0), use_color_dict=False, images_per_row=6)
# plot_matrices(torch.stack(list(dataset_dct[2][3]['id_object_mask'].values())), images_per_row=6)


# In[ ]:


# print(dataset[19][0].argmax(0))
# print(dataset_dct[19][0].shape)
# visualize_matrices(dataset_dct[19][0].unsqueeze(0), use_color_dict=False, images_per_row=6)
# plot_matrices(torch.stack(list(dataset_dct[19][3]['id_object_mask'].values())), images_per_row=6)


# In[ ]:


# if __name__ == "__main__":
#     args = init_args({
#         "dataset": "c-Ashape",
#         "seed": 1,
#         "n_examples": 10,
#         "canvas_size": 16,
#         "rainbow_prob": 0.,
#         "color_avail": "1,2",
#         "w_type": "image+mask",
#         "max_n_distractors": 2,
#     })
#     dataset, args = get_dataset(args, is_load=True)
#     convert_args = init_args({
#         "image_size_3d": (256, 256),
#         "color_map_3d": "same",
#         "num_processes_3d": 10,
#         "seed_3d": 42,
#     })
#     start_time = time.time()
#     dataset_dct = convert_babyarc(dataset, convert_args)
#     print(f"Time to generate: {time.time() - start_time}")


# In[ ]:


# print(deepcopy(dataset[0][0]).argmax(0))
# print(dataset_dct[0][0].shape)
# visualize_matrices(dataset_dct[0][0].unsqueeze(0), use_color_dict=False, images_per_row=6)
# plot_matrices(torch.stack(list(dataset_dct[0][3]['id_object_mask'].values())), images_per_row=6)


# In[ ]:


# print(deepcopy(dataset[2][0]).argmax(0))
# print(dataset_dct[2][0].shape)
# visualize_matrices(dataset_dct[2][0].unsqueeze(0), use_color_dict=False, images_per_row=6)
# plot_matrices(torch.stack(list(dataset_dct[2][3]['id_object_mask'].values())), images_per_row=6)


# In[ ]:


#print(deepcopy(dataset[3][0]).argmax(0))
# print(dataset_dct[3][0].shape)
# visualize_matrices(dataset_dct[3][0].unsqueeze(0), use_color_dict=False, images_per_row=6)
# plot_matrices(torch.stack(list(dataset_dct[3][3]['id_object_mask'].values())), images_per_row=6)


# In[ ]:


# if __name__ == "__main__":
#     args = init_args({
#         "dataset": "c-Line+Lshape+Rect+RectSolid",
#         "seed": 1,
#         "n_examples": 400,
#         "canvas_size": 16,
#         "rainbow_prob": 0.,
#         "color_avail": "1,2",
#         "w_type": "image+mask",
#         "max_n_distractors": 2,
#     })
#     dataset, args = get_dataset(args, is_load=True)
#     convert_args = init_args({
#         "image_size_3d": (256, 256),
#         "color_map_3d": "random",
#         "num_processes_3d": 20,
#         "add_thick_surf": (0, 0.5),
#         "add_thick_depth": (0.5, 0.5),
#         "seed_3d": 42,
#     })
#     start_time = time.time()
#     dataset_dct = convert_babyarc(dataset, convert_args)
#     print(f"Time to generate: {time.time() - start_time}")


# In[ ]:


# print(dataset[0][0].argmax(0))
# print(dataset_dct[0][0].shape)
# visualize_matrices(dataset_dct[0][0].unsqueeze(0), use_color_dict=False, images_per_row=6)
# plot_matrices(torch.stack(list(dataset_dct[0][3]['id_object_mask'].values())), images_per_row=6)


# ### 1.4.2 Relations

# In[ ]:


# if __name__ == "__main__":
#     relation_args = init_args({
#         "dataset": "c-Parallel+Vertical",
#         "seed": 1,
#         "n_examples": 10,
#         "canvas_size": 16,
#         "rainbow_prob": 0.,
#         "color_avail": "1,2",
#         "w_type": "image+mask",
#         "max_n_distractors": 2,
#     })
#     relation_dataset, args = get_dataset(relation_args, is_load=True)
#     convert_args = init_args({
#         "image_size_3d": (256, 256),
#         "color_map_3d": "same",
#         "num_processes_3d": 10,
#         "add_thick_surf": (0.5, 0.5),
#         "add_thick_depth": (0.5, 0.5),
#         "seed_3d": 42,
#     })
#     start_time = time.time()
#     relation_dataset_dct = convert_babyarc(relation_dataset, convert_args)
#     print(f"Time to generate: {time.time() - start_time}")


# In[ ]:


# print(relation_dataset[0][0].argmax(0))
# print(relation_dataset_dct[0][0].shape)
# print(relation_dataset[0][1])
# print(relation_dataset_dct[0][1])
# visualize_matrices(relation_dataset_dct[0][0].unsqueeze(0), use_color_dict=False, images_per_row=6)
# plot_matrices(torch.stack(list(relation_dataset_dct[0][3]['id_object_mask'].values())), images_per_row=6)
# plot_matrices(torch.stack(relation_dataset_dct[0][1]), images_per_row=6)


# In[ ]:


# print(relation_dataset[1][0].argmax(0))
# print(relation_dataset_dct[1][0].shape)
# print(relation_dataset[1][1])
# visualize_matrices(relation_dataset_dct[1][0].unsqueeze(0), use_color_dict=False, images_per_row=6)
# plot_matrices(torch.stack(list(relation_dataset_dct[1][3]['id_object_mask'].values())), images_per_row=6)
# plot_matrices(torch.stack(relation_dataset_dct[1][1]), images_per_row=6)


# In[ ]:


# if __name__ == "__main__":
#     relation_args = init_args({
#         "dataset": "c-Parallel+Vertical",
#         "seed": 1,
#         "n_examples": 400,
#         "canvas_size": 16,
#         "rainbow_prob": 0.,
#         "color_avail": "1,2",
#         "w_type": "image+mask",
#         "max_n_distractors": 2,
#     })
#     relation_dataset, args = get_dataset(relation_args, is_load=False)
#     # Test random
#     convert_args = init_args({
#         "image_size_3d": (256, 256),
#         "color_map_3d": "same",
#         "num_processes_3d": 20,
#         "add_thick_surf": (0, 0.5),
#         "add_thick_depth": (0, 0.5),
#         "seed_3d": 42,
#     })
#     start_time = time.time()
#     relation_dataset_dct = convert_babyarc(relation_dataset, convert_args)
#     print(f"Time to generate: {time.time() - start_time}")


# In[ ]:


# print(relation_dataset[0][0].argmax(0))
# print(relation_dataset_dct[0][0].shape)
# print(relation_dataset[0][1])
# print(relation_dataset_dct[0][1])
# visualize_matrices(relation_dataset_dct[0][0].unsqueeze(0), use_color_dict=False, images_per_row=6)
# plot_matrices(torch.stack(list(relation_dataset_dct[0][3]['id_object_mask'].values())), images_per_row=6)
# plot_matrices(torch.stack(relation_dataset_dct[0][1]), images_per_row=6)


# In[ ]:


# print(relation_dataset[5][0].argmax(0))
# print(relation_dataset_dct[5][0].shape)
# print(relation_dataset[5][1])
# visualize_matrices(relation_dataset_dct[5][0].unsqueeze(0), use_color_dict=False, images_per_row=6)
# plot_matrices(torch.stack(list(relation_dataset_dct[5][3]['id_object_mask'].values())), images_per_row=6)
# plot_matrices(torch.stack(relation_dataset_dct[5][1]), images_per_row=6)


# In[ ]:


# print(relation_dataset[10][0].argmax(0))
# print(relation_dataset_dct[10][0].shape)
# print(relation_dataset[10][1])
# visualize_matrices(relation_dataset_dct[10][0].unsqueeze(0), use_color_dict=False, images_per_row=6)
# plot_matrices(torch.stack(list(relation_dataset_dct[10][3]['id_object_mask'].values())), images_per_row=6)
# plot_matrices(torch.stack(relation_dataset_dct[10][1]), images_per_row=6)


# ## 1.5 Test Dataset Loading

# In[ ]:


# from reasoning.experiments.concept_energy import get_dataset, ConceptDataset, ConceptCompositionDataset


# ### 1.5.1 Concepts

# In[ ]:


# if __name__ == "__main__":
#     args = init_args({
#         "dataset": "y-Line",
#         "n_examples": 440,
#         "canvas_size": 16,
#         "rainbow_prob": 0.,
#         "color_avail": "1,2",
#         "w_type": "image+mask",
#         "max_n_distractors": 2,
#         # 3d args
#         "num_processes_3d": 20,
#         "image_size_3d": (256, 256),
#         "color_map_3d": "same",
#         "add_thick_surf": (0, 0.5),
#         "add_thick_depth": (0, 0.5),
#         "seed_3d": 42,
#     })
#     dataset, args = get_dataset(args, is_load=True)


# In[ ]:


# visualize_matrices(dataset[0][0].unsqueeze(0), use_color_dict=False, images_per_row=6)
# plot_matrices(torch.stack(list(dataset[0][3]['id_object_mask'].values())), images_per_row=6)
# plot_matrices(torch.stack(dataset[0][1]), images_per_row=6)


# In[ ]:


# visualize_matrices(dataset[7][0].unsqueeze(0), use_color_dict=False, images_per_row=6)
# plot_matrices(torch.stack(list(dataset[7][3]['id_object_mask'].values())), images_per_row=6)
# plot_matrices(torch.stack(dataset[7][1]), images_per_row=6)


# ### 1.5.2 Relations

# In[ ]:


# if __name__ == "__main__":
#     relation_args = init_args({
#         "dataset": "y-Parallel+Vertical",
#         "n_examples": 440,
#         "canvas_size": 16,
#         "rainbow_prob": 0.,
#         "color_avail": "1,2",
#         "w_type": "image+mask",
#         "max_n_distractors": 2,
#         # 3d args
#         "num_processes_3d": 20,
#         "image_size_3d": (256, 256),
#         "color_map_3d": "same",
#         "add_thick_surf": (0, 0.5),
#         "add_thick_depth": (0, 0.5),
#         "seed_3d": 42,
#     })
#     relation_dataset, args = get_dataset(relation_args, is_load=True)


# In[ ]:


# visualize_matrices(relation_dataset[0][0].unsqueeze(0), use_color_dict=False, images_per_row=6)
# plot_matrices(torch.stack(list(relation_dataset[0][3]['id_object_mask'].values())), images_per_row=6)
# plot_matrices(torch.stack(relation_dataset[0][1]), images_per_row=6)


# In[ ]:


# visualize_matrices(relation_dataset[5][0].unsqueeze(0), use_color_dict=False, images_per_row=6)
# plot_matrices(torch.stack(list(relation_dataset[5][3]['id_object_mask'].values())), images_per_row=6)
# plot_matrices(torch.stack(relation_dataset[5][1]), images_per_row=6)


# In[ ]:




