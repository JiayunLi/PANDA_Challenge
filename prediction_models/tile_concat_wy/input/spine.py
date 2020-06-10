import os,sys
import skimage.io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections
from collections import OrderedDict
from skimage.color import rgb2hsv
from skimage.measure import regionprops
from skimage.morphology import reconstruction, thin, skeletonize
from PIL import Image, ImageDraw

def skeleton_endpoints(skel):
    # Find row and column locations that are non-zero
    (rows,cols) = np.nonzero(skel)
    # Initialize empty list of co-ordinates
    skel_coords = []
    # For each non-zero pixel...
    for (r,c) in zip(rows,cols):
        # Extract an 8-connected neighbourhood
        (col_neigh,row_neigh) = np.meshgrid(np.array([c-1,c,c+1]), np.array([r-1,r,r+1]))
        # Cast to int to index into image
        col_neigh = col_neigh.astype('int')
        row_neigh = row_neigh.astype('int')
        # Convert into a single 1D array and check for non-zero locations
        pix_neighbourhood = skel[row_neigh,col_neigh].ravel() != 0
        # If the number of non-zero locations equals 2, add this to
        # our list of co-ordinates
        if np.sum(pix_neighbourhood) == 2:
            skel_coords.append((r,c))
    return skel_coords

def bfs(grid, start, goal):
    width, height = grid.shape
    queue = collections.deque([[start]])
    seen = set([start])
    steps = np.zeros_like(grid)
    while queue:
        for i in range(len(queue)):
            path = queue.popleft()
            x, y = path[-1]
            if (x,y) == goal:
                return steps, path
            for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1), (x+1, y+1), (x-1, y-1), (x+1, y-1), (x-1, y+1)):
                if 0 <= x2 < width and 0 <= y2 < height and grid[x2][y2] == 1 and (x2, y2) not in seen:
                    queue.append(path + [(x2, y2)])
                    seen.add((x2,y2))
                    steps[x2, y2] = steps[x,y] + 1
    return -1, None

def spine(img, **kwargs):
    result = OrderedDict()
    thresh = kwargs['thresh'] if 'thresh' in kwargs else 0.9
    bthresh = kwargs['bthresh'] if 'bthresh' in kwargs else 20
    min_size = kwargs['min_size'] if 'min_size' in kwargs else 40
    tan_size = kwargs['tan_size'] if 'tan_size' in kwargs else 4
    patch_size = kwargs['patch_size'] if 'patch_size' in kwargs else 128
    step_size = kwargs['step_size'] if 'step_size' in kwargs else int(0.8 * patch_size)
    slide_thresh = kwargs['slide_thresh'] if 'slide_thresh' in kwargs else 0.6

    ## first step: find the spine
    im2 = img
    im3 = np.mean(im2, 2) # returns the mean over channel dimension
    im4 = im3 < thresh * np.max(im3) # True: content, False white pixel
    im4[im2[:, :, 1] > im3] = 0  # areas that are too greenish
    im4[im2[:, :, 2] > im2[:, :, 0] + bthresh] = 0  # areas that are too blueish
    im4 = im4.astype('int')

    im5 = im4
    cc = regionprops(im4)
    for prop in cc:
        if prop.bbox_area < min_size:
            im5[prop.coords] = 0 # remove cc with smaller size
    im6 = im5
    seed = np.copy(im6)
    seed[1:-1, 1:-1] = np.max(im6)
    im7 = reconstruction(seed, im6, method='erosion') # image erosion -> the best mask
    im8 = skeletonize(im7) # find skelenton
    im8 = thin(im8) # shrink skelenton

    im9 = np.zeros_like(im8).astype('float')
    cc = regionprops(im8.astype(np.uint8))
    stespath = []
    location = []
    polymasks = np.zeros_like(im8).astype('int')
    for prop in cc:
        tim = np.zeros_like(im8).astype(np.uint8)
        tim[prop.bbox[0]:prop.bbox[2], prop.bbox[1]:prop.bbox[3]] = prop.image
        E = skeleton_endpoints(tim)
        maxval = -1
        for i in range(min(5, len(E))):
            for j in range(min(5, len(E))):
                tim2, path = bfs(im8.astype(np.int64), list(E)[i], list(E)[-j])
                if np.max(tim2) > maxval:
                    maxval = np.max(tim2)
                    stespath = path
        tim1 = np.zeros_like(im8).astype('float')
        for i in range(len(stespath)):
            tim1[stespath[i]] = 1
        im9 = im9 + tim1
    # im9 -> spine image

    ## second step: find the tile location:
    mask = im7.astype('int')
    cc = regionprops(im9.astype(np.uint8))
    rad = patch_size / 2.0
    rad3 = patch_size * 2
    for prop in cc:
        tim = np.zeros_like(im9).astype('int')
        tim[prop.bbox[0]:prop.bbox[2], prop.bbox[1]:prop.bbox[3]] = prop.image
        E = skeleton_endpoints(tim)
        _, pts = bfs(tim, E[0], E[1]) # find the shortest path on the spine
        pvts = np.arange(0, len(pts) - patch_size, step_size) # find pivots on the spine
        for pvt in pvts:
            ## for each pvt 1: find perpendicular line and its width
            p1 = pts[int(pvt - tan_size)]
            p2 = pts[int(pvt + tan_size)]
            tan = (np.array(p2) - np.array(p1)) / np.sqrt(np.sum((np.array(p2) - np.array(p1)) ** 2))
            x2 = pts[int(pvt)][0] + [-rad3 * tan[1], rad3 * tan[1]];
            y2 = pts[int(pvt)][1] + [rad3 * tan[0], -rad3 * tan[0]];
            x2 = np.linspace(x2[0], x2[1])
            y2 = np.linspace(y2[0], y2[1])
            y2 = y2[np.multiply(x2 >= 0, x2 < tim.shape[0])]
            x2 = x2[np.multiply(x2 >= 0, x2 < tim.shape[0])]
            x2 = x2[np.multiply(y2 >= 0, y2 < tim.shape[1])]
            y2 = y2[np.multiply(y2 >= 0, y2 < tim.shape[1])] # x2, y2 represents the line perpendicular to the spline
            valid = mask[x2.round().astype('int').clip(0, tim.shape[0]-1), y2.round().astype('int').clip(0, tim.shape[1]-1)] == 1
            x2 = x2[valid]
            y2 = y2[valid]
            x2 = [x2[0], x2[-1]]
            y2 = [y2[0], y2[-1]] # x2, y2 is now the boundary points of the perpendicular line
            dist = np.sqrt((x2[1] - x2[0]) ** 2 + (y2[1] - y2[0]) ** 2)  # tissue width of current pvt point

            ## for each width assign bx n based on patch width
            n = max(1, int((dist - patch_size + step_size) / (step_size)))
            x3 = np.linspace(x2[0], x2[1], n)
            y3 = np.linspace(y2[0], y2[1], n)

            ## find the four corordinates of the patch and return the four cordinates
            for k in range(n):
                x = x3[k] + [rad * tan[1], -rad * tan[1], -rad * tan[1] + 2 * rad * tan[0],
                             rad * tan[1] + 2 * rad * tan[0]]
                y = y3[k] + [rad * tan[0], -rad * tan[0], -rad * tan[0] - 2 * rad * tan[1],
                             rad * tan[0] - 2 * rad * tan[1]]
                polymask = Image.new('L', (im8.shape[1], im8.shape[0]), 0)
                ImageDraw.Draw(polymask).polygon([y[0], x[0], y[1], x[1], y[2], x[2], y[3], x[3]], outline=1, fill=1)
                polymask = np.array(polymask)
                valid = np.multiply(mask, polymask)
                if np.sum(valid) > slide_thresh * np.sum(polymask): # more than thresh are tissues
                    valid = np.multiply(polymask, polymasks).astype('bool')
                    if np.sum(valid) < slide_thresh * np.sum(polymask): # overlap with current selection smaller than thresh
                        polymasks += polymask
                        location.append([[x[0], y[0]], [x[1], y[1]], [x[2], y[2]], [x[3], y[3]]])
    result['tile_location'] = location
    result['mask'] = mask
    result['spline'] = im9
    result['patch_mask'] = polymasks
    return result

if __name__ == "__main__":
    img = None
    kwargs = {'thresh': 5}
    output = spine(img, **kwargs)
    print(output)
