import cv2
import numpy as np
import collections
from collections import OrderedDict
from skimage.measure import label
from skimage.morphology import reconstruction, thin, skeletonize
from PIL import Image, ImageDraw
from scipy import ndimage

def skeleton_endpoints(skel):
    h,w = skel.shape
    # Find row and column locations that are non-zero
    (rows, cols) = np.nonzero(skel)
    # Initialize empty list of co-ordinates
    skel_coords = []
    # For each non-zero pixel...
    for (r, c) in zip(rows, cols):
        # Extract an 8-connected neighbourhood
        (col_neigh, row_neigh) = np.meshgrid(np.array([c - 1, c, c + 1]), np.array([r - 1, r, r + 1]))
        # Cast to int to index into image
        col_neigh = col_neigh.astype('int')
        row_neigh = row_neigh.astype('int')
        # Convert into a single 1D array and check for non-zero locations
        pix_neighbourhood = skel[row_neigh.clip(0,h-1), col_neigh.clip(0,w-1)].ravel() != 0
        # If the number of non-zero locations equals 2, add this to
        # our list of co-ordinates
        if np.sum(pix_neighbourhood) == 2:
            skel_coords.append((r, c))
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
            if (x, y) == goal:
                return steps, path
            for x2, y2 in (
            (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1), (x + 1, y + 1), (x - 1, y - 1), (x + 1, y - 1),
            (x - 1, y + 1)):
                if 0 <= x2 < width and 0 <= y2 < height and grid[x2][y2] == 1 and (x2, y2) not in seen:
                    queue.append(path + [(x2, y2)])
                    seen.add((x2, y2))
                    steps[x2, y2] = steps[x, y] + 1
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
    overlap_thresh = kwargs['overlap_thresh'] if 'overlap_thresh' in kwargs else 0.2
    h_step_size = kwargs['h_step_size'] if 'h_step_size' in kwargs else 0.5

    ## first step: find the spine
    im2 = img
    im3 = np.mean(im2, 2)  # returns the mean over channel dimension
    im4 = im3 < thresh * np.max(im3)  # True: content, False white pixel
    im4[im2[:, :, 1] > im3] = 0  # areas that are too greenish
    im4[im2[:, :, 2] > im2[:, :, 0] + bthresh] = 0  # areas that are too blueish
    im4 = im4.astype('int')

    im5 = im4

    seed = np.copy(im5)
    seed[1:-1, 1:-1] = np.max(im5)
    im6 = reconstruction(seed, im5, method='erosion')  # image erosion -> the best mask
    mask = im6.astype('int')

    im7 = skeletonize(im6)  # find skelenton
    im7 = thin(im7)  # shrink skelenton

    spine = np.zeros_like(im7).astype('float')
    cc, num_blob = label(mask, connectivity=2, return_num=True)
    stespath = []
    location = []
    IOU = []
    polymasks = np.zeros_like(im7).astype('int')
    rad = patch_size / 2.0
    rad3 = patch_size * 2
    for prop in range(1, num_blob + 1):
        #         print("{}/{}".format(prop,num_blob))
        tim = np.zeros_like(im7).astype(np.uint8)
        tim[cc == prop] = 1
        if np.sum(tim) < min_size:
            continue
        skel = skeletonize(tim)
        skel = thin(skel)
        E = skeleton_endpoints(skel)
        maxval = -1
        for i in range(min(5, len(E))):
            for j in range(min(5, len(E))):
                tim2, path = bfs(skel.astype(np.int64), list(E)[i], list(E)[-j])
                if np.max(tim2) > maxval:
                    maxval = np.max(tim2)
                    stespath = path
        tim1 = np.zeros_like(im7).astype('float')
        for i in range(len(stespath)):
            tim1[stespath[i]] = 1
        spine = spine + tim1  # spine -> spine image

        ## second step: find the tile location:
        E = skeleton_endpoints(tim1)
        if len(E) < 2:
            continue
        _, pts = bfs(tim, E[0], E[1])  # find the shortest path on the spine
        pvt = rad
        if not pts:
            continue
        while pvt < len(pts) - tan_size:
            #             print(pvt)
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
            y2 = y2[np.multiply(y2 >= 0, y2 < tim.shape[1])]  # x2, y2 represents the line perpendicular to the spline
            valid = mask[x2.round().astype('int').clip(0, tim.shape[0] - 1), y2.round().astype('int').clip(0, tim.shape[
                1] - 1)] == 1
            x2 = x2[valid]
            y2 = y2[valid]
            if np.sum(valid) < 2:
                pvt = int(pvt + step_size)
                continue
            x2 = [x2[0], x2[-1]]
            y2 = [y2[0], y2[-1]]  # x2, y2 is now the boundary points of the perpendicular line
            dist = np.sqrt((x2[1] - x2[0]) ** 2 + (y2[1] - y2[0]) ** 2)  # tissue width of current pvt point

            ## for each width assign bx n based on patch width
            n = max(1, int(dist / (h_step_size * patch_size)))
            #             x3 = np.linspace(x2[0] + rad * tan[1], x2[1]-rad * tan[1], n)
            #             y3 = np.linspace(y2[0] + rad * tan[0], y2[1]-rad * tan[0], n)
            x3 = np.linspace(x2[0], x2[1], n)
            y3 = np.linspace(y2[0], y2[1], n)

            ## find the four corordinates of the patch and return the four cordinates
            mid = int(n / 2)
            for k in range(n):
                mod = int((k + 1) / 2)
                if k % 2 == 1:
                    pos = min(mid + mod, n - 1)
                else:
                    pos = max(mid - mod, 0)
                x = x3[pos] + [rad * tan[1], -rad * tan[1], -rad * tan[1] + 2 * rad * tan[0],
                               rad * tan[1] + 2 * rad * tan[0]]
                y = y3[pos] + [rad * tan[0], -rad * tan[0], -rad * tan[0] - 2 * rad * tan[1],
                               rad * tan[0] - 2 * rad * tan[1]]
                polymask = Image.new('L', (im7.shape[1], im7.shape[0]), 0)
                ImageDraw.Draw(polymask).polygon([y[0], x[0], y[1], x[1], y[2], x[2], y[3], x[3]], outline=1, fill=1)
                polymask = np.array(polymask)
                valid = np.multiply(mask, polymask)
                if np.sum(valid) > slide_thresh * np.sum(polymask):  # more than thresh are tissues
                    valid1 = np.multiply(polymask.astype('bool'), polymasks.astype('bool')).astype('bool')
                    if np.sum(valid1) < overlap_thresh * np.sum(
                            polymask):  # overlap with current selection smaller than thresh
                        # step_size = 0.8 * step_size
                        polymasks += polymask
                        location.append([[y[0], x[0]], [y[1], x[1]], [y[2], x[2]], [y[3], x[3]]])
                        IOU.append(np.sum(valid) / np.sum(polymask))
                    else:
                        # step_size = 1.1 * step_size
                        pass
            pvt = int(pvt + step_size)
    result['tile_location'] = location
    result['patch_mask'] = polymasks
    result['mask'] = mask
    result['spine'] = spine
    result['IOU'] = IOU
    return result

def remove_pen_marks(img, scale=1):
    # Define elliptic kernel
    kernel5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # use cv2.inRange to mask pen marks (hardcoded for now)
    lower = np.array([0, 0, 0])
    upper = np.array([200, 255, 255])
    img_mask1 = cv2.inRange(img, lower, upper)

    # Use erosion and findContours to remove masked tissue (side effect of above)
    img_mask1 = cv2.erode(img_mask1, kernel5x5, iterations=4)
    img_mask2 = np.zeros(img_mask1.shape, dtype=np.uint8)
    _, contours, _ = cv2.findContours(img_mask1, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        #         print(contour.shape)
        x, y = contour[:, 0, 0], contour[:, 0, 1]
        w, h = x.max() - x.min(), y.max() - y.min()
        if w > 100 / scale and h > 100 / scale:
            cv2.drawContours(img_mask2, [contour], 0, 1, -1)
    # expand the area of the pen marks
    img_mask2 = cv2.dilate(img_mask2, kernel5x5, iterations=3)
    img_mask2 = (1 - img_mask2)

    # Mask out pen marks from original image
    img = cv2.bitwise_and(img, img, mask=img_mask2)

    img[img == 0] = 255
    return img, img_mask1, img_mask2

def tile(img, mask, location, iou, sz=256, N=36, scale = 8, mode = "random"):
    result = []
    idxsort = np.argsort(iou)[::-1]
    for i in range(min(N, len(idxsort))):
        idx = idxsort[i]
        cnt = np.expand_dims(location[idx], 1) * scale
        cnt = cnt.astype('int')
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        width = int(rect[1][0])
        height = int(rect[1][1])
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height - 1],
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1]], dtype="float32")

        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        warped_img = cv2.warpPerspective(img, M, (width, height), borderValue = (255,255,255))
        warped_mask = cv2.warpPerspective(mask, M, (width, height), borderValue = (0,0,0))
        shape = warped_img.shape

        pad0, pad1 = 0,0
        if shape[0] < sz:
            pad0 = (sz - shape[0] % sz) % sz
        if shape[1] < sz:
            pad1 = (sz - shape[1] % sz) % sz
        if pad0 or pad1:
            warped_mask = np.pad(warped_mask, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], [0, 0]], mode='constant', constant_values=0)
            warped_img = np.pad(warped_img, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], [0, 0]], mode='constant', constant_values=255)
        shape = warped_img.shape
        top, left = int((shape[0] - sz) / 2), int((shape[1] - sz) / 2)
        result.append({'img': warped_img[top:top+sz,left:left+sz,:], 'mask': warped_mask[top:top+sz,left:left+sz,:], 'location': cnt})
    if len(idxsort) < N:
        if mode == "random":
            complete_idx = np.random.choice(len(idxsort), size=N - len(idxsort))
            for i in complete_idx:
                result.append(result[i])
        elif mode == "blank":
            for i in range(N - len(idxsort)):
                result.append({'img': 255 * np.ones((sz, sz, 3)).astype(np.uint8),
                               'mask': np.zeros((sz, sz, 3)).astype(np.uint8), 'location': None})
    return result

def tile_img(img, location, iou, sz=256, N=36, scale = 8, mode = "random"):
    result = []
    idxsort = np.argsort(iou)[::-1]
    for i in range(min(N, len(idxsort))):
        idx = idxsort[i]
        cnt = np.expand_dims(location[idx], 1) * scale
        cnt = cnt.astype('int')
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        width = int(rect[1][0])
        height = int(rect[1][1])
        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height - 1],
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1]], dtype="float32")

        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)

        # directly warp the rotated rectangle to get the straightened rectangle
        warped_img = cv2.warpPerspective(img, M, (width, height))
        shape = warped_img.shape

        if shape[0] < sz or shape[1] < sz:
            pad0, pad1 = (sz - shape[0] % sz) % sz, (sz - shape[1] % sz) % sz
            warped_img = np.pad(warped_img, [[pad0 // 2, pad0 - pad0 // 2], [pad1 // 2, pad1 - pad1 // 2], [0, 0]], mode='constant', constant_values=255)

        result.append({'img': warped_img[:sz,:sz,:], 'location': cnt})
    if len(idxsort) < N:
        if mode == "random":
            complete_idx = np.random.choice(len(idxsort), size=N - len(idxsort))
            for i in complete_idx:
                result.append(result[i])
        elif mode == "blank":
            for i in range(N - len(idxsort)):
                result.append({'img': 255 * np.ones((sz, sz, 3)).astype(np.uint8),
                               'location': None})
    return result

def tile_rect(img, mask, bn_mask, sz=256, N=36, scale=8, overlap_ratio=0.2, mode="random", **kwargs):
    slide_thresh = kwargs['slide_thresh'] if 'slide_thresh' in kwargs else 0.1
    bdx = ndimage.find_objects(bn_mask)
    mask_shape = bn_mask.shape
    x0, x1, y0, y1 = bdx[0][0].start, bdx[0][0].stop, bdx[0][1].start, bdx[0][1].stop
    eq_mask_size = int(sz / scale)
    x_n = (x1 - x0) / (overlap_ratio * eq_mask_size)
    y_n = (y1 - y0) / (overlap_ratio * eq_mask_size)
    x_grid = np.linspace(x0, x1 + sz/2, int(x_n) + 1).astype('int')
    y_grid = np.linspace(y0, y1 + sz/2, int(y_n) + 1).astype('int')
    grid_iou = []
    tiles_location = []
    tim = np.zeros_like(bn_mask)
    update_patch_mask = np.zeros_like(bn_mask)
    for x in x_grid:
        for y in y_grid:
            if x >= 0 and x < mask_shape[0] - eq_mask_size and y >= 0 and y < mask_shape[1] - eq_mask_size:
                tiles_location.append([x, y])
                iou = np.sum(bn_mask[x:x + eq_mask_size, y:y + eq_mask_size]) / (eq_mask_size * eq_mask_size)
                grid_iou.append(iou)
                tim[x:x + eq_mask_size, y:y + eq_mask_size] += bn_mask[x:x + eq_mask_size, y:y + eq_mask_size]
                update_patch_mask[x:x + eq_mask_size, y:y + eq_mask_size] += 1
    tim = (tim > 0).astype('int')
    ra = np.sum(np.multiply(tim, bn_mask)) / np.sum(bn_mask)

    idxsort = np.argsort(grid_iou)[::-1]
    result = []
    count = 0
    img_shape = img.shape
    for idx in idxsort:
        if grid_iou[idx] < slide_thresh:
            continue
        x, y = tiles_location[idx][0] * scale, tiles_location[idx][1] * scale
        if x >= 0 and x < img_shape[0] - sz and y >= 0 and y < img_shape[1] - sz:
            tile = img[x:x + sz, y:y + sz, :]
            tile_mask = mask[x:x + sz, y:y + sz]
            cnt = np.array([[y, x], [y, x + sz], [y + sz, x], [y + sz, x + sz]])
            cnt = np.expand_dims(cnt, 1)
            result.append({'img': tile, 'mask': tile_mask, 'location': cnt})
            count += 1
    if count < N:
        if mode == "random":
            complete_idx = np.random.choice(count, size=N - count)
            for i in complete_idx:
                result.append(result[i])
        elif mode == "blank":
            for i in range(N - len(idxsort)):
                result.append({'img': 255 * np.ones((sz, sz, 3)).astype(np.uint8),
                               'mask': np.zeros((sz, sz, 3)).astype(np.uint8), 'location': None})
    return result, ra, update_patch_mask

def tile_rect_img(img, bn_mask, sz=256, N=36, scale=8, overlap_ratio=0.2, mode="random",**kwargs):
    slide_thresh = kwargs['slide_thresh'] if 'slide_thresh' in kwargs else 0.1
    bdx = ndimage.find_objects(bn_mask)
    mask_shape = bn_mask.shape
    x0, x1, y0, y1 = bdx[0][0].start, bdx[0][0].stop, bdx[0][1].start, bdx[0][1].stop
    eq_mask_size = int(sz / scale)
    x_n = (x1 - x0) / (overlap_ratio * eq_mask_size)
    y_n = (y1 - y0) / (overlap_ratio * eq_mask_size)
    x_grid = np.linspace(x0, x1 + sz/2, int(x_n) + 1).astype('int')
    y_grid = np.linspace(y0, y1 + sz/2, int(y_n) + 1).astype('int')
    grid_iou = []
    tiles_location = []
    tim = np.zeros_like(bn_mask)
    for x in x_grid:
        for y in y_grid:
            if x >= 0 and x < mask_shape[0] - eq_mask_size and y >= 0 and y < mask_shape[1] - eq_mask_size:
                tiles_location.append([x, y])
                iou = np.sum(bn_mask[x:x + eq_mask_size, y:y + eq_mask_size]) / (eq_mask_size * eq_mask_size)
                grid_iou.append(iou)
                tim[x:x + eq_mask_size, y:y + eq_mask_size] += bn_mask[x:x + eq_mask_size, y:y + eq_mask_size]
    tim = (tim > 0).astype('int')
    ra = np.sum(np.multiply(tim, bn_mask)) / np.sum(bn_mask)

    idxsort = np.argsort(grid_iou)[::-1]
    result = []
    count = 0
    img_shape = img.shape
    for idx in idxsort:
        if grid_iou[idx] < slide_thresh:
            continue
        x, y = tiles_location[idx][0] * scale, tiles_location[idx][1] * scale
        if x >= 0 and x < img_shape[0] - sz and y >= 0 and y < img_shape[1] - sz:
            tile = img[x:x + sz, y:y + sz, :]
            cnt = np.array([[y, x], [y, x + sz], [y + sz, x], [y + sz, x + sz]])
            cnt = np.expand_dims(cnt, 1)
            result.append({'img': tile, 'location': cnt})
            count += 1
    if count < N:
        if mode == "random":
            complete_idx = np.random.choice(count, size=N -count)
            for i in complete_idx:
                result.append(result[i])
        elif mode == "blank":
            for i in range(N - len(idxsort)):
                result.append({'img': 255 * np.ones((sz, sz, 3)).astype(np.uint8),
                               'location': None})
    return result, ra

if __name__ == "__main__":
    img = None
    kwargs = {'thresh': 5}
    output = spine(img, **kwargs)
    print(output)
