import numpy as np
from skimage import color
from skimage import morphology as skmp


def get_tissue_area(slide, w_br=False):
    level = slide.level_count
    level -= 1
    dw_samples_dim = slide.level_dimensions[level]
    # get sample from lowest available resolution
    dw_sample = slide.read_region((0, 0), level, (dw_samples_dim[0], dw_samples_dim[1]))
    # convert to RGB (originally should be RGBA)
    dw_sample = np.asarray(dw_sample.convert('RGB'))
    # convert to HSV color space: H: hue, S: saturation, V: value
    dw_sample_hsv = color.rgb2hsv(dw_sample)

    # Get first ROI to remove all kinds of markers (Blue, Green, black)
    roi1 = (dw_sample_hsv[:, :, 0] <= 0.67) | (
            (dw_sample_hsv[:, :, 1] <= 0.15) & (dw_sample_hsv[:, :, 2] <= 0.75))
    # exclude marker roi
    roi1 = ~roi1
    skmp.remove_small_holes(roi1, area_threshold=500, connectivity=20, in_place=True)
    skmp.remove_small_objects(roi1, min_size=300, connectivity=20, in_place=True)

    # remove background: regions with low value(black) or very low saturation (white)
    roi2 = (dw_sample_hsv[:, :, 1] >= 0.05) & (dw_sample_hsv[:, :, 2] >= 0.25)
    roi2 *= roi1

    skmp.remove_small_holes(roi2, area_threshold=500, connectivity=20, in_place=True)
    skmp.remove_small_objects(roi2, min_size=300, connectivity=20, in_place=True)

    if w_br:
        tempt1 = (100.0 * dw_sample[:, :, 2]) / (1.0 + dw_sample[:, :, 0] + dw_sample[:, :, 1])
        tempt2 = 256.0 / (1.0 + dw_sample[:, :, 0] + dw_sample[:, :, 1] + dw_sample[:, :, 2])
        br = tempt1 * tempt2
        return roi2, br

    return roi2
