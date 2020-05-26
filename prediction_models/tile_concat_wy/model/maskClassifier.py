import numpy as np

def extract_features(mask, clsrange=[1, 6]):
    counts = []
    for i in range(clsrange[0], clsrange[1]):
        counts.append(np.count_nonzero(mask == i))
    percents = np.array(counts).astype(np.float32)
    percents /= percents.sum()
    return counts, percents