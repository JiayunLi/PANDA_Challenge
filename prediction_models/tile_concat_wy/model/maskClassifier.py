import numpy as np

def extract_features(mask):
    counts = []
    for i in range(1,6):
        counts.append(np.count_nonzero(mask == i))
    percents = np.array(counts).astype(np.float32)
    percents /= percents.sum()
    return counts, percents