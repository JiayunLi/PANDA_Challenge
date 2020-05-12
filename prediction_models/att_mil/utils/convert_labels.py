import numpy as np
import sys
sys.path.append("..")


class ConvertRad:
    def __init__(self, logs, binary=False):
        self.binary = binary
        self.logs = logs

    # Simple maximum pooling with slide-level correction to get tile labels
    def convert(self, mask, slide_name, slide_pg, slide_sg):
        slide_patterns = set([slide_pg, slide_sg])
        unique, counts = np.unique(mask, return_counts=True)
        pattern_counts = dict(zip(unique, counts))
        max_count, max_pattern = 0, 0
        for pattern, cur_count in pattern_counts.items():
            # 0: background, 1: stroma, 2: benign epithelium
            if pattern <= 2:
                continue
            # Maybe wrong prediction. Since the pattern was not included in the slide prediction
            if pattern not in slide_patterns:
                self.logs.append(f"Slide {slide_name}, Pixel pattern{pattern} not included in slide")
                continue
            if cur_count > max_count:
                max_count = cur_count
                max_pattern = pattern
        # Makes everything starts from 0: benign/stroma/background, 1: G3, 2: G4
        if max_pattern > 0:
            max_pattern -= 2

        if self.binary:
            return int(max_pattern >= 1)
        return max_pattern


class ConvertKaro:
    def __init__(self, logs, binary=False):
        self.binary = binary
        self.logs = logs

    def convert(self, mask, slide_name, slide_pg, slide_sg):
        # Since there is no Gleason pattern labels for Karolinska data, we only train tiles from pure Gleason slides
        if slide_sg != slide_pg:
            return -1
        unique, counts = np.unique(mask, return_counts=True)
        pattern_counts = dict(zip(unique, counts))
        # If there is any cancerous regions
        if 2 in pattern_counts and pattern_counts[2] > 0:
            if self.binary:
                return 1
            else:
                # 1: G3, 2: G4
                return slide_pg - 2
        # No cancerous regions: return 0: benign/stroma/background
        else:
            return 0
