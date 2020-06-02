import numpy as np
import sys
sys.path.append("..")


def convert_isup(pg, sg):
    if pg == 0 and sg == 0:
        return 0
    if pg == 3 and sg == 3:
        return 1
    if pg == 3 and sg == 4:
        return 2
    if pg == 4 and sg == 3:
        return 3
    if pg + sg == 8:
        return 4
    if (pg + sg == 9) or (pg + sg == 10):
        return 5


class ConvertRad:
    def __init__(self, logs, binary=False):
        self.binary = binary
        self.logs = logs

    # Simple maximum pooling with slide-level correction to get tile labels
    def convert(self, mask, slide_name, slide_pg, slide_sg):
        slide_patterns = set([slide_pg, slide_sg])
        unique, counts = np.unique(mask, return_counts=True)

        pattern_counts_dict = dict(zip(unique, counts))
        pattern_counts = [0] * 6
        for cur, count in pattern_counts_dict.items():
            pattern_counts[cur] = count
        counts_rank = np.argsort(pattern_counts)[::-1]
        tile_pg, tile_sg = None, None
        has_cancer = False
        for cur in counts_rank:
            if pattern_counts[cur] == 0:
                continue
            if cur > 2:
                if not has_cancer:
                    has_cancer = True
                if tile_pg is None:
                    tile_pg = cur
                elif tile_sg is None:
                    tile_sg = cur
        if has_cancer:
            if not tile_sg:
                tile_sg = tile_pg
        else:
            tile_pg, tile_sg = 0, 0
        if tile_pg not in slide_patterns or tile_sg not in slide_patterns:
            self.logs.append(f"Slide {slide_name}, Pixel pattern{tile_pg} or {tile_sg} not included in slide")
        isup_grade = convert_isup(tile_pg, tile_sg)
        return isup_grade

        # max_count, max_pattern = 0, 0
        # for pattern, cur_count in pattern_counts.items():
        #     # 0: background, 1: stroma, 2: benign epithelium
        #     if pattern <= 2:
        #         continue
        #     # Maybe wrong prediction. Since the pattern was not included in the slide prediction
        #     if pattern not in slide_patterns:
        #         self.logs.append(f"Slide {slide_name}, Pixel pattern{pattern} not included in slide")
        #         continue
        #     if cur_count > max_count:
        #         max_count = cur_count
        #         max_pattern = pattern
        # # Makes everything starts from 0: benign/stroma/background, 1: G3, 2: G4
        # if max_pattern > 0:
        #     max_pattern -= 2
        #
        # if self.binary:
        #     return int(max_pattern >= 1)
        # return max_pattern


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
