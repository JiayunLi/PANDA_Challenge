"""
DataLoader to extract tiles from a slide
"""
import pandas as pd


class BiopsySlides(data.Dataset):
    def __init__(self, params, test_file, test_slides_dir, transform):
        self.params = params
        self.test_df = pd.read_csv(test_file)
        self.test_slides_dir = test_slides_dir
        self.transform = transform

