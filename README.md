# PANDA_Challenge
A repository for collaborating on PANDA challenge.
### docker: Contains Dockerfile for code development.
* Solved tqdm and jupyter notebook dependency issue by（already added to dockerfile）
```bash
conda install -c conda-forge ipywidgets
pip install --upgrade jupyter_client
```

* Solved skimage open tiff image issue by installing latest tiff package (already added to dockerfile)
```bash
pip install tifffile
```
### data：holds data files

### preprocessing:
#### Usage
Generate a set of tiles for a given slide.  
Tile generation is computational expensive, so use pre-generated LMDB file for training. 

```python
from preprocessing.tile_generation import generate_grid
from preprocessing.normalization import reinhard_bg

test_slides_dir = "/PANDA_Challenge/train_images/"
image_id = "example"
im_size = 512 # Size of tile to be extracted at highest scanning magnification
overlap = 0.25 # Overlap percentage for tiles
ts_thres = 0.6 # Discard tiles with tissue region less that ts_thres
dw_rate = 4 # Tiles downsample rate

# Initialize a tile generator
tile_generator = generate_grid.TileGeneratorGrid(test_slides_dir,
                                                         f"{image_id}.tiff", masks_dir=None, verbose=False)
# Initialize a normalizer
tile_normalizer = reinhard_bg.ReinhardNormalizer()
# use the pre-computed LAB mean and std values
tile_normalizer.fit(None)
# Need to specify masks_dir for tile_generator if you want to generate labeled masks for each tile. 
# Orig_tiles: tiles without normalization; norm_tiles: normalized tiles. 
orig_tiles, norm_tiles, locations, tissue_masks, label_masks \
            = tile_generator.extract_all_tiles(im_size, overlap, ts_thres, dw_rate, tile_normalizer)
```  
An example dataloader can be found in `prediction_models/att_mil/datasets/test_slides.py`

