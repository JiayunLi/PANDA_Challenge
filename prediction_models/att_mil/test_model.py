import torchvision.transforms as T
from PIL import Image
from prediction_models.att_mil.datasets import test_slides
from preprocessing.normalization import reinhard_bg
import torch
import pandas as pd


class TestParams:
    def __init__(self, test_slides_dir, im_size, input_size, dw_rate, ts_thres, overlap, num_channels=3):
        self.test_slides_dir = test_slides_dir
        self.im_size = im_size
        self.input_size = input_size
        self.dw_rate = dw_rate
        self.overlap = overlap
        self.ts_thres = ts_thres
        self.num_channels = num_channels


def test(model, meanstd, test_slides_df, test_params, num_workers, cuda):
    normalize = T.Compose([
        T.Resize(test_params.input_size, interpolation=Image.ANTIALIAS),
        T.ToTensor(),
        T.Normalize(mean=meanstd['mean'], std=meanstd['std'])])
    device = "cpu" if not cuda else "gpu"
    tile_normalizer = reinhard_bg.ReinhardNormalizer()
    # use the pre-computed LAB mean and std values
    tile_normalizer.fit(None)
    dataset = test_slides.BiopsySlides(test_params, test_slides_df, normalize, tile_normalizer)
    loader = \
        torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, drop_last=False,
                                    num_workers=num_workers, pin_memory=True)
    pred_data = []
    test_iter = iter(loader)
    with torch.no_grad():
        for step in range(len(loader)):
            tiles, image_id = test_iter.next()
            tiles = torch.squeeze(tiles, dim=0)
            image_id = str(image_id[0])
            tiles = tiles.to(device)
            slide_probs, _, _ = model(tiles)
            print(slide_probs.data)
            _, predicted = torch.max(slide_probs.data, 1)
            predicted = int(predicted.item())
            pred_data.append({"image_id": image_id, "isup_grade": predicted})

    pred_df = pd.DataFrame(columns=["image_id", "isup_grade"], data=pred_data)
    pred_df.to_csv("submission.csv", index=False)

