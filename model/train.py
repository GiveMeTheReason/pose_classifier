import glob
import os
import random

import numpy as np
import torch
import torch.utils.data

# import model.loader as loader
# import model.losses as losses
# import model.model_cnn as model_cnn
# import model.train_loop as train_loop
# import model.transforms as transforms
# import utils.utils as utils
# import utils.utils_o3d as utils_o3d
from config import CONFIG


def main():
    seed = 0
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    # label_map = {**GESTURES_MAP}

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_list = [
        d for d in glob.glob(os.path.join(PC_DATA_DIR, 'G*/*/*/*'))
        if d.split(os.path.sep)[-3] in GESTURES_SET
    ]
    train_len = int(
        CONFIG['train']['train_ratio'] * len(data_list))
    test_len = len(data_list) - train_len
    train_list, test_list = map(
        list, torch.utils.data.random_split(data_list, [train_len, test_len]))

    train_datasets = loader.split_datasets(
        loader.HandGesturesDataset,
        batch_size=batch_size,
        max_workers=max_workers,
        path_list=train_list,
        label_map=label_map,
        transforms=pc_to_rgb,
        base_fps=base_fps,
        target_fps=target_fps,
        data_type=loader.AllowedDatasets.PCD,
        with_rejection=with_rejection,
    )
    train_loader = loader.MultiStreamDataLoader(
        train_datasets, image_size=resized_image_size, num_workers=0)

    test_datasets = loader.split_datasets(
        loader.HandGesturesDataset,
        batch_size=batch_size,
        max_workers=max_workers,
        path_list=test_list,
        label_map=label_map,
        transforms=rgb_depth_to_rgb,
        base_fps=base_fps,
        target_fps=target_fps,
        data_type=loader.AllowedDatasets.PROXY,
        with_rejection=with_rejection,
    )
    test_loader = loader.MultiStreamDataLoader(
        test_datasets, image_size=resized_image_size, num_workers=1)

    for counter, (images, labels) in enumerate(train_loader):
        print()


if __name__ == '__main__':
    main()
