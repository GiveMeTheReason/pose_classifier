import glob
import os
import random

import numpy as np
import torch
import torch.utils.data

from config import CONFIG

import loaders


def main():
    seed = 0
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'

    # label_map = {**GESTURES_MAP}

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    data_list = [
        d for d in glob.glob(os.path.join(CONFIG.path.mediapipe_points, '02.G*/select*/*.csv'))
    ]
    train_len = int(
        0.8 * len(data_list))
    test_len = len(data_list) - train_len
    train_list, test_list = map(
        list, torch.utils.data.random_split(data_list, [train_len, test_len]))

    train_datasets = loaders.MediapipePoseDataset.split_datasets(
        batch_size=128,
        max_workers=8,
        samples=train_list,
        label_map={'select': 1, 'no_gesture': 0},
        transforms=None,
    )
    train_loader = loaders.MultiStreamDataLoader(
        train_datasets, num_workers=0)

    test_datasets = loaders.MediapipePoseDataset.split_datasets(
        batch_size=1,
        max_workers=1,
        samples=test_list,
        label_map={'select': 1, 'no_gesture': 0},
        transforms=None,
    )
    test_loader = loaders.MultiStreamDataLoader(
        test_datasets, num_workers=1)

    for counter, (images, labels) in enumerate(train_loader):
        print(counter)


if __name__ == '__main__':
    main()
