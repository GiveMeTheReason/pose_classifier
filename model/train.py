import glob
import os
import random

import numpy as np
import torch
import torch.utils.data

import model.classifiers as classifiers
import model.losses as losses
import model.train_loop as train_loop
import loaders

from config import CONFIG


seed = CONFIG.train_params.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def get_last_experiment_id() -> int:
    existed_folders = glob.glob(
        os.path.join(CONFIG.train_params.output_data, 'experiment_*')
    )
    max_id = 0
    for folder in existed_folders:
        folder_id: str = folder.split('_')[-1]
        if not folder_id.isnumeric():
            continue
        max_id = max(max_id, int(folder_id))
    return max_id


def get_experiment_folder() -> str:
    experiment_id = str(get_last_experiment_id() + 1).zfill(3)
    experiment_folder = os.path.join(
        CONFIG.train_params.output_data,
        f'experiment_{experiment_id}',
    )
    if not os.path.exists(experiment_folder):
        os.mkdir(experiment_folder)
    return experiment_folder


def main():
    experiment_folder = get_experiment_folder()
    log_path = os.path.join(
        experiment_folder,
        'log_file.txt',
    )
    checkpoint_path = os.path.join(
        experiment_folder,
        'checkpoint.pth',
    )

    cuda_devices = torch.cuda.device_count()
    if cuda_devices:
        device = f'cuda:{cuda_devices - 1}'
    else:
        device = 'cpu'
    use_wandb = CONFIG.train_params.use_wandb

    with_rejection = CONFIG.gesture_set.with_rejection
    label_map = {gesture: i for i, gesture in enumerate(CONFIG.gesture_set.gestures)}
    if with_rejection:
        label_map['_rejection'] = len(label_map)

    batch_size = CONFIG.train_params.batch_size
    max_workers = CONFIG.train_params.max_workers

    epochs = CONFIG.train_params.epochs
    validate_each_epoch = CONFIG.train_params.validate_each_epoch

    lr = CONFIG.train_params.learning_rate
    weight_decay = CONFIG.train_params.weight_decay
    weight_loss = CONFIG.train_params.weight_loss

    data_list = sorted(glob.glob(os.path.join(CONFIG.mediapipe.points_pose_world, '*.npy')))
    random.shuffle(data_list)

    train_len = int(CONFIG.train_params.train_share * len(data_list))
    train_list = data_list[:train_len]
    test_list = data_list[train_len:]

    train_datasets = loaders.MediapipePoseDataset.split_datasets(
        batch_size=batch_size,
        max_workers=max_workers,
        samples=train_list,
        label_map=label_map,
        transforms=None,
    )
    train_loader = loaders.MultiStreamDataLoader(
        train_datasets, num_workers=0)

    test_datasets = loaders.MediapipePoseDataset.split_datasets(
        batch_size=1,
        max_workers=1,
        samples=test_list,
        label_map=label_map,
        transforms=None,
    )
    test_loader = loaders.MultiStreamDataLoader(
        test_datasets, num_workers=1)

    model = classifiers.BaselineClassifier()
    model.to(device)

    # if os.path.exists(checkpoint_path):
    #     model.load_state_dict(torch.load(checkpoint_path))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = losses.CrossEntropyLoss(weight=torch.tensor(weight_loss), device=device)

    train_loop.train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        train_list=train_list,
        test_list=test_list,
        label_map=label_map,
        optimizer=optimizer,
        loss_func=loss_func,
        epochs=epochs,
        validate_each_epoch=validate_each_epoch,
        checkpoint_path=checkpoint_path,
        log_path=log_path,
        device=device,
        use_wandb=use_wandb,
    )


if __name__ == '__main__':
    main()
