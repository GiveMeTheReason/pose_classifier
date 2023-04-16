import glob
import os
import random

import numpy as np
import torch

import model.classifiers as classifiers
import model.losses as losses
import model.train_loop as train_loop
import model.transforms as transforms
import loaders

from config import DATA_CONFIG, TRAIN_CONFIG


seed = TRAIN_CONFIG.train_params.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def get_last_experiment_id() -> int:
    existed_folders = glob.glob(
        os.path.join(TRAIN_CONFIG.train_params.output_data, 'experiment_*')
    )
    max_id = 0
    for folder in existed_folders:
        folder_id: str = folder.split('_')[-1]
        if not folder_id.isnumeric():
            continue
        max_id = max(max_id, int(folder_id))
    return max_id


def get_experiment_folder(continue_training: bool = False) -> str:
    if continue_training:
        experiment_id = str(TRAIN_CONFIG.train_params.experiment_id).zfill(3)
    else:
        experiment_id = str(get_last_experiment_id() + 1).zfill(3)
    experiment_folder = os.path.join(
        TRAIN_CONFIG.train_params.output_data,
        f'experiment_{experiment_id}',
    )
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder, exist_ok=True)
    return experiment_folder


def main():
    continue_training = TRAIN_CONFIG.train_params.continue_training
    experiment_folder = get_experiment_folder(continue_training)
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
        # device = 'cuda:1'
    else:
        device = 'cpu'

    label_map = TRAIN_CONFIG.gesture_set.label_map

    to_keep = TRAIN_CONFIG.transforms_params.to_keep
    shape_limit = TRAIN_CONFIG.transforms_params.shape_limit

    use_wandb = TRAIN_CONFIG.train_params.use_wandb

    batch_size = TRAIN_CONFIG.train_params.batch_size
    max_workers = TRAIN_CONFIG.train_params.max_workers

    epochs = TRAIN_CONFIG.train_params.epochs
    validate_each_epoch = TRAIN_CONFIG.train_params.validate_each_epoch

    lr = TRAIN_CONFIG.train_params.learning_rate
    weight_decay = TRAIN_CONFIG.train_params.weight_decay
    weight_loss = TRAIN_CONFIG.train_params.weight_loss

    data_list = []
    for gesture in TRAIN_CONFIG.gesture_set.gestures:
        data_list.extend(glob.glob(os.path.join(
            DATA_CONFIG.mediapipe.points_pose_world_windowed_filtered_labeled,
            f'G*_{gesture}_*_trial*.npy',
        )))
    random.shuffle(data_list)

    train_len = int(TRAIN_CONFIG.train_params.train_share * len(data_list))
    train_list = data_list[:train_len]
    test_list = data_list[train_len:]

    train_transforms = transforms.TrainTransforms(
        to_keep=to_keep,
        shape_limit=shape_limit,
        device=device,
    )
    test_transforms = transforms.TestTransforms(
        to_keep=to_keep,
        shape_limit=shape_limit,
        device=device,
    )
    labels_transforms = transforms.LabelsTransforms(
        shape_limit=shape_limit,
        device=device,
    )

    # train_datasets = loaders.MediapipePoseIterableDataset.split_datasets(
    #     samples=train_list,
    #     label_map=label_map,
    #     transforms=train_transforms,
    #     labels_transforms=labels_transforms,
    #     batch_size=batch_size,
    #     max_workers=max_workers,
    # )
    # train_loader = loaders.MultiStreamDataLoader(
    #     train_datasets, num_workers=0)

    # test_datasets = loaders.MediapipePoseIterableDataset.split_datasets(
    #     samples=test_list,
    #     label_map=label_map,
    #     transforms=test_transforms,
    #     labels_transforms=labels_transforms,
    #     batch_size=1,
    #     max_workers=1,
    # )
    # test_loader = loaders.MultiStreamDataLoader(
    #     test_datasets, num_workers=0)

    train_datasets = loaders.MediapipePoseMappedDataset(
        samples=train_list,
        label_map=label_map,
        transforms=train_transforms,
        labels_transforms=labels_transforms,
    )
    test_datasets = loaders.MediapipePoseMappedDataset(
        samples=test_list,
        label_map=label_map,
        transforms=test_transforms,
        labels_transforms=labels_transforms,
    )

    train_loader = loaders.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=max_workers)
    test_loader = loaders.DataLoader(test_datasets, batch_size=batch_size, shuffle=True, num_workers=max_workers)

    model = classifiers.LSTMClassifier(len(label_map))
    model.to(device)

    if continue_training and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_func = losses.CrossEntropyLoss(weight=torch.tensor(weight_loss), device=device)

    train_loop.train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
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
