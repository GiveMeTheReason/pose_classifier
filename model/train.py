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

from config import CONFIG, TRAIN_CONFIG


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


def get_experiment_folder() -> str:
    experiment_id = str(get_last_experiment_id() + 1).zfill(3)
    experiment_folder = os.path.join(
        TRAIN_CONFIG.train_params.output_data,
        f'experiment_{experiment_id}',
    )
    if not os.path.exists(experiment_folder):
        os.makedirs(experiment_folder, exist_ok=True)
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
        # device = 'cuda:1'
    else:
        device = 'cpu'
    use_wandb = TRAIN_CONFIG.train_params.use_wandb

    with_rejection = TRAIN_CONFIG.gesture_set.with_rejection
    label_map = {gesture: i for i, gesture in enumerate(TRAIN_CONFIG.gesture_set.gestures, start=1)}
    if with_rejection:
        # label_map['_rejection'] = len(label_map)
        label_map['_rejection'] = 0

    batch_size = TRAIN_CONFIG.train_params.batch_size
    max_workers = TRAIN_CONFIG.train_params.max_workers

    epochs = TRAIN_CONFIG.train_params.epochs
    validate_each_epoch = TRAIN_CONFIG.train_params.validate_each_epoch

    lr = TRAIN_CONFIG.train_params.learning_rate
    weight_decay = TRAIN_CONFIG.train_params.weight_decay
    weight_loss = TRAIN_CONFIG.train_params.weight_loss

    data_list = []
    for gesture in TRAIN_CONFIG.gesture_set.gestures:
        data_list.extend(glob.glob(os.path.join(CONFIG.mediapipe.points_pose_world_windowed_filtered_labeled, f'G*_{gesture}_*_trial*.npy')))
    random.shuffle(data_list)

    train_len = int(TRAIN_CONFIG.train_params.train_share * len(data_list))
    train_list = data_list[:train_len]
    test_list = data_list[train_len:]

    train_transforms = transforms.TrainTransforms(device=device)
    test_transforms = transforms.TestTransforms(device=device)

    train_datasets = loaders.MediapipePoseLSTMDataset.split_datasets(
        batch_size=batch_size,
        max_workers=max_workers,
        samples=train_list,
        label_map=label_map,
        transforms=train_transforms,
    )
    train_loader = loaders.MultiStreamDataLoader(
        train_datasets, num_workers=0)

    test_datasets = loaders.MediapipePoseLSTMDataset.split_datasets(
        batch_size=1,
        max_workers=1,
        samples=test_list,
        label_map=label_map,
        transforms=test_transforms,
    )
    test_loader = loaders.MultiStreamDataLoader(
        test_datasets, num_workers=0)

    # train_datasets = loaders.MediapipeIterDataset(
    #     samples=train_list*batch_size,
    #     label_map=label_map,
    #     transforms=train_transforms,
    # )
    # test_datasets = loaders.MediapipeIterDataset(
    #     samples=test_list*batch_size,
    #     label_map=label_map,
    #     transforms=test_transforms,
    # )

    # train_loader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, num_workers=max_workers)
    # test_loader = DataLoader(test_datasets, batch_size=batch_size, shuffle=True, num_workers=max_workers)

    # model = classifiers.BaselineClassifier()
    model = classifiers.LSTMClassifier(len(label_map))
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
