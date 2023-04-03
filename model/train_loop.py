import datetime
import typing as tp

import torch
from torch.utils.data import DataLoader

import wandb


def now() -> str:
    return datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S')


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    train_list: tp.List[str],
    test_list: tp.List[str],
    label_map: tp.Dict[str, int],
    optimizer: torch.optim.Optimizer,
    loss_func: torch.nn.Module,
    epochs: int = 1,
    validate_each_epoch: int = 1,
    checkpoint_path: str = 'checkpoint.pth',
    log_path: str = 'log_file.txt',
    device: str = 'cpu' if not torch.cuda.is_available() else 'cuda',
):
    def log_msg(
        msg: str,
        to_terminal: bool = False,
        to_log_file: bool = False,
    ):
        if to_log_file:
            with open(log_filename, 'a', encoding='utf-8') as log_file:
                log_file.write(msg)
        if to_terminal:
            print(msg)

    wandb.init(project='pose_classifier')
    wandb.config = {
        'learning_rate': 1e-4,
        'epochs': epochs,
        'batch_size': 2
    }
    wandb.watch(model)

    msg = (
        f'\n{now()} | Training for {epochs} epochs started!\n'
        f'Train set length: {len(train_list)}\n'
        f'Test set length: {len(test_list)}\n'
    )
    log_msg(msg, to_terminal=True, to_log_file=True)

    for epoch in range(epochs):

        model.train()
        train_accuracy = torch.zeros(1, device=device)
        train_loss = torch.zeros(1, device=device)
        train_n = torch.zeros(1, device=device)

        confusion_matrix_train = torch.zeros(
            (len(label_map), len(label_map)),
            dtype=torch.int,
            device=device,
        )

        for counter, (samples, labels) in enumerate(train_loader):

            samples: torch.Tensor
            labels: torch.Tensor

            samples = samples.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            prediction: torch.Tensor = model(samples)
            batch_loss: torch.Tensor = loss_func(prediction, labels)
            batch_loss.backward()
            optimizer.step()

            # msg = (
            #     f'{now()} TRAIN\n'
            #     f'{epoch=}, {counter=}/{len(train_list)*120*target_fps//base_fps}\n'
            #     f'{prediction=}\n'
            #     f'{labels=}'
            # )
            # log_msg(msg, to_terminal=True, to_log_file=False)

            prediction_probs, prediction_labels = prediction.max(1)
            train_accuracy += (prediction_labels == labels).sum().float()
            train_loss += batch_loss
            train_n += len(labels)

            pred = prediction.argmax(dim=1)
            for i in range(len(labels)):
                confusion_matrix_train[pred[i], labels[i]] += 1

            # break

        train_accuracy /= train_n
        train_loss /= len(train_list)

        msg = (
            f'{now()} [Epoch: {epoch+1:02}] Train acc: {train_accuracy:.4f} | loss: {train_loss:.4f}\n'
            f'{confusion_matrix_train=}\n'
        )
        log_msg(msg, to_terminal=True, to_log_file=True)

        torch.save(model.state_dict(), checkpoint_path)

        wandb.log({
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            **{
                f'train_acc_{label_str}':
                confusion_matrix_train[label_int, label_int] / confusion_matrix_train[:, label_int].sum()
                for label_str, label_int in label_map.items()
            }
        })

        if not ((epoch+1) % validate_each_epoch == 0):
            continue

        model.eval()
        val_accuracy = torch.zeros(1, device=device)
        val_loss = torch.zeros(1, device=device)
        val_n = torch.zeros(1, device=device)

        confusion_matrix_val = torch.zeros(
            (len(label_map), len(label_map)),
            dtype=torch.int,
            device=device,
        )

        with torch.no_grad():
            for counter, (val_samples, val_labels) in enumerate(test_loader):

                val_samples: torch.Tensor
                val_labels: torch.Tensor

                val_samples = val_samples.to(device)
                val_labels = val_labels.to(device)

                prediction = model(val_samples)
                prediction_probs, prediction_labels = prediction.max(1)

                # msg = (
                #     f'{now()} VAL\n'
                #     f'{epoch=}, {counter=}/{len(test_list)*120*target_fps//base_fps}\n'
                #     f'{prediction=}\n'
                #     f'{val_labels=}'
                # )
                # log_msg(msg, to_terminal=True, to_log_file=False)

                val_accuracy += (prediction_labels == val_labels).sum().float()
                val_loss += loss_func(prediction, val_labels).item()
                val_n += len(val_labels)

                pred = prediction.argmax(dim=1)
                for i in range(len(val_labels)):
                    confusion_matrix_val[pred[i], val_labels[i]] += 1

                # break

        val_accuracy /= val_n
        val_loss /= len(test_list)

        msg = (
            f'{now()} [Epoch: {epoch+1:02}] Valid acc: {val_accuracy:.4f} | loss: {val_loss:.4f}\n'
            f'{confusion_matrix_val=}\n'
        )
        log_msg(msg, to_terminal=True, to_log_file=True)

        wandb.log({
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            **{
                f'val_acc_{label_str}':
                confusion_matrix_val[label_int, label_int] / confusion_matrix_val[:, label_int].sum()
                for label_str, label_int in label_map.items()
            }
        })

    msg = 'Training finished!\n\n'
    log_msg(msg, to_terminal=True, to_log_file=True)
