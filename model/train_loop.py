import datetime
import typing as tp
import tqdm

import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassConfusionMatrix

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
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    use_wandb: bool = False,
) -> None:
    def log_msg(
        msg: str,
        to_terminal: bool = False,
        to_log_file: bool = False,
    ):
        if to_log_file:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(msg)
        if to_terminal:
            print(msg)

    msg = f'Usung device: {device}'
    log_msg(msg, to_terminal=True, to_log_file=False)

    if use_wandb:
        wandb.init(project='pose_classifier', dir='./.wandb', name=checkpoint_path)
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

    confusion_matrix_metric = MulticlassConfusionMatrix(num_classes=len(label_map)).to(device)

    for epoch in range(1, epochs+1):

        model.train()
        train_accuracy = torch.zeros(1, device=device)
        train_loss = torch.zeros(1, device=device)
        train_n = torch.zeros(1, device=device)

        confusion_matrix_train = torch.zeros(
            (len(label_map), len(label_map)),
            dtype=torch.int,
            device=device,
        )

        for counter, (samples, labels) in tqdm.tqdm(enumerate(train_loader), desc=f'TRAIN | Epoch {epoch}', total=len(train_list)):

            samples: torch.Tensor
            labels: torch.Tensor

            samples = samples.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            prediction, *_ = model(samples)
            batch_loss = loss_func(prediction.permute(0, 2, 1), labels)
            batch_loss.backward()
            optimizer.step()

            prediction_probs, prediction_labels = prediction.max(dim=-1)
            train_accuracy += (prediction_labels == labels).sum()
            train_loss += batch_loss
            train_n += labels.shape[0] * labels.shape[1]

            confusion_matrix_train += confusion_matrix_metric(prediction_labels, labels)

            # break

        train_accuracy /= train_n
        train_loss /= len(train_list)

        msg = (
            f'{now()} [Epoch: {epoch:02}] Train acc: {train_accuracy.item():.4f} | loss: {train_loss.item():.4f}\n'
            f'{confusion_matrix_train=}\n'
        )
        log_msg(msg, to_terminal=True, to_log_file=True)

        torch.save(model.state_dict(), checkpoint_path)

        if use_wandb:
            wandb.log({
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                **{
                    f'train_acc_{label_str}':
                    confusion_matrix_train[label_int, label_int] / confusion_matrix_train[:, label_int].sum()
                    for label_str, label_int in label_map.items()
                }
            })

        if not (epoch % validate_each_epoch == 0):
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
            for counter, (val_samples, val_labels) in tqdm.tqdm(enumerate(test_loader), desc=f'VAL | Epoch {epoch}', total=len(test_list)):

                val_samples: torch.Tensor
                val_labels: torch.Tensor

                val_samples = val_samples.to(device)
                val_labels = val_labels.to(device)

                val_prediction, *_ = model(val_samples)

                val_prediction_probs, val_prediction_labels = val_prediction.max(dim=-1)
                val_accuracy += (val_prediction_labels == val_labels).sum().float()
                val_loss += loss_func(val_prediction.permute(0, 2, 1), val_labels)
                val_n += val_labels.shape[0] * val_labels.shape[1]

                confusion_matrix_val += confusion_matrix_metric(val_prediction_labels, val_labels)

                # break

        val_accuracy /= val_n
        val_loss /= len(test_list)

        msg = (
            f'{now()} [Epoch: {epoch:02}] Valid acc: {val_accuracy.item():.4f} | loss: {val_loss.item():.4f}\n'
            f'{confusion_matrix_val=}\n'
        )
        log_msg(msg, to_terminal=True, to_log_file=True)

        if use_wandb:
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
    if use_wandb:
        wandb.finish()
