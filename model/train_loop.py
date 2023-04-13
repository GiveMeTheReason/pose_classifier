# pyright: reportUnboundVariable=false

import datetime
import typing as tp
import tqdm

import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassConfusionMatrix


def now() -> str:
    return datetime.datetime.now().strftime('%Y.%m.%d %H:%M:%S')


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
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

    if use_wandb:
        import wandb

    def log_msg(
        msg: str,
        to_terminal: bool = False,
        to_log_file: bool = False,
    ) -> None:
        if to_log_file:
            with open(log_path, 'a', encoding='utf-8') as log_file:
                log_file.write(msg)
        if to_terminal:
            print(msg)

    def run_model(mode: str) -> None:
        mode = mode.capitalize()
        if mode == 'Train':
            model.train()
            loader = train_loader
        elif mode == 'Val':
            model.eval()
            loader = test_loader
        else:
            raise Exception('Unknown mode')

        loss = torch.zeros(1, device=device)

        confusion_matrix = torch.zeros(
            (len(label_map), len(label_map)),
            dtype=torch.int,
            device=device,
        )

        for counter, (samples, labels) in enumerate(tqdm.tqdm(loader, desc=f'{mode} | Epoch {epoch}')):

            if mode == 'Train':
                optimizer.zero_grad(set_to_none=True)

                prediction, *_ = model(samples)
                batch_loss = loss_func(prediction.permute(0, 2, 1), labels)
                batch_loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    prediction, *_ = model(samples)
                    batch_loss = loss_func(prediction.permute(0, 2, 1), labels)

            prediction_probs, prediction_labels = prediction.max(dim=-1)
            loss += batch_loss

            confusion_matrix += confusion_matrix_metric(prediction_labels, labels)

            # break

        accuracy = confusion_matrix.trace() / confusion_matrix.sum()
        f1_score = 2 * confusion_matrix.trace() / (confusion_matrix.trace() + confusion_matrix.sum())
        loss /= len(loader)

        msg = (
            f'{now()} [Epoch: {epoch:02}]\n'
            f'{mode} Accuracy: {accuracy.item():.4f}\n'
            f'{mode} F1-score: {f1_score.item():.4f}\n'
            f'{mode} Loss: {loss.item():.4f}\n'
            f'{confusion_matrix=}\n'
        )
        log_msg(msg, to_terminal=True, to_log_file=True)

        if mode == 'Train':
            torch.save(model.state_dict(), checkpoint_path)

        if use_wandb:
            wandb.log({
                f'{mode} loss': loss,
                f'{mode} accuracy': accuracy,
                f'{mode} F1-score': f1_score,
                **{
                    f'{mode} accuracy {label_str}':
                    confusion_matrix[label_int, label_int] / confusion_matrix[label_int, :].sum()
                    for label_str, label_int in label_map.items()
                },
                **{
                    f'{mode} F1-score {label_str}':
                    2 * confusion_matrix[label_int, label_int] / (2 * confusion_matrix[label_int, label_int] + confusion_matrix.sum() - confusion_matrix.trace())
                    for label_str, label_int in label_map.items()
                },
            })

    msg = f'Using device: {device}'
    log_msg(msg, to_terminal=True, to_log_file=False)

    if use_wandb:
        wandb.init(project='pose_classifier', dir='./.wandb', name=checkpoint_path)
        wandb.watch(model)

    train_len = len(train_loader.dataset)
    test_len = len(test_loader.dataset)

    msg = (
        f'{now()} | Training for {epochs} epochs started!\n'
        f'Train set length: {train_len}\n'
        f'Test set length: {test_len}\n'
    )
    log_msg(msg, to_terminal=True, to_log_file=True)

    confusion_matrix_metric = MulticlassConfusionMatrix(num_classes=len(label_map)).to(device)

    for epoch in range(1, epochs+1):

        run_model('Train')

        if not (epoch % validate_each_epoch == 0):
            continue

        run_model('Val')

    msg = 'Training finished!\n\n'
    log_msg(msg, to_terminal=True, to_log_file=True)
    if use_wandb:
        wandb.finish()
