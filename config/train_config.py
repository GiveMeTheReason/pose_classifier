import os
import typing as tp

from config.base_config import ConfigBaseClass

##################################################

gestures = (
    'select',
    'call',
    'start',
    'yes',
    'no',
)
with_rejection = True

class GestureSet(ConfigBaseClass):
    gestures: tp.Tuple[str, ...] = gestures
    with_rejection: bool = with_rejection

##################################################

seed = 0

output_data = os.path.join(
    'output_data',
)
use_wandb = False

train_share = 0.8

batch_size = 128 * 5
max_workers = 8

epochs = 10
validate_each_epoch = 1

learning_rate = 1e-4
weight_decay = 1e-5
weight_loss = [1.0] * len(gestures)
if with_rejection:
    weight_loss.append(1.0)

class TrainParams(ConfigBaseClass):
    seed: int = seed

    output_data: str = output_data
    use_wandb: bool = use_wandb

    train_share: float = train_share

    batch_size: int = batch_size
    max_workers: int = max_workers

    epochs: int = epochs
    validate_each_epoch: int = validate_each_epoch

    learning_rate: float = learning_rate
    weight_decay: float = weight_decay
    weight_loss: tp.List[float] = weight_loss

##################################################

class TRAIN_CONFIG(ConfigBaseClass):
    gesture_set: GestureSet = GestureSet()
    train_params: TrainParams = TrainParams()
