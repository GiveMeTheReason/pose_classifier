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
label_map: tp.Dict = {gesture: i for i, gesture in enumerate(gestures, start=1)}
if with_rejection:
    # label_map['_rejection'] = len(label_map)
    label_map['_rejection'] = 0
inv_label_map = {value: key for key, value in label_map.items()}

class GestureSet(ConfigBaseClass):
    gestures: tp.Tuple[str, ...] = gestures
    with_rejection: bool = with_rejection
    label_map: tp.Dict[str, int] = label_map
    inv_label_map: tp.Dict[int, str] = inv_label_map

##################################################

seed = 0

experiment_id = 1
is_continue = False

output_data = os.path.join(
    'output_data',
)
use_wandb = False

train_share = 0.8

batch_size = 1
max_workers = 0

epochs = 100
validate_each_epoch = 1

learning_rate = 1e-4
weight_decay = 1e-5
weight_loss = [1.0] * len(gestures)
if with_rejection:
    weight_loss.append(1.0)

class TrainParams(ConfigBaseClass):
    seed: int = seed

    experiment_id: int = experiment_id
    is_continue: bool = is_continue

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
