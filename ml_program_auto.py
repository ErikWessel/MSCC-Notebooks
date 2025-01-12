# %%
import gc
import math
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import rasterio
import rasterio.plot
import seaborn as sns
import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import yaml
from ml_commons import *
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models import (EfficientNet_V2_S_Weights, Swin_V2_S_Weights,
                                efficientnet_v2_s, swin_v2_s)

cudnn.benchmark = True
sns.set_theme()

# %% [markdown]
# Load configuration file that specifies training routine

# %%
config = yaml.safe_load(open('ml_config.yml'))

# %%
prefix_dir = config['paths']['prefix_dir']
dataset_dir = os.path.join(prefix_dir, config['paths']['dataset_dir'])
output_dir = os.path.join(config['paths']['machine_learning_dir'], 'output')

# %% [markdown]
# Release memory to ensure enough capacity on the selected device

# %%
gc.collect()
with torch.no_grad():
    torch.cuda.empty_cache()

# %%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using {device} for inference')

# %% [markdown]
# Create model using transfer learning from either EfficientNetV2 or SwinTransformerV2

# %%
def print_model(model:nn.Module, name:str):
    spacing = '  '
    model_str = spacing + f'\n{spacing}'.join(str(model).splitlines())
    print(f'--- {name} ---\n{model_str}\n{"-" * (8 + len(name))}')

# %%
print(f'Using model: ' + config['model']['name'])
model_name = str(config['model']['name']).lower()
use_transfer_learning = config['model']['use_transfer_learning']
if model_name == 'swintransformer':
    model = swin_v2_s(weights = Swin_V2_S_Weights.DEFAULT if use_transfer_learning else None)
elif model_name == 'efficientnet':
    model = efficientnet_v2_s(weights = EfficientNet_V2_S_Weights.DEFAULT if use_transfer_learning else None)
else:
    raise RuntimeError(f'Model "' + config['model']['name'] + '" is unknown')

if config['model']['freeze_parameters']:
    for param in model.parameters(): #freeze model
        param.requires_grad = False

print_model(get_model_head(model), 'Initial Model Head')
num_features = get_num_features(model)
print(f'Classification layer has {num_features} input features')
new_model_head = nn.Sequential()
# Before linear layer
if config['processing']['use_dropout']:
    new_model_head.append(nn.Dropout(p=config['processing']['dropout_p'], inplace=True))
# Linear layer
linear_layer = nn.Linear(num_features, 1 if config['processing']['use_one_neuron_regression'] else len(classes))
new_model_head.append(linear_layer)
# After linear layer
if (config['processing']['use_ordinal_regression'] or config['processing']['use_one_neuron_regression']) \
        and config['processing']['activation_function'] != False:
    activation_function = str(config['processing']['activation_function']).lower()
    if activation_function == 'sigmoid':
        activation_function = nn.Sigmoid()
    elif activation_function == 'relu':
        activation_function = nn.ReLU()
    elif activation_function == 'tanh':
        activation_function = nn.Tanh()
    else:
        raise RuntimeError(f'Unkown activation function: {activation_function}')
    new_model_head = new_model_head.append(activation_function)

set_model_head(model, new_model_head)
print_model(get_model_head(model), 'Modified Model Head')
# model = nn.DataParallel(model)
model = model.to(device)

# %% [markdown]
# Load weights and define prerequisite functions for model training

# %%
training_weights = pd.read_csv(os.path.join(dataset_dir, 'training_weights.csv'), index_col='label')
training_weights.T

# %%
def prediction_to_label_ordinal_regression(pred: torch.Tensor) -> torch.Tensor:
    return (pred > 0.5).cumprod(axis=1).sum(axis=1) - 1

# %%
class OrdinalRegression:
    def __init__(self, weights:Optional[torch.Tensor]) -> None:
        if weights is None:
            self.weights = torch.Tensor([1] * len(classes)).to(device)
        else:
            self.weights = weights
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        modified_targets = torch.zeros_like(predictions)
        for i, target in enumerate(targets):
            modified_targets[i, 0 : target + 1] = 1
        loss_function_name = str(config['processing']['ordinal_regression_loss']).lower()
        if loss_function_name == 'mse':
            loss = torch.mean((nn.MSELoss(reduction='none')(predictions, modified_targets) * self.weights).sum(axis=1))
        elif loss_function_name == 'l1':
            loss = torch.mean((nn.L1Loss(reduction='none')(predictions, modified_targets) * self.weights).sum(axis=1))
        else:
            raise RuntimeError(f'Unknown loss function: {loss_function_name}')
        return loss

# %%
def prediction_to_label_one_neuron_regression(pred: torch.Tensor) -> torch.Tensor:
    return ((pred * max(classes)).round()).int().flatten()

# %%
class OneNeuronRegression:
    def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        modified_predictions = predictions.flatten()
        classes_max = max(classes)
        loss_function_name = str(config['processing']['ordinal_regression_loss']).lower()
        if loss_function_name == 'mse':
            loss = torch.mean(nn.MSELoss(reduction='none')(modified_predictions * classes_max, targets.float()))
        elif loss_function_name == 'l1':
            loss = torch.mean(nn.L1Loss(reduction='none')(modified_predictions * classes_max, targets.float()))
        else:
            raise RuntimeError(f'Unknown loss function: {loss_function_name}')
        return loss

# %% [markdown]
# Load datasets and apply techniques like data augmentation or use a weighted sampler

# %%
dataset_transforms = [
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
]
composed_transforms = transforms.Compose(dataset_transforms)

# %%
use_manual_labels = config['processing']['use_manual_labels']

# %%
train_dataset       = AimlsseImageDataset(DatasetType.TRAINING,     dataset_dir, transfrom=composed_transforms, use_manual_labels=use_manual_labels)
validation_dataset  = AimlsseImageDataset(DatasetType.VALIDATION,   dataset_dir, transfrom=None, use_manual_labels=use_manual_labels)

# %%
def mean_std(loader:DataLoader):
  sum, squared_sum, num_batches = 0,0,0
  for data, _, _ in loader:
    sum += torch.mean(data,dim=[0,1,2])
    squared_sum += torch.mean(data**2,dim=[0,1,2])
    num_batches += 1
  mean = sum/num_batches
  std = (squared_sum/num_batches - mean**2)**0.5
  return mean, std

# %%
def batch_normalization(dataset:AimlsseImageDataset, dataset_type:DatasetType, dataset_transforms):
    mean, std = mean_std(dataset)
    print(f'{dataset_type.name} - mean {mean:.3f}, std {std:.3f}')
    if dataset_transforms is None:
        dataset_transforms = []
    return AimlsseImageDataset(dataset_type, dataset_dir,
                               transfrom = transforms.Compose(dataset_transforms + [transforms.Normalize(mean, std)]),
                               use_manual_labels=use_manual_labels)

# %%
if config['processing']['batch_normalization']:
    train_dataset       = batch_normalization(train_dataset, DatasetType.TRAINING, dataset_transforms)
    validation_dataset  = batch_normalization(validation_dataset, DatasetType.VALIDATION, None)

# %%
if config['processing']['use_weighted_sampler']:
    num_samples = len(train_dataset)
    weights = [0] * num_samples
    for i in range(num_samples):
        label = train_dataset.get_label(i)
        weights[i] = training_weights.loc[class_names]['weight'].iloc[label]
    training_sampler = WeightedRandomSampler(weights, num_samples)
    train_dataloader =  DataLoader(train_dataset,       batch_size=config['processing']['batch_size'], sampler=training_sampler)
else:
    train_dataloader =  DataLoader(train_dataset,       batch_size=config['processing']['batch_size'], shuffle=True)
validation_dataloader = DataLoader(validation_dataset,  batch_size=config['processing']['batch_size'], shuffle=True)

# %% [markdown]
# If enabled in the config file, show example images of the dataset with corresponding labels

# %%
sample_batch_index = 0

# %%
if config['output']['show_samples']:
    plot_samples(train_dataset, config['processing']['batch_size'], sample_batch_index)
    sample_batch_index += 1

# %% [markdown]
# Define all prerequisites that are necessary for model training, depending on the settings in the configuration file

# %%
if config['processing']['use_weighted_loss_function']:
    loss_function_weights = torch.tensor(training_weights['weight'].to_list())
    loss_function_weights = loss_function_weights.to(device)
else:
    loss_function_weights = None
print(f'Loss weights: {loss_function_weights}')

if config['processing']['use_ordinal_regression']:
    criterion = OrdinalRegression(loss_function_weights)
    outputs_to_predictions = prediction_to_label_ordinal_regression
elif config['processing']['use_one_neuron_regression']:
    criterion = OneNeuronRegression()
    if loss_function_weights is not None:
        raise Warning('Unable to use one neuron regression with loss function weights')
    outputs_to_predictions = prediction_to_label_one_neuron_regression
else:
    criterion = nn.CrossEntropyLoss(loss_function_weights)
    outputs_to_predictions = lambda outputs: torch.max(outputs, 1)[1]

learning_rate = math.pow(10, -config['processing']['learning_rate_exp'])
weight_decay = math.pow(10, -config['processing']['weight_decay_exp']) if config['processing']['use_weight_decay'] else 0.0
optimizer_name = str(config['processing']['optimizer']).lower()
if optimizer_name == 'adam':
    optimizer = optim.Adam(get_model_head(model).parameters(), lr=learning_rate, weight_decay=weight_decay)
elif optimizer_name == 'sgd':
    optimizer = optim.SGD(get_model_head(model).parameters(), lr=learning_rate, momentum=config['processing']['momentum'])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# %% [markdown]
# Train the machine learning model

# %%
model_data = ModelData(train_dataset, validation_dataset, train_dataloader, validation_dataloader)
checkpoint_filepath = os.path.join(config['paths']['machine_learning_dir'], 'checkpoints', 'chk.pt')
print(f'Model Checkpoints will be stored in: {checkpoint_filepath}')
output_filepath = os.path.join(config['paths']['machine_learning_dir'], 'output', config['output']['output_name'])
print(f'The results will be stored in: {output_filepath}')
model_trained = train_model(model, device, model_data, criterion, outputs_to_predictions, optimizer, scheduler,
                            checkpoint_filepath, num_epochs=config['processing']['num_epochs'],
                            batch_accumulation=config['processing']['batch_accumulation'], config=config)
print('Copying data from checkpoint to results..')
state = load_state(checkpoint_filepath)
save_state(output_filepath, state)
print(f'Results stored in: {output_filepath}')
print('Done!')


