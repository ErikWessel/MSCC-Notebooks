import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
import torchvision
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix,
                             precision_recall_fscore_support)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import EfficientNet, SwinTransformer
from tqdm.auto import tqdm

class_names =   ['CLR', 'FEW', 'SCT', 'BKN', 'OVC']
classes = list(range(len(class_names)))
class_dict = {name: i for i, name in enumerate(class_names)}

def get_model_head(model:nn.Module):
    if isinstance(model, SwinTransformer):
        return model.head
    elif isinstance(model, EfficientNet):
        return model.classifier
    else:
        raise RuntimeError(f'Unknown model of type {type(model)}')

def set_model_head(model:nn.Module, head:nn.Module):
    if isinstance(model, SwinTransformer):
        model.head = head
    elif isinstance(model, EfficientNet):
        model.classifier = head
    else:
        raise RuntimeError(f'Unknown model of type {type(model)}')

def get_num_features(model:nn.Module):
    head = get_model_head(model)
    if isinstance(model, SwinTransformer):
        return head.in_features
    elif isinstance(model, EfficientNet):
        return head[1].in_features
    else:
        raise RuntimeError(f'Unknown model of type {type(model)}')

class DatasetType(str, Enum):
    TRAINING = 'training'
    VALIDATION = 'validation'
    TEST = 'test'

class AimlsseImageDataset(Dataset):
    def __init__(self, type:DatasetType, dataset_dir:str, transfrom=None, target_transform=None, use_manual_labels=False) -> None:
        super().__init__()
        self.img_labels = pd.read_csv(os.path.join(dataset_dir, f'{type.value}_labels.csv'))
        self.img_dir = os.path.join(dataset_dir, type.value)
        self.transform = transfrom
        self.target_transform = target_transform
        self.use_manual_labels = use_manual_labels
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, index) -> Tuple[np.ndarray, int, int]:
        image = self.get_image(index)
        label = self.get_label(index)
        return image, label, index
    
    def get_image(self, index) -> np.ndarray:
        path = self.get_image_path(index)
        image = rasterio.open(path)
        image_tensor = torch.from_numpy(image.read()).clamp(0.001, 1)
        image.close()
        # Select label and perform transformations
        if self.transform:
            image_tensor = self.transform(image_tensor)
        return image_tensor

    def get_label(self, index) -> int:
        if self.use_manual_labels:
            label = class_dict[self.img_labels.iloc[index]['true cloud cover']]
        else:
            label = class_dict[self.img_labels.iloc[index]['max cloud cover']]
        if self.target_transform:
            label = self.target_transform(label)
        return label
    
    def get_timedelta_minutes(self, index) -> float:
        return self.img_labels.iloc[index]['timedelta [minutes]']
    
    def get_image_path(self, index) -> str:
        return os.path.join(self.img_dir, self.img_labels.iloc[index]['station'],
                                  f'{self.img_labels.iloc[index].product_id}.tif')

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def imshow(input_tensor: torch.Tensor, figsize:Tuple[float, float], title=None) -> Tuple[plt.Figure, plt.Axes]:
    """Imshow for Tensor."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    ax.imshow(input_tensor.permute(1, 2, 0))
    return fig, ax

def plot_samples(dataset:AimlsseImageDataset, batch_size:int, batch_idx:int):
    sample_dataloader =  DataLoader(dataset, batch_size=batch_size, shuffle=False)
    # Get a batch of training data
    i = 0
    data_iter = iter(sample_dataloader)
    while i <= batch_idx:
        sample_inputs, sample_classes, indices = next(data_iter)
        i += 1
    print(sample_inputs.shape, [class_names[c] for c in sample_classes.unique().numpy()])
    print(indices)

    # Make a grid from batch
    out = torchvision.utils.make_grid(sample_inputs)
    out.shape
    
    fig, ax = imshow(out, (20, 20))
    for i, row in enumerate(list(chunks([class_names[c] for c in sample_classes.numpy()], 8))):
        for j, label in enumerate(row):
            sample_idx = i * 8 + j
            data_idx = int(indices[sample_idx])
            timedelta_minutes = dataset.get_timedelta_minutes(data_idx)
            ax.text(100 + j * 300, -15 + i * 730, f'{label}\n{timedelta_minutes:.1f}', fontsize=24)

class MLState:
    model_state: Dict = {}
    optimizer_state: Dict = {}
    epoch: int = 0
    scoreboard: List = []
    config: Optional[Dict] = None
    
    def __init__(self, model_state: Dict = {}, optimizer_state: Dict = {}, epoch: int = 0,
                 scoreboard: List = [], config: Optional[Dict] = None):
        self.model_state = model_state
        self.optimizer_state = optimizer_state
        self.epoch = epoch
        self.scoreboard = scoreboard
        self.config = config

def save_state(path:str, state:MLState):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': state.epoch,
        'model_state_dict': state.model_state,
        'optimizer_state_dict': state.optimizer_state,
        'scoreboard': state.scoreboard,
        'config': state.config
    }, path)
    
def load_state(path:str) -> MLState:
    if os.path.exists(path):
        checkpoint = torch.load(path)
        config = checkpoint['config'] if 'config' in checkpoint else None
        return MLState(
            checkpoint['model_state_dict'],
            checkpoint['optimizer_state_dict'],
            checkpoint['epoch'],
            checkpoint['scoreboard'],
            config
        )
    else:
        raise ValueError(f'{path} is not a valid path to a stored machine learning state')

class ConfusionMatrixNormalization(str, Enum):
    ALL_SAMPLES = 'all',
    TRUE_LABELS = 'true',
    PREDICTED_LABELS = 'pred'

def display_confusion_matrix(model:nn.Module, device, dataloader:DataLoader, outputs_to_predictions:Callable[[torch.Tensor], torch.Tensor],
                             normalization:Optional[ConfusionMatrixNormalization] = None) -> pd.DataFrame:
    normalization_mode = None
    title = 'Confusion matrix'
    if normalization is not None:
        normalization_mode = normalization.value
        title += ' [normalization: ' + normalization.name.lower().replace('_', ' ') + ']'
    # set the model to evaluation mode
    model.eval()

    # initialize the true and predicted labels
    true_labels: List = []
    predicted_labels: List = []
    all_indices: List = []

    time_start = time.perf_counter()
    # get predictions for N images from the dataloader
    with torch.no_grad():
        inputs: torch.Tensor
        labels: torch.Tensor
        indices: torch.Tensor
        for inputs, labels, indices in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            predicted = outputs_to_predictions(outputs)
            true_labels += labels.cpu().numpy().tolist()
            predicted_labels += predicted.cpu().numpy().tolist()
            all_indices += indices.tolist()
    time_end = time.perf_counter()
    print(f'{len(predicted_labels)} predictions in {time_end - time_start:.2f} seconds')

    # compute the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels, labels=range(len(classes)), normalize=normalization_mode)

    # display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(include_values=True, cmap='viridis', ax=None, xticks_rotation='horizontal')
    if normalization is not None:
        disp.ax_.get_images()[0].set_clim(0, 1)
    plt.grid(False)
    plt.title(title)
    plt.show()

    data_out = pd.DataFrame({'true label': true_labels, 'predicted label': predicted_labels, 'index': all_indices})
    print(data_out)
    data_out['true label'] = data_out['true label'].apply(lambda x: class_names[x])
    data_out['predicted label'] = data_out['predicted label'].apply(lambda x: class_names[x])
    return data_out

@dataclass
class ModelData:
    training_dataset:   AimlsseImageDataset
    validation_dataset: AimlsseImageDataset
    training_dataloader:    DataLoader
    validation_dataloader:  DataLoader

def train_model(model:nn.Module, device, data:ModelData, criterion:Callable[[List[List[float]], List[float]], torch.Tensor],
                outputs_to_predictions:Callable[[torch.Tensor], torch.Tensor], optimizer, scheduler, checkpoint_filepath, num_epochs=25,
                batch_accumulation:Union[int, str]=1, load_checkpoint=True, config:Optional[Dict]=None):
    since = time.time()
    assert isinstance(batch_accumulation, (int, str)), 'Batch accumulation needs to be either a number from 1 to len(dataloader) or the string \'all\''
    if isinstance(batch_accumulation, int):
        assert batch_accumulation > 0, 'The values of at least one batch need to be accumulated'
    else:
        assert batch_accumulation == 'all', 'The only allowed string is \'all\''
    os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)
    if load_checkpoint and os.path.exists(checkpoint_filepath):
        checkpoint = torch.load(checkpoint_filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        scoreboard = checkpoint['scoreboard']
    else:
        scoreboard = []
        start_epoch = 0

    if start_epoch >= num_epochs:
        print(f'Starting from epoch {start_epoch} but only wanted {num_epochs} epochs - returning current model instead')
        return model
    
    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                dataset = data.training_dataset
                dataloader = data.training_dataloader
                model.train()  # Set model to training mode
            else:
                dataset = data.validation_dataset
                dataloader = data.validation_dataloader
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            y_true = []
            y_pred = []
            start_time = time.perf_counter()
            inputs: torch.Tensor
            labels: torch.Tensor
            indices: torch.Tensor
            if batch_accumulation == 'all':
                batch_accumulation = len(dataloader)
            for batch_index, (inputs, labels, indices) in enumerate(tqdm(dataloader)):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = outputs_to_predictions(outputs)
                    loss = criterion(outputs, labels)
                    loss = loss / batch_accumulation

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()

                        if ((batch_index + 1) % batch_accumulation == 0) or (batch_index + 1 == len(dataloader)):
                            optimizer.step()
                            optimizer.zero_grad()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
            if phase == 'train':
                scheduler.step()
            end_time = time.perf_counter()
            phase_name = None
            if phase == 'train':
                phase_name = 'Training'
            elif phase == 'val':
                phase_name = 'Validation'
            delta_time = end_time - start_time
            print(f'{phase_name} took {delta_time:.1f} [s]')

            epoch_loss = running_loss / len(dataset)
            epoch_acc = running_corrects.double() / len(dataset)
            print(f'\tLoss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            label_precision, label_recall, label_f1_score, _ = precision_recall_fscore_support(y_true, y_pred, labels=classes)
            epoch_precision, epoch_recall, epoch_f1_score, _ = precision_recall_fscore_support(y_true, y_pred, labels=classes, average='weighted')
            for i, cc in enumerate(class_names):
                print(f'\t{cc} - Precision: {label_precision[i]:.3f} Recall: {label_recall[i]:.3f} F1-Score: {label_f1_score[i]:.3f}')
            print(f'Total -> Precision: {epoch_precision:.3f} Recall: {epoch_recall:.3f} F1-Score: {epoch_f1_score:.3f}')
            
            scoreboard_entry = {
                'epoch': epoch,
                'phase': phase_name,
                'loss': epoch_loss,
                'accuracy': epoch_acc,
                'precision': epoch_precision,
                'recall': epoch_recall,
                'f1_score': epoch_f1_score,
                'time': delta_time
            }
            for i, cc in enumerate(class_names):
                scoreboard_entry[f'precision {cc}'] = label_precision[i]
                scoreboard_entry[f'recall {cc}'] = label_recall[i]
                scoreboard_entry[f'f1_score {cc}'] = label_f1_score[i]

            scoreboard += [scoreboard_entry]
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scoreboard': scoreboard,
            'config': config
        }, checkpoint_filepath)
        
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')

    return model