import os
from config import Instance


def get_torch_model_path(instance: Instance):
    return os.path.join(instance.get_resources_path(), f'torch_model_epoch.torch')


def get_training_metrics_path(instance: Instance):
    return os.path.join(instance.get_resources_path(), f'training_metrics_epoch.hdf5')


def get_preprocessed_dataset_path(instance: Instance):
    return os.path.join(instance.get_resources_path('preprocessed_data'), 'preprocessed_data.dill')
