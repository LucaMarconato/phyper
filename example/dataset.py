from config import instances, Instance
from sklearn.model_selection import KFold, StratifiedKFold

from torch.utils.data import Dataset, SequentialSampler, DataLoader
from sklearn.datasets import load_iris
import itertools
import torch
import numpy as np
from paths import get_preprocessed_dataset_path, get_transformed_dataset_path
import time


# just for the purpose of presenting this library, we suppose that applying a transformation to the data (
# identity/log/square) and centering the data (removing the mean) are both super heavy steps and so we want to pickle
# the intermediate results in a file to avoid repeating those operations unnecessarity
class Iris(Dataset):
    def __init__(self, instance: Instance):
        super().__init__()
        self.iris = load_iris()
        self.target = self.iris.target
        path = get_preprocessed_dataset_path(instance)
        self.data = np.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i, :]
        y = self.target[i]
        x = torch.tensor(x).float()
        y = torch.tensor(y).long()
        return x, y

    def get_targets(self):
        return self.iris.target


def transform_data(instance: Instance):
    iris = load_iris()
    data = iris.data

    # super heavy transformation operation
    print('transforming data... ', end='')
    time.sleep(5)
    s = instance.transformation
    if s == 'identity':
        pass
    elif s == 'log':
        data = np.log(data + 1)
    elif s == 'square':
        data = np.square(data)
    else:
        raise RuntimeError(f'method unknown: {s}')
    print('DONE')

    path = get_transformed_dataset_path(instance)
    np.save(path, data)


def load_transformed_data(instance: Instance):
    path = get_transformed_dataset_path(instance)
    return np.load(path)


def center_data(instance: Instance):
    path_in = get_transformed_dataset_path(instance)
    path_out = get_preprocessed_dataset_path(instance)
    data = np.load(path_in)

    print('centering data...', end='')
    time.sleep(5)
    if instance.centering:
        means = np.mean(data, axis=0)
        data -= means
    np.save(path_out, data)
    print('DONE')


def get_data_loaders(instance: Instance):
    # extract the variable "test": the indices in the iris dataset of the test set
    iris = Iris(instance)
    l = list(range(len(iris)))
    y = iris.get_targets()
    to_shuffle = list(zip(l, y))
    np.random.shuffle(to_shuffle)
    l, y = zip(*to_shuffle)
    l = np.array(l).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    training_validation_indices, test_indices = StratifiedKFold(n_splits=(instance.cv_k + 1)).split(l,
                                                                                                 y).__iter__().__next__()
    training_validation = l[training_validation_indices, :].tolist()
    test = l[test_indices, :].ravel().tolist()

    # extract the variables "training" and "validation": the indices in the iris dataset of the training and
    # validation set for the current cross-validation fold
    splits = KFold(n_splits=instance.cv_k).split(training_validation)
    training_indices, validation_indices = next(itertools.islice(splits, instance.cv_fold, None))
    a = np.array(training_validation)
    training = a[training_indices, :].ravel().tolist()
    validation = a[validation_indices, :].ravel().tolist()
    assert len(set(training) | set(validation) | set(test)) == len(l)

    # create samplers and then data loader based on the indices above
    training_sampler = SequentialSampler(training_indices)
    validation_sampler = SequentialSampler(validation_indices)
    training_validation_sampler = SequentialSampler(training_validation_indices)
    test_sampler = SequentialSampler(test_indices)

    training_loader = DataLoader(iris, batch_size=instance.batch_size, sampler=training_sampler)
    validation_loader = DataLoader(iris, batch_size=instance.batch_size, sampler=validation_sampler)
    training_validation_loader = DataLoader(iris, batch_size=instance.batch_size, sampler=training_validation_sampler)
    test_loader = DataLoader(iris, batch_size=instance.batch_size, sampler=test_sampler)

    return training_loader, validation_loader, training_validation_loader, test_loader


if __name__ == '__main__':
    instance = instances[0]
    transform_data(instance)
    center_data(instance)
    # preprocess_data(instance)
    # training_loader, validation_loader, test_loader = get_data_loaders(instance)
    # for x, y in training_loader:
    #     print(x.shape, y.shape)
