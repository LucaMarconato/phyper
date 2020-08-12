from config import instances, Instance
from sklearn.model_selection import KFold, StratifiedKFold

from torch.utils.data import Dataset, SequentialSampler, DataLoader
from sklearn.datasets import load_iris
import itertools
import torch
import numpy as np
from paths import get_preprocessed_dataset_path
import dill


# just for the purpose of presenting this library, we suppose that we perform a super heavy prepreprocessing step to
# the data that makes it convenient to do it only once (in the __init__ function of Iris(Dataset)). We then pickle
# the data and load the pickled data when training models
class Iris(Dataset):
    def __init__(self, instance: Instance):
        super().__init__()
        self.iris = load_iris()
        self.preprocessing_method = instance.preprocessing_method
        self.raw_data = self.iris.data

        # super heavy preprocessing function
        def preprocess_data(data, method):
            import time
            # 20 seconds of heavy work
            time.sleep(20)
            if method == 'identity':
                return data
            elif method == 'log':
                return np.log(data + 1)
            elif method == 'square':
                return np.square(data)
            else:
                raise RuntimeError(f'method unknown: {method}')

        print('preprocesing data... ', end='')
        self.data = preprocess_data(self.raw_data, method=self.preprocessing_method)
        print('DONE')

        self.target = self.iris.target

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


def preprocess_data(instance: Instance):
    iris = Iris(instance)
    path = get_preprocessed_dataset_path(instance)
    dill.dump(iris, open(path, 'wb'))


def load_preprocess_data(instance: Instance):
    path = get_preprocessed_dataset_path(instance)
    return dill.load(open(path, 'rb'))


def get_data_loaders(instance: Instance):
    # extract the variable "test": the indices in the iris dataset of the test set
    iris = load_preprocess_data(instance)
    l = list(range(len(iris)))
    y = iris.get_targets()
    to_shuffle = list(zip(l, y))
    np.random.shuffle(to_shuffle)
    l, y = zip(*to_shuffle)
    l = np.array(l).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    train_validation_indices, test_indices = StratifiedKFold(n_splits=(instance.cv_k + 1)).split(l,
                                                                                                 y).__iter__().__next__()
    training_validation = l[train_validation_indices, :].tolist()
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
    test_sampler = SequentialSampler(test_indices)

    training_loader = DataLoader(iris, batch_size=instance.batch_size, sampler=training_sampler)
    validation_loader = DataLoader(iris, batch_size=instance.batch_size, sampler=validation_sampler)
    test_loader = DataLoader(iris, batch_size=instance.batch_size, sampler=test_sampler)

    return training_loader, validation_loader, test_loader


if __name__ == '__main__':
    instance = instances[0]
    preprocess_data(instance)
    training_loader, validation_loader, test_loader = get_data_loaders(instance)
    for x, y in training_loader:
        print(x.shape, y.shape)
