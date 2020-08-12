import torch
import torch.optim
import torch.nn as nn
import h5py
import pandas as pd

from config import Instance
from model import Model
from dataset import get_data_loaders
from paths import get_torch_model_path, get_training_metrics_path, get_best_model_torch_model_path, \
    get_best_model_training_metrics_path, get_cross_validation_scores_path


def train(instance: Instance, train_also_on_validation_set=False):
    model = Model(instance)
    cross_entropy_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=instance.learning_rate)
    training_loader, validation_loader, training_validation_loader, test_loader = get_data_loaders(instance)

    if not train_also_on_validation_set:
        training_metrics_h5 = h5py.File(get_training_metrics_path(instance), 'w')
    else:
        training_metrics_h5 = h5py.File(get_best_model_training_metrics_path(), 'w')

    for epoch in range(instance.n_epochs):
        model.train()
        training_loss = 0
        training_n_correct = 0
        if not train_also_on_validation_set:
            loader = training_loader
        else:
            loader = training_validation_loader
        for x, true_y in loader:
            optimizer.zero_grad()
            y = model(x)
            loss = cross_entropy_loss(y, true_y)
            loss.backward()
            training_loss += loss.item()
            training_n_correct += torch.sum((true_y == torch.max(y, 1)[1]).double())
            optimizer.step()

        if epoch % instance.log_interval == 0 or epoch == instance.n_epochs - 1:
            training_accuracy = 100 * training_n_correct / len(loader.sampler)
            print(f'(Train) Epoch [{epoch}/{instance.n_epochs}] Loss: {training_loss:.2f} Acc: {training_accuracy:.2f}')

            if not train_also_on_validation_set:
                path = get_torch_model_path(instance)
            else:
                path = get_best_model_torch_model_path()

            training_metrics_h5[f'epoch{epoch}/training_loss'] = training_loss
            training_metrics_h5[f'epoch{epoch}/training_accuracy'] = training_accuracy
            torch.save(model.state_dict(), path)

            if not train_also_on_validation_set:
                validation_or_test_loader = validation_loader
                s0 = 'Val'
                s1 = 'validation'
            else:
                validation_or_test_loader = test_loader
                s0 = 'Test'
                s1 = 'test'
            model.eval()
            validation_or_test_loss = 0
            validation_or_test_n_correct = 0
            with torch.no_grad():
                for x, true_y in validation_or_test_loader:
                    y = model(x)
                    loss = cross_entropy_loss(y, true_y)
                    validation_or_test_loss += loss.item()
                    validation_or_test_n_correct += torch.sum((true_y == torch.max(y, 1)[1]).double())
                validation_or_test_accuracy = 100 * validation_or_test_n_correct / len(validation_or_test_loader.sampler)
                print(
                    f'({s0}) Epoch [{epoch}/{instance.n_epochs}] Loss: {validation_or_test_loss:.2f} Acc: {validation_or_test_accuracy:.2f}')
                training_metrics_h5[f'epoch{epoch}/{s1}_loss'] = validation_or_test_loss
                training_metrics_h5[f'epoch{epoch}/{s1}_accuracy'] = validation_or_test_accuracy
    training_metrics_h5.close()


def train_best_model():
    path = get_cross_validation_scores_path()
    df = pd.read_csv(path)
    min_loss = df['average_validation_loss'].min()
    bests = df.loc[df['average_validation_loss'] == min_loss, 'cross_validated_model_hash']
    best = bests.iloc[0]
    from config import instances
    instance = Instance.get_instance_from_hash(best, instances=instances, resource_name='cross_validated_model')
    train(instance, train_also_on_validation_set=True)


if __name__ == '__main__':
    train_best_model()
    # from example.config import instances
    #
    # instance = instances[0]
    # train(instance)
