import torch
import torch.optim
import torch.nn as nn
import h5py

from config import Instance
from model import Model
from dataset import get_data_loaders
from paths import get_torch_model_path, get_training_metrics_path


def train(instance: Instance):
    model = Model(instance)
    cross_entropy_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=instance.learning_rate)
    training_loader, validation_loader, test_loader = get_data_loaders(instance)

    training_metrics_h5 = h5py.File(get_training_metrics_path(instance))
    for epoch in range(instance.n_epochs):
        model.train()
        training_loss = 0
        training_n_correct = 0
        for x, true_y in training_loader:
            optimizer.zero_grad()
            y = model(x)
            loss = cross_entropy_loss(y, true_y)
            loss.backward()
            training_loss += loss.item()
            training_n_correct += torch.sum((true_y == torch.max(y, 1)[1]).double())
            optimizer.step()

        if epoch % instance.log_interval == 0:
            training_accuracy = 100 * training_n_correct / len(training_loader.sampler)
            print(f'(Train) Epoch [{epoch}/{instance.n_epochs}] Loss: {training_loss:.2f} Acc: {training_accuracy:.2f}')
            model.eval()
            validation_loss = 0
            validation_n_correct = 0
            with torch.no_grad():
                for x, true_y in validation_loader:
                    y = model(x)
                    loss = cross_entropy_loss(y, true_y)
                    validation_loss += loss.item()
                    validation_n_correct += torch.sum((true_y == torch.max(y, 1)[1]).double())
                validation_accuracy = 100 * validation_n_correct / len(validation_loader.sampler)
                print(
                    f'(Val) Epoch [{epoch}/{instance.n_epochs}] Loss: {validation_loss:.2f} Acc: {validation_accuracy:.2f}')

                torch.save(model.state_dict(), get_torch_model_path(instance))
                training_metrics_h5[f'epoch{epoch}/training_loss'] = training_loss
                training_metrics_h5[f'epoch{epoch}/training_accuracy'] = training_accuracy
                training_metrics_h5[f'epoch{epoch}/validation_loss'] = validation_loss
                training_metrics_h5[f'epoch{epoch}/validation_accuracy'] = validation_accuracy
    training_metrics_h5.close()


if __name__ == '__main__':
    from example.config import instances

    instance = instances[0]
    train(instance)
