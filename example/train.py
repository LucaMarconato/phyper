import torch
import torch.optim
import torch.nn as nn

from example.config import Instance
from example.model import Model
from example.dataset import get_data_loaders


def train(instance: Instance):
    model = Model(instance)
    cross_entropy_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=instance.learning_rate)
    training_loader, validation_loader, test_loader = get_data_loaders(instance)

    for epoch in range(instance.n_epochs):
        model.train()
        train_loss = 0
        for x, true_y in training_loader:
            optimizer.zero_grad()
            y = model(x)
            loss = cross_entropy_loss(y, true_y)
            loss.backward()
            print(loss)
            loss_item = loss.item()
            print(loss_item)
            train_loss += loss_item
            optimizer.step()
        pass
        # accuracy = 100 * torch.sum(true_y == torch.max(y.data, 1)[1]).double() / len(Y)
        # print('Epoch [%d/%d] Loss: %.4f   Acc: %.4f'
        #       % (epoch + 1, num_epoch, loss.item(), acc.item()))


# _, predicted = torch.max(out.data, 1)
#
# # get accuration
# print('Accuracy of the network %.4f %%' % (100 * torch.sum(Y == predicted).double() / len(Y)))

if __name__ == '__main__':
    from example.config import instances

    instance = instances[0]
    train(instance)
