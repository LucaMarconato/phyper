import click
from config import Instance, instances


def f(instance_hash, resource_name=None):
    return Instance.get_instance_from_hash(instance_hash, instances, resource_name)


@click.command()
@click.option('--instance-hash', type=str, required=True)
def train_model(instance_hash):
    from example.train import train
    train(f(instance_hash))


@click.command()
@click.option('--instance-hash', type=str, required=True)
# @click.option('--resource-name', type=str, required=False)
def preprocess_data(instance_hash): #, resource_name=None
    from example.dataset import preprocess_data
    preprocess_data(f(instance_hash, resource_name='preprocessed_data'))


@click.group()
def cli():
    pass


cli.add_command(train_model)
cli.add_command(preprocess_data)

if __name__ == '__main__':
    cli()
