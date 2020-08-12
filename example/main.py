import click
from config import Instance, instances


def f(instance_hash, resource_name=None):
    return Instance.get_instance_from_hash(instance_hash, instances, resource_name)


@click.command()
@click.option('--instance-hash', type=str, required=True)
def transform_data(instance_hash):
    from dataset import transform_data
    transform_data(f(instance_hash, resource_name='transformed_data'))


@click.command()
@click.option('--instance-hash', type=str, required=True)
def preprocess_data(instance_hash):
    from dataset import center_data
    center_data(f(instance_hash, resource_name='preprocessed_data'))


@click.command()
@click.option('--instance-hash', type=str, required=True)
def train_model(instance_hash):
    from train import train
    train(f(instance_hash))

@click.command()
def train_best_model():
    from train import train_best_model
    train_best_model()

@click.group()
def cli():
    pass


cli.add_command(transform_data)
cli.add_command(preprocess_data)
cli.add_command(train_model)
cli.add_command(train_best_model)

if __name__ == '__main__':
    cli()
