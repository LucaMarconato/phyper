import phyper


# describing the hyperparameters and specifying the default values
class NonKeys:
    n_epochs = 2000
    batch_size = 32
    learning_rate = 0.001
    log_interval = 200


class Instance(phyper.Parser, NonKeys):
    seed = 0
    n_hidden_layers = 3
    cv_k = 5
    cv_fold = 0
    # for the sake of presenting how the library works, please imagine this variable to specify a heavy
    # preprocessing method to apply to the data, and that it takes MINUTES to compute
    #
    # possible values: 'identity' 'log', 'square'
    transformation = 'identity'
    # subtracting the mean after transformation
    centering = False


# describing intermediate quantities depending on a subset of the hyperparameters
parser = Instance(hashed_resources_folder='../example/derived_data')

parser.register_new_resource(name='transformed_data', dependencies=['transformation'])
parser.register_new_resource(name='preprocessed_data', dependencies=['transformation', 'centering'])
parser.register_new_resource(
    name='cross_validated_model',
    dependencies=parser.get_dependencies_for_resources('preprocessed_data') + ['n_hidden_layers'])

my_instance: Instance = parser.load_instance_from_disk_by_hash(
    '0d7146e6c9d5397a8c604924fc7d8d8b4b3efd903b462b9e7e22fc0275a5a280')
my_resource: Instance = parser.load_instance_from_disk_by_hash(
    '6dbab4c9ca812a269ead275cb01e655c19fc619feeb538a7e61821713d396eee', resource_name='transformed_data')

from pprint import pprint

pprint(my_instance.get_hyperparameters())
pprint(my_resource.get_hyperparameters())
