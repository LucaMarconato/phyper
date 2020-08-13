import phyper
from typing import List


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


# TODO: maybe use an enum instead of strings for resources, to avoid typos and use code-completion

# describing intermediate quantities depending on a subset of the hyperparameters
parser = Instance(hashed_resources_folder='derived_data')

parser.register_new_resource(name='transformed_data', dependencies=['transformation'])
parser.register_new_resource(name='preprocessed_data', dependencies=['transformation', 'centering'])
parser.register_new_resource(
    name='cross_validated_model',
    dependencies=parser.get_dependencies_for_resources('preprocessed_data') + ['n_hidden_layers'])

# specifying and instanciating values for the hyperparameters
d = {'transformation': ['identity', 'log', 'square'],
     'centering': [True, False],
     'n_hidden_layers': [1, 2, 3],
     'cv_fold': list(range(5))}

instances: List[Instance] = parser.get_instances_from_dictionary(d)
