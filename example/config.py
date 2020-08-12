import phyper
from typing import List


# describe the hyperparameters
class NonKeys:
    n_epochs = 20
    batch_size = 32
    learning_rate = 0.001
    log_interval = 20


class Instance(phyper.Parser, NonKeys):
    seed = 0
    n_hidden_layers = 3
    cv_k = 5
    cv_fold = 0
    # for the sake of presenting how the library works, please imagine this variable to specify a heavy
    # preprocessing method to apply to the data, and that it takes MINUTES to compute
    #
    # possible values: 'identity' 'log', 'square'
    preprocessing_method = 'identity'


# describe intermediate quantities depending on a subset of the hyperparameters
parser = Instance(hashed_resources_folder='derived_data')
parser.register_new_resource(name='preprocessed_data', dependencies=['preprocessing_method'])

# specify and instanciate values for the hyperparameters
d = {'preprocessing_method': ['identity', 'log', 'square'],
     'n_hidden_layers': [1, 2, 3],
     'cv_fold': list(range(5))}

instances: List[Instance] = parser.get_instances_from_dictionary(d)
