import phyper
from typing import List


# describe the hyperparameters
class NonKeys:
    n_epochs = 10
    batch_size = 1
    learning_rate = 0.001


class Instance(phyper.Parser, NonKeys):
    seed = 0
    n_hidden_layers = 3
    cv_k = 5
    cv_fold = 0
    umap_n_neighbors = 15
    umap_n_components = 2


# describe intermediate quantities depending on a subset of the hyperparameters
parser = Instance(hashed_resources_folder='derived_data')
parser.register_new_resource(name='nn_model', dependencies=['seed', 'n_hidden_layers', 'cv_k', 'cv_fold'])
parser.register_new_resource(name='umap',
                             dependencies=parser.get_dependencies_for_resources('nn_model') + ['umap_n_neighbors',
                                                                                               'umap_n_components'])

# specify and instanciate values for the hyperparameters
d = {'n_hidden_layers': [1, 2, 3],
     'cv_fold': list(range(5)),
     'umap_n_neighbors': [5, 15, 30],
     'umap_n_components': [2, 3]}

instances: List[Instance] = parser.get_instances_from_dictionary(d)
