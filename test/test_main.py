import phyper
from typing import List
from pprint import pprint
import pandas as pd


class NonKeys:
    n_epochs = 11
    batch_size = 10
    resume_training = False
    another_non_key = True


class MyParser(phyper.Parser, NonKeys):
    my_testa: str = 1
    ehi = None
    bbbbb = 32
    c = 'ehi'


hashed_resources_folder = 'hashed_resources'
my_parser = MyParser(hashed_resources_folder)
my_parser.register_new_resource(name='normalizer', dependencies=['my_testa', 'ehi', 'bbbbb'])

print(my_parser.get_hyperparameters())
print(my_parser.get_hashable_hyperparameters())
my_instance = my_parser.new_instance()
my_instance.get_instance_hash()
print(my_instance.get_hyperparameters())
print(my_instance.get_hashable_hyperparameters())
print(my_instance.get_instance_hash())
print(my_instance.get_instance_hash('normalizer'))
# print(my_instance.get_instance_hash('c'))
print(my_instance.get_resources_path())
print(my_instance.get_resources_path('normalizer'))

d = {'n_epochs': [50], 'c': ['c0', 'c1'], 'my_testa': [1, 2, 3]}
instances: List[MyParser] = my_parser.get_instances_from_dictionary(d)

for instance in instances:
    print(instance.get_instance_hash())
