import phyper
from test.test_import import print_hyperparameters, print_hyperparameters2, print_instances

phyper.set_hashed_resources_folder('hashed_resources')
phyper.new_hyperparameter('test', is_key=False)
phyper.new_hyperparameter('a', is_key=True, default_value=2)

# instance = phyper.Instance()
# print(f'instance.a = {instance.a}')
# print(f'vars(instance) = {vars(instance)}')

import argparse


# parser = argparse.ArgumentParser()
# parser.add_argument('--test', type=int, default=0, required=False)
# args = parser.parse_args()
# print(args.test)
# print(args.testa)

class MyPharser(phyper.Parser):
    my_test = 2
    ehi = 3


class NonKeys(phyper.Parser):
    epochs = 10
    batch_size = 10


# pharser = phyper.Parser()
# pharser.new_hyperparameter('test', is_key=True, default_value=4)
# instance = pharser.new_instance()
# print(instance.test)
# # print(instance.testa)
# print('----------')
# print_hyperparameters()
# print_hyperparameters2()
# print('--------')
# my_pharser = MyPharser()
# my_instance = my_pharser.new_instance()
# print(my_pharser.my_test)
# print(my_pharser.ehi)
#
#
my_parser = MyPharser()
phyper.set_parser(my_parser)
# phyper.set_hyperparameters(my_parser)
# instance = phyper.Instance()
# print(instance.my_test)

my_instance = phyper.new_parsed_instance()
print(my_instance.my_test)
print(my_instance.ehi)
# print(instance.ehii)
instance = phyper.new_instance()
print(instance.test)
print(instance.a)

print_instances()

