import phyper


def print_hyperparameters():
    print('printing hyperparameters from imported file')
    phyper.print_hyperparameters()


def print_hyperparameters2():
    print('printing hyperparameters (2nd way) from imported file')
    import pprint
    pprint.pprint(phyper.hyperparameters)


def print_instances():
    print('printing instances')
    my_instance = phyper.new_parsed_instance()
    instance = phyper.new_instance()
    print(my_instance.my_test)
    print(instance.a)
    print(my_instance.my_test)
    print(instance.a)
    mmy_instance = phyper.new_instance()
    print(mmy_instance.test)
    print(mmy_instance.test)