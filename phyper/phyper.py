import sys
import os
import pprint

this = sys.modules[__name__]
this.hashed_resources_folder = None
this.hyperparameters = dict()
this.is_key = dict()
this.no_instance_defined_yet = True
this.parser = None


def set_hashed_resources_folder(path):
    if this.hashed_resources_folder is None:
        this.hashed_resources_folder = path
        os.makedirs(this.hashed_resources_folder, exist_ok=True)
    else:
        if path != this.hashed_resources_folder:
            raise RuntimeError(path, this.hashed_resources_folder)


def new_hyperparameter(name: str, is_key: bool, default_value=None):
    assert this.no_instance_defined_yet is True
    if name in this.hyperparameters.keys():
        raise RuntimeError(f'{name} already defined')
    else:
        this.is_key[name] = is_key
        this.hyperparameters[name] = default_value


def print_hyperparameters():
    pprint.pprint(this.hyperparameters)


class Parser:
    def __init__(self):
        self.hyperparameters = dict()
        self.is_key = dict()

    def new_hyperparameter(self, name: str, is_key: bool, default_value=None):
        if name in self.hyperparameters.keys():
            raise RuntimeError(f'{name} already defined')
        else:
            self.is_key[name] = is_key
            self.hyperparameters[name] = default_value

    def new_instance(self):
        instance = ParsedInstance()
        for name, value in self.hyperparameters.items():
            setattr(instance, name, value)
        d = self.get_hyperparameters()
        for k, v in d.items():
            setattr(instance, k, v)
        return instance

    def get_hyperparameters(self):
        keys = dir(self)
        keys = [key for key in keys if key not in self.__dict__]
        keys = [key for key in keys if not hasattr(Parser, key)]
        keys = [key for key in keys if not key.startswith('_')]
        values = [getattr(self, key) for key in keys]
        d = dict(zip(keys, values))
        return d


class ParsedInstance():
    pass
    # def __eq__(self, other):
    #     if not isinstance(other, ParsedInstance):
    #         return NotImplemented
    #     return vars(self) == vars(other)

    # def __contains__(self, key):
    #     return key in self.__dict__


class Instance:
    def __init__(self):
        this.no_instance_defined_yet = False
        for name, value in this.hyperparameters.items():
            setattr(self, name, value)


def set_hyperparameters(parser: Parser):
    d = parser.get_hyperparameters()
    for k, v in d.items():
        new_hyperparameter(k, False, v)


def set_parser(parser: Parser):
    assert this.parser is None
    this.parser = parser

def new_parsed_instance():
    assert this.parser is not None
    instance = this.parser.new_instance()
    return instance

def new_instance():
    instance = Instance()
    return instance