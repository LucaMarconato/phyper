from __future__ import annotations
import os
import hashlib
from collections import OrderedDict
from typing import List, Optional, Callable, Dict, Any
import pandas as pd
import colorama
import pprint


class Parser:
    def __init__(self, hashed_resources_folder: str):
        self._hashed_resources_folder = hashed_resources_folder
        os.makedirs(self._hashed_resources_folder, exist_ok=True)
        self._models_folder = os.path.join(self._hashed_resources_folder, 'all')
        os.makedirs(self._models_folder, exist_ok=True)
        self._is_parser = True
        self._dependencies = {}
        self._parser = None

    def new_instance(self):
        instance = self.__new__(type(self))
        instance._is_parser = False
        instance._parser = self
        d = self.get_hyperparameters(calling_from_pure_parser=True)
        for k, v in d.items():
            setattr(instance, k, v)
        return instance

    def get_hyperparameters(self, calling_from_pure_parser=False):
        keys = dir(self)
        if calling_from_pure_parser:
            keys = [key for key in keys if key not in self.__dict__]
            keys = [key for key in keys if not hasattr(Parser, key)]
        keys = [key for key in keys if not callable(getattr(self, key))]
        keys = [key for key in keys if not key.startswith('_')]
        values = [getattr(self, key) for key in keys]
        d = dict(zip(keys, values))
        return d

    def get_hashable_hyperparameters(self):
        d = self.get_hyperparameters()
        hashable_hyperparameters = set(type(self).__dict__.keys()).intersection(d.keys())
        return hashable_hyperparameters

    def get_instance_hash(self, resource_name: Optional[str] = None):
        assert self._is_parser is False
        h = hashlib.sha256()
        keys = self.get_hashable_hyperparameters()
        d = self.get_hyperparameters()
        od = OrderedDict(sorted({k: d[k] for k in keys}.items()))
        for k, v in od.items():
            use_hash = True
            if resource_name is not None:
                dependencies = self._parser._dependencies[resource_name]
                if k not in dependencies:
                    use_hash = False
            if use_hash:
                h.update(str(k).encode('utf-8'))
                h.update(str(v).encode('utf-8'))
        hd = h.hexdigest()

        if resource_name is None:
            hash_folder = os.path.join(self._parser._models_folder, hd)
        else:
            hash_folder = os.path.join(self._parser._hashed_resources_folder, resource_name, hd)
        os.makedirs(hash_folder, exist_ok=True)
        return hd

    def register_new_resource(self, name: str, dependencies: List[str]):
        assert self._is_parser is True
        l = self.get_hashable_hyperparameters()
        for k in dependencies:
            assert k in l
        self._dependencies[name] = dependencies
        path = os.path.join(self._hashed_resources_folder, name)
        os.makedirs(path, exist_ok=True)

    def get_dependencies_for_resources(self, name: str):
        if self._is_parser:
            parser = self
        else:
            parser = self._parser
        return parser._dependencies[name]

    def get_resources_path(self, resource_name: Optional[str] = None):
        if self._is_parser:
            parser = self
        else:
            parser = self._parser
        if resource_name is None:
            return os.path.join(parser._models_folder, self.get_instance_hash())
        else:
            return os.path.join(parser._hashed_resources_folder, resource_name, self.get_instance_hash(resource_name))

    @staticmethod
    def cartesian_product(d):
        index = pd.MultiIndex.from_product(d.values(), names=d.keys())
        return pd.DataFrame(index=index).reset_index()

    def get_instances_from_df(self, df):
        assert self._is_parser is True
        instances = []
        for _, row in df.iterrows():
            instance = self.new_instance()
            d = row.to_dict()
            for k, v in d.items():
                setattr(instance, k, v)
            instances.append(instance)
        hashes = [instance.get_instance_hash() for instance in instances]
        if len(hashes) != len(set(hashes)):
            print(f'{colorama.Fore.YELLOW}warning: some instances differ only by non-hashable parameters, review your '
                  f'list to avoid unnecessary computations', colorama.Fore.RESET)
        return instances

    def get_instances_from_dictionary(self, d):
        assert self._is_parser is True
        df = self.cartesian_product(d)
        return self.get_instances_from_df(df)

    @staticmethod
    def get_resources(instances: List[Parser], resource_name: str) -> List[Parser]:
        unique = {}
        for instance in instances:
            resource_hash = instance.get_instance_hash(resource_name)
            unique[resource_hash] = instance
        # maybe here I should set all the other hyperparameters to None
        return list(unique.values())

    @staticmethod
    def get_projections(instances: List[Parser], hyperparameter_names: List[str]):
        df = pd.DataFrame(columns=hyperparameter_names)
        rows = []
        # maybe I should add a test to check the all the keys are equal among instances, as expected
        for instance in instances:
            d = instance.get_hyperparameters()
            row = {name: d[name] for name in hyperparameter_names}
            rows.append(row)
        df = df.append(rows)
        df.drop_duplicates(inplace=True)
        return df

    @staticmethod
    def get_filtered_instances(instances: List[Parser], hyperparameters: Dict[str, Any]):
        l = list()
        for instance in instances:
            d = instance.get_hyperparameters()
            for k, v in hyperparameters.items():
                if d[k] != v:
                    break
            else:
                l.append(instance)
        return l

    @staticmethod
    def get_instances_hashes(instances: List[Parser], resource_name: Optional[str] = None):
        hashes = [instance.get_instance_hash(resource_name=resource_name) for instance in instances]
        return hashes

    @staticmethod
    def get_instance_from_hash(instance_hash: str, instances: List[Parser], resource_name: Optional[str] = None):
        l = [instance for instance in instances if instance.get_instance_hash(resource_name) == instance_hash]
        # maybe here, when resource_name is not None, we want to delete the hyperparamters that are not relative to the resource
        # in fact otherwise one could have subtle bugs
        return l[0]

    @staticmethod
    def snakemake_helper_get_model_descriptions_paths(instance: Parser):
        assert instance._is_parser is False
        paths = []
        for resource_name in instance._dependencies.keys():
            path = Parser._snakemake_helper_get_model_description_path_for_resource(instance, resource_name)
            paths.append(path)
        return paths

    @staticmethod
    def _snakemake_helper_get_model_description_path_for_resource(instance: Parser, resource_name: str):
        path = os.path.join(instance.get_resources_path(resource_name), 'parameters.json')
        return path

    @staticmethod
    def snakemake_helper_log_hyperparameters(instances: List[Parser]):
        # here I should maybe put an integrity check and test that all the instances have all the same set of resources
        instance0 = instances[0]
        for resource_name in instance0._dependencies.keys():
            path = Parser._snakemake_helper_get_model_description_path_for_resource(instance0, resource_name)
            s = ''
            for instance in instances:
                d = instance.get_hyperparameters()
                dependencies = instance.get_dependencies_for_resources(resource_name)
                d = {k: v for k, v in d.items() if k in dependencies}
                s += pprint.pformat(d)
                s += '\n'
            with open(path, 'w') as f:
                f.write(s)

    @staticmethod
    def _snakemake_helper_get_wildcarded_path(path: str, instance: Parser, resource_name: Optional[str] = None):
        h = instance.get_instance_hash(resource_name)
        assert path.count(h) == 1
        return path.replace(h, f'{{{resource_name + "_" if resource_name is not None else ""}hash}}')

    @staticmethod
    def snakemake_helper_get_wildcarded_path(path_function: Callable[[Parser], str], instance: Parser,
                                             resource_name: Optional[str] = None):
        wildcarded_path = Parser._snakemake_helper_get_wildcarded_path(path_function(instance),
                                                                       instance,
                                                                       resource_name)
        return wildcarded_path
