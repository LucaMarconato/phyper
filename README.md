# Phyper
A machine learning pipeline trains models depending on many hyperparamters, and precomputes intermediate resources which depend on a subset of all the hyperparameters. Some hyperparameters should define a unique model, while others are required to be specified for running the pipeline but should not lead to different storage of intermediate resources or model outputs. 

This library simplifies the usage of hyperparameters within pipelines.

Features:
* easy, convenient syntax for defining hyperparameters
* use hyperparameters in any point within a pipeline, from data preprocessing to the final ML model
* IDE-enabled code completion for hyperparameters (tested with PyCharm)
* generating hashes for final outputs and intermediate resources to avoid redundant computations or resource clashing between different models; the hashes depend on a predefined subset of hyperparameters
* helper functions for using Snakemake as a pipeline manager
