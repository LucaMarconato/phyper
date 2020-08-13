import os
from config import Instance


def get_torch_model_path(instance: Instance):
    return os.path.join(instance.get_resources_path(), 'torch_model.torch')


def get_training_metrics_path(instance: Instance):
    return os.path.join(instance.get_resources_path(), 'training_metrics.hdf5')


def get_transformed_dataset_path(instance: Instance):
    return os.path.join(instance.get_resources_path('transformed_data'), 'transformed_data.npy')


def get_preprocessed_dataset_path(instance: Instance):
    return os.path.join(instance.get_resources_path('preprocessed_data'), 'preprocessed_data.npy')


processed_data_folder = 'processed_data'
os.makedirs(processed_data_folder, exist_ok=True)


def get_cross_validation_scores_path():
    return os.path.join(processed_data_folder, 'cv_scores.csv')


def get_best_model_torch_model_path():
    return os.path.join(processed_data_folder, 'best_model.torch')


def get_best_model_training_metrics_path():
    return os.path.join(processed_data_folder, 'best_model_training_metrics.hdf5')


jupyter_plot_folder = os.path.join(processed_data_folder, 'jupyter_plots')
os.makedirs(jupyter_plot_folder, exist_ok=True)


def jupyter_plot(filename):
    return os.path.join(jupyter_plot_folder, filename)

# def get_best_model_test_set_predictions_path():
#     return os.path.join(processed_data_folder, 'best_model_test_set_predictions.hdf5')
