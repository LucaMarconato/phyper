from config import Instance
from typing import List
from paths import get_training_metrics_path, get_cross_validation_scores_path
import h5py
import pandas as pd


def compute_score_for_each_model(instances: List[Instance]):
    cv_k = instances[0].cv_k
    df = Instance.get_projections(instances, hyperparameter_names=instances[0].get_dependencies_for_resources(
        'cross_validated_model'))
    df_scores = pd.DataFrame(columns=['cross_validated_model_hash', 'average_validation_loss'])
    for _, row in df.iterrows():
        d = row.to_dict()
        filtered_instances = Instance.get_filtered_instances(instances, d)
        assert len(filtered_instances) == cv_k
        assert set(Instance.get_projections(filtered_instances, ['cv_fold']).cv_fold.to_list()) == set(range(cv_k))
        average_validation_loss = 0
        for instance in filtered_instances:
            path = get_training_metrics_path(instance)
            with h5py.File(path, 'r') as f5:
                keys = f5.keys()
                # keys are like ['epoch10', 'epoch20', ...]
                epochs = [int(s.split('epoch')[1]) for s in keys]
                last_epoch = max(epochs)
                metrics = f5[f'epoch{last_epoch}']
                validation_loss = metrics['validation_loss'][...].item()
                average_validation_loss += validation_loss
        average_validation_loss /= cv_k
        instance_hash = filtered_instances[0].get_instance_hash(resource_name='cross_validated_model')
        assert len(
            set([instance.get_instance_hash(resource_name='cross_validated_model') for instance in filtered_instances]))
        df_scores = df_scores.append(
            {'cross_validated_model_hash': instance_hash, 'average_validation_loss': average_validation_loss},
            ignore_index=True)
    path = get_cross_validation_scores_path()
    df_scores.to_csv(path)


if __name__ == '__main__':
    from config import instances

    compute_score_for_each_model(instances)
