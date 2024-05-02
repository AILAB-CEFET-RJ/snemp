from os import cpu_count
from hiclass import LocalClassifierPerNode

def fit_model(x_train, y_train, base_model, hierarchical_classifier, binary_policy):
    # base_model = base_model(**base_model_kwargs) # max_iter=1000
    hierarchical_classifier_kwargs = {}
    if hierarchical_classifier is LocalClassifierPerNode:
        if binary_policy:
            hierarchical_classifier_kwargs['binary_policy'] = binary_policy

    model = hierarchical_classifier(
        local_classifier=base_model,
        n_jobs=cpu_count(),
        **hierarchical_classifier_kwargs
    )

    model.fit(x_train, y_train)
    return model