from aif360.datasets import GermanDataset
import numpy as np
import pandas as pd


def load_german_dataset():
    dataset = GermanDataset(
        protected_attribute_names=['age'],  # Only use 'age' as protected attr., not 'sex' which is also in this dataset
        privileged_classes=[lambda x: x >= 25],  # age >= 25 is considered privileged
        features_to_drop=['personal_status', 'sex']  # Remove sex-related attributes
    )
    return dataset.split([0.8], shuffle=True)


