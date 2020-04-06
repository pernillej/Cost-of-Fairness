from aif360.datasets import GermanDataset, StructuredDataset
import numpy as np
import pandas as pd


def load_german_dataset():
    """
    Collect the aif360 preprocessed German Credit Data Set.
    Assigns 'age' as the protected attribute with age >= 25 considered privileged.
    Sex-related attributes are removed (the other option for privileged attribute)

    :return: The German Credit Data Set
    """
    dataset = GermanDataset(
        protected_attribute_names=['age'],  # Only use 'age' as protected attr., not 'sex' which is also in this dataset
        privileged_classes=[lambda x: x >= 25],  # age >= 25 is considered privileged
        features_to_drop=['personal_status', 'sex']  # Remove sex-related attributes
    )
    return dataset.split([0.8], shuffle=True)


def to_dataframe(dataset):
    """
    Convert the ai360 data set into a Pandas Dataframe

    :param dataset: aif360 StructuredDataset type to convert into a dataframe
    :return: Tuple containing:
        - The converted dataframe
        - Dictionary of attributes with the following structure:
          attributes = {
            "feature_names": self.feature_names,
            "label_names": self.label_names,
            "protected_attribute_names": self.protected_attribute_names,
            "instance_names": self.instance_names,
            "instance_weights": self.instance_weights,
            "privileged_protected_attributes": self.privileged_protected_attributes,
            "unprivileged_protected_attributes": self.unprivileged_protected_attributes
          }
    """
    return dataset.convert_to_dataframe()


def from_dataframe(dataframe, attributes):
    """
    Convert Pandas Dataframe into aif360 StructuredDataset

    :param dataframe: The dataframe to convert
    :param attributes: Dictionary of attributes relating to the dataframe
    :return: aif360 StructuredDataset type generated from the params
    """
    dataset = StructuredDataset(df=dataframe, label_names=attributes["label_names"],
                                protected_attribute_names=attributes["protected_attribute_names"],
                                unprivileged_protected_attributes=attributes["unprivileged_protected_attributes"],
                                privileged_protected_attributes=attributes["privileged_protected_attributes"])
    return dataset


def get_X_and_y(dataframe, label_name):
    X = dataframe.drop(label_name, axis=1)
    y = dataframe[label_name]
    return X, y
