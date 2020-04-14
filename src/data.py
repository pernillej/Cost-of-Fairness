from aif360.datasets import GermanDataset, StructuredDataset, BinaryLabelDataset
import numpy as np
import pandas as pd


def load_german_dataframe():
    """
    Collect the aif360 preprocessed German Credit Dataset as a Pandas Dataframe

    :return: The German Credit Data in a Dataframe
    """
    dataset = load_german_dataset()
    return to_dataframe(dataset, favorable_label=1., unfavorable_label=2.)  # Labels specific to German dataset


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
    return dataset


def to_dataframe(dataset, favorable_label=None, unfavorable_label=None):
    """
    Convert the ai360 data set into a Pandas Dataframe

    :param dataset: aif360 StructuredDataset type to convert into a dataframe
    :param favorable_label: Favorable label value to add to attributes, in case of binary label. Default: None
    :param unfavorable_label: Unfavorable label value to add to attributes, in case of binary label. Default: None
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
    df, attributes = dataset.convert_to_dataframe()

    attributes["favorable_label"] = favorable_label
    attributes["unfavorable_label"] = unfavorable_label

    return df, attributes


def dataframe_to_dataset(dataframe, attributes):
    """
    Convert Pandas Dataframe into aif360 Dataset, either a BinaryLabelDataset or the base class StructuredDataset

    :param dataframe: The dataframe to convert
    :param attributes: Dictionary of attributes relating to the dataframe
    :return: aif360 Dataset type generated from the params
    """
    if attributes["favorable_label"] and attributes["unfavorable_label"]:
        dataset = BinaryLabelDataset(df=dataframe,
                                     favorable_label=attributes["favorable_label"],
                                     unfavorable_label=attributes["unfavorable_label"],
                                     label_names=attributes["label_names"],
                                     protected_attribute_names=attributes["protected_attribute_names"],
                                     unprivileged_protected_attributes=attributes["unprivileged_protected_attributes"],
                                     privileged_protected_attributes=attributes["privileged_protected_attributes"])
    else:
        dataset = StructuredDataset(df=dataframe, label_names=attributes["label_names"],
                                    protected_attribute_names=attributes["protected_attribute_names"],
                                    unprivileged_protected_attributes=attributes["unprivileged_protected_attributes"],
                                    privileged_protected_attributes=attributes["privileged_protected_attributes"])
    return dataset


def get_privileged_and_unprivileged_groups(attributes):
    unprivileged = []
    privileged = []
    for i in range(len(attributes["protected_attribute_names"])):
        unprivileged.append(
            {attributes["protected_attribute_names"][i]: attributes["unprivileged_protected_attributes"][i][0]
             })
        privileged.append(
            {attributes["protected_attribute_names"][i]: attributes["privileged_protected_attributes"][i][0]
             })
    return unprivileged, privileged


def get_drop_features(features, selected_features):
    drop_features = []
    for i in range(len(features)):
        if selected_features[i] == 1:
            drop_features.append(features[i])
    if len(drop_features) == 0:  # If no features in selected_features, include all features instead
        drop_features = []
    return drop_features
