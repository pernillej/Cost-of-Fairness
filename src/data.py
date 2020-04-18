from aif360.datasets import GermanDataset, CompasDataset


def load_german_dataset():
    """
    Collect the aif360 preprocessed German Credit Data Set.
    Assigns 'age' as the protected attribute with age >= 25 considered privileged.
    Sex-related attributes are removed (the other option for privileged attribute)

    :return: The German Credit Data Set
    """
    dataset = GermanDataset(
        protected_attribute_names=['age'],
        privileged_classes=[lambda x: x >= 25],
        features_to_drop=['personal_status', 'sex']
    )
    return dataset


def load_compas_dataset():
    """
    Collect the aif360 preprocessed Compas Data Set.
    Assigns 'race' as the protected attribute with Caucasian considered privileged.
    Sex-related attributes are removed (the other option for privileged attribute)

    :return: The Compas Dataset
    """
    dataset = CompasDataset(
        protected_attribute_names=['race'],
        privileged_classes=[['Caucasian']],
        features_to_drop=['sex']
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
            "unprivileged_protected_attributes": self.unprivileged_protected_attributes,
            "favorable_label": The favorable label,
            "unfavorable_label": The unfavorable label
          }
    """
    df, attributes = dataset.convert_to_dataframe()

    attributes["favorable_label"] = favorable_label
    attributes["unfavorable_label"] = unfavorable_label

    return df, attributes
