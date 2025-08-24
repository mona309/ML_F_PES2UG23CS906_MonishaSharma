# lab.py
import torch

def get_entropy_of_dataset(tensor: torch.Tensor):
    """
    Calculate the entropy of the entire dataset.
    Formula: Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i

    Args:
        tensor (torch.Tensor): Input dataset as a tensor, where the last column is the target.

    Returns:
        float: Entropy of the dataset.
    """
    target_col = tensor[:, -1]   # last column is target
    classes, counts = torch.unique(target_col, return_counts=True)
    probs = counts.float() / target_col.shape[0]

    entropy = -torch.sum(probs * torch.log2(probs))
    return float(entropy)


def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
    """
    Calculate the average information (weighted entropy) of an attribute.
    Formula: Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) where S_v is subset with attribute value v.

    Args:
        tensor (torch.Tensor): Input dataset as a tensor.
        attribute (int): Index of the attribute column.

    Returns:
        float: Average information of the attribute.
    """
    total_len = tensor.shape[0]
    values, counts = torch.unique(tensor[:, attribute], return_counts=True)

    avg_info = 0.0
    for v, count in zip(values, counts):
        subset = tensor[tensor[:, attribute] == v]
        subset_entropy = get_entropy_of_dataset(subset)
        avg_info += (count.item() / total_len) * subset_entropy

    return float(avg_info)


def get_information_gain(tensor: torch.Tensor, attribute: int):
    """
    Calculate Information Gain for an attribute.
    Formula: Information_Gain = Entropy(S) - Avg_Info(attribute)

    Args:
        tensor (torch.Tensor): Input dataset as a tensor.
        attribute (int): Index of the attribute column.

    Returns:
        float: Information gain for the attribute (rounded to 4 decimals).
    """
    dataset_entropy = get_entropy_of_dataset(tensor)
    avg_info = get_avg_info_of_attribute(tensor, attribute)

    info_gain = dataset_entropy - avg_info
    return round(float(info_gain), 4)


def get_selected_attribute(tensor: torch.Tensor):
    """
    Select the best attribute based on highest information gain.

    Returns a tuple with:
    1. Dictionary mapping attribute indices to their information gains
    2. Index of the attribute with highest information gain
    
    Example: ({0: 0.123, 1: 0.768, 2: 1.23}, 2)

    Args:
        tensor (torch.Tensor): Input dataset as a tensor.

    Returns:
        tuple: (dict of attribute:index -> information gain, index of best attribute)
    """
    n_attributes = tensor.shape[1] - 1  # exclude target
    info_gains = {}

    for attr in range(n_attributes):
        info_gains[attr] = get_information_gain(tensor, attr)

    # find attribute with max info gain
    best_attr = max(info_gains, key=info_gains.get)
    return (info_gains, best_attr)
