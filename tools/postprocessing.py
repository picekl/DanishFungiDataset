import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from itertools import combinations
from scipy.special import softmax
from sklearn.metrics import f1_score, accuracy_score, top_k_accuracy_score
from dataset_cls import ExtraFeaturesDataset


def post_processing_pipeline(
        df: pd.DataFrame,
        model,
        dataloader,
        device,
        target_feature: str,
        selected_features: list[str],
) -> dict[str, dict]:
    metadata_distributions = {}
    for feature in selected_features:
        metadata_distributions[feature] = get_target_to_feature_conditional_distributions(
            df,
            feature,
            target_feature,
            add_to_missing=False
        )

    target_distribution = df.groupby(target_feature).size() / len(df)

    predictions, predictions_raw, ground_truth_labels, ground_truth_features = predict_with_features(model, dataloader, device)

    feature_prior_ratios = {}
    weighted_predictions_complete = {}
    for feature in selected_features:
        metadata_distribution = metadata_distributions[feature]
        seen_feature_values = ground_truth_features[feature]

        weighted_predictions, weighted_predictions_raw, feature_prior_ratio = weight_predictions_by_feature_distribution(
            target_to_feature_conditional_distributions=metadata_distribution,
            target_distribution=target_distribution,
            ground_truth_labels=ground_truth_labels,
            raw_predictions=predictions_raw,
            ground_truth_feature_categories=seen_feature_values
        )
        weighted_predictions_complete[feature] = {
            "predictions": weighted_predictions,
            "predictions_raw": weighted_predictions_raw
        }
        feature_prior_ratios[feature] = feature_prior_ratio

    merged_predictions = post_process_prior_combinations(predictions_raw, feature_prior_ratios)
    weighted_predictions_complete.update(merged_predictions)

    for combination, _weighted_predictions in weighted_predictions_complete.items():
        _predictions = _weighted_predictions["predictions"]
        _predictions_raw = _weighted_predictions["predictions_raw"]
        print(combination, get_metrics(ground_truth_labels, _predictions, _predictions_raw))

    return weighted_predictions_complete


def get_target_to_feature_conditional_distributions(
    df: pd.DataFrame, feature: str, target_feature: str, add_to_missing: bool = True
) -> pd.Series:
    """Returns MultiIndex DataFrame of target conditional distributions per feature category.
    Returns all possible conditional distributions. Assume even distribution whole feature category is missing.
    """

    feature_per_target_counts = df.groupby([feature, target_feature])[
        feature
    ].sum().unstack(  # (feature_category, target_feature_category), count
        fill_value=0
    ).stack() + int(  # Add missing combinations
        add_to_missing
    )  # Add 1 to all counts to avoid zero division
    per_feature_counts = feature_per_target_counts.groupby(feature).sum()

    # Handle 0 counts -> Assume even distribution
    if any(per_feature_count == 0 for per_feature_count in per_feature_counts):
        zeroed_features_indexes = per_feature_counts[per_feature_counts == 0]
        for zeroed_feature in zeroed_features_indexes.index:
            feature_per_target_counts[zeroed_feature] += 1
        per_feature_counts = feature_per_target_counts.groupby(feature).sum()

    distributions = feature_per_target_counts / per_feature_counts
    assert all(
        abs(distributions.groupby(feature).sum() - 1) < 1e-6
    ), f"Target distributions do not sum to 1 per feature category of {feature}"

    return distributions


def predict_with_features(
        model,
        loader: DataLoader,
        device,
) -> tuple[list, list, list, dict]:
    """ Makes predictions on the dataloader, which must contain ExtraFeaturesDataset. Returns predictions and ground truth target labels and extra features (metadata for post-processing)"""
    assert isinstance(loader.dataset, ExtraFeaturesDataset), "Dataset in loader must be of type ExtraFeaturesDataset"
    extra_features = loader.dataset.get_extra_features_names()
    batch_size = loader.batch_size

    predictions = np.zeros(len(loader.dataset))
    ground_truth_labels = []
    predictions_raw = []

    ground_truth_features = {feature: [] for feature in extra_features}

    for i, (images, labels, paths, features) in enumerate(tqdm.tqdm(loader, total=len(loader))):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            y_preds = model(images)

        predictions[i * batch_size: (i + 1) * batch_size] = y_preds.argmax(1).to('cpu').numpy()
        ground_truth_labels.extend(labels.to('cpu').numpy())
        predictions_raw.extend(y_preds.to('cpu').numpy())

        for extra_feature in extra_features:
            ground_truth_features[extra_feature].extend(features[extra_feature])

    return predictions.tolist(), predictions_raw, ground_truth_labels, ground_truth_features


def weight_predictions_by_feature_distribution(
    target_to_feature_conditional_distributions: pd.Series,
    target_distribution: pd.Series,
    ground_truth_labels: list,
    raw_predictions: list,
    ground_truth_feature_categories: list,
) -> tuple[list, list, list]:
    """

    Args:
        target_to_feature_conditional_distributions:
        target_distribution:
        ground_truth_labels:
        raw_predictions:
        ground_truth_feature_categories:

    Returns:

    """
    weighted_predictions = []
    weighted_predictions_raw = []
    feature_prior_ratios = []

    for lbl, raw_prediction, ground_truth_feature_category in tqdm.tqdm(
        zip(ground_truth_labels, raw_predictions, ground_truth_feature_categories),
        total=len(ground_truth_labels),
    ):
        predictions = softmax(raw_prediction)
        target_to_feature_conditional_distribution = (
            target_to_feature_conditional_distributions[
                int(ground_truth_feature_category)
            ]
        )
        p_feature = (predictions * target_to_feature_conditional_distribution) / (
            sum(predictions * target_to_feature_conditional_distribution)
        )

        prior_ratio = p_feature / target_distribution
        max_index = np.argmax(prior_ratio * predictions)

        feature_prior_ratios.append(prior_ratio)
        weighted_predictions_raw.append(prior_ratio * predictions)
        weighted_predictions.append(max_index)

    return weighted_predictions, weighted_predictions_raw, feature_prior_ratios


def post_process_prior_combinations(
        raw_predictions: list,
        feature_prior_ratios: dict
):
    features = list(feature_prior_ratios.keys())
    metrics_by_combination = {}
    all_combinations_selected_features = []
    for num_features in range(2, len(features) + 1):
        all_combinations_selected_features.extend(combinations(features, num_features))

    merged_predictions = {}
    for combination in all_combinations_selected_features:
        selected_feature_prior_ratios = [feature_prior_ratios[feature] for feature in combination]

        _merged_predictions, _merged_predictions_raw = weight_predictions_combined_feature_priors(
            raw_predictions=raw_predictions,
            feature_prior_ratios=selected_feature_prior_ratios
        )
        merged_predictions[f"{'+'.join(combination)}"] = {
            "predictions": _merged_predictions,
            "predictions_raw": _merged_predictions_raw
        }

    return merged_predictions


def weight_predictions_combined_feature_priors(
    raw_predictions: list,
    feature_prior_ratios: list
) -> tuple[list, list]:
    merged_predictions = []
    merged_predictions_raw = []

    for index, raw_prediction in enumerate(raw_predictions):
        prediction = softmax(raw_prediction)

        weighted_prediction = prediction
        for feature_prior_ratio in feature_prior_ratios:
            weighted_prediction *= feature_prior_ratio[index]

        merged_prediction = weighted_prediction / (sum(weighted_prediction))
        max_index = np.argmax(merged_prediction)

        merged_predictions_raw.append(merged_prediction)
        merged_predictions.append(max_index)

    return merged_predictions, merged_predictions_raw


def get_metrics(
        ground_truth_labels: list,
        predictions: list,
        predictions_raw: list,
):
    f1 = f1_score(ground_truth_labels, predictions, average='macro')
    accuracy = accuracy_score(ground_truth_labels, predictions)
    recall_3 = top_k_accuracy_score(ground_truth_labels, predictions_raw, k=3)
    return f1, accuracy, recall_3

