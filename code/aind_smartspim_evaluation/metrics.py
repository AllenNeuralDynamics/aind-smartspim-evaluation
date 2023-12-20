from typing import List, Optional

import numpy as np
from aind_registration_evaluation import util
from aind_registration_evaluation._shared.types import ArrayLike
from aind_registration_evaluation.metric import (
    compute_feature_space_distances, get_pairs_from_distances)
from aind_registration_evaluation.sample import *
from aind_registration_evaluation.util.intersection import \
    generate_overlap_slices


def run_misalignment(
    image_1_data: ArrayLike,
    image_2_data: ArrayLike,
    transform: np.matrix,
    n_keypoints: int,
    pad_width: int,
    overlap_ratio: Optional[float] = 0.10,
    orientation: Optional[str] = "x",
    visualize: Optional[bool] = True,
    name_1: Optional[str] = "",
    name_2: Optional[str] = "",
) -> List[np.ndarray]:

    image_1_shape = image_1_data.shape
    image_2_shape = image_2_data.shape

    bounds_1, bounds_2 = util.calculate_bounds(image_1_shape, image_2_shape, transform)

    # Compute keypoints between images

    slices_1, slices_2, offset_img_1 = generate_overlap_slices(
        shapes=[image_1_shape, image_2_shape],
        orientation=orientation,
        overlap_ratio=overlap_ratio,
    )

    img_1_dict = generate_key_features_per_img2d(
        image_1_data[slices_1],
        n_keypoints=n_keypoints,
        pad_width=pad_width,
    )
    img_2_dict = generate_key_features_per_img2d(
        image_2_data[slices_2],
        n_keypoints=n_keypoints,
        pad_width=pad_width,
    )

    left_image_keypoints = img_1_dict["keypoints"]
    right_image_keypoints = img_2_dict["keypoints"]

    feature_vector_img_1 = (
        img_1_dict["features"],
        left_image_keypoints,
    )
    feature_vector_img_2 = (
        img_2_dict["features"],
        right_image_keypoints,
    )

    distances = compute_feature_space_distances(
        feature_vector_img_1, feature_vector_img_2, feature_weight=0.2
    )

    point_matches_pruned = get_pairs_from_distances(
        distances=distances, delete_points=True, metric_threshold=0.1
    )

    if not len(point_matches_pruned):
        print("No keypoints found!")
        return np.nan, np.nan

    # Tomorrow map points to the same
    # coordinate system
    # Working only with translation at the moment
    offset_ty = transform[0, -1]
    offset_tx = transform[1, -1]

    # Moving image keypoints back to intersection area
    if orientation == "y":
        left_image_keypoints[:, 0] += offset_img_1
    else:
        # x
        left_image_keypoints[:, 1] += offset_img_1

    right_image_keypoints[:, 0] += offset_ty
    right_image_keypoints[:, 1] += offset_tx

    # distance between points
    point_distances = np.array([])
    picked_left_points = []
    picked_right_points = []

    for left_idx, right_idx in point_matches_pruned.items():
        picked_left_points.append(left_image_keypoints[left_idx])
        picked_right_points.append(right_image_keypoints[right_idx])

        loc_dif = np.sqrt(
            np.sum(
                np.power(
                    left_image_keypoints[left_idx] - right_image_keypoints[right_idx],
                    2,
                ),
                axis=-1,
            )
        )
        point_distances = np.append(point_distances, loc_dif)

    picked_left_points = np.array(picked_left_points)
    picked_right_points = np.array(picked_right_points)

    median = calculate_central_value(
        point_distances, central_type="median", outlier_threshold=1
    )
    mean = calculate_central_value(
        point_distances, central_type="mean", outlier_threshold=1
    )

    threshold = 5
    point_distances_median_idx = np.where(
        (point_distances >= median - threshold)
        & (point_distances <= median + threshold)
    )
    point_distances_mean_idx = np.where(
        (point_distances >= mean - threshold) & (point_distances <= mean + threshold)
    )

    print(f"\n[!] Median euclidean distance in pixels/voxels: {median}")
    print(f"[!] Mean euclidean distance in pixels/voxels: {mean}")

    if visualize:
        util.visualize_misalignment_images(
            image_1_data,
            image_2_data,
            [bounds_1, bounds_2],
            picked_left_points[point_distances_median_idx],
            picked_right_points[point_distances_median_idx],
            transform,
            f"Misalignment metric {name_1} {name_2} - Median error {median}",
        )

        util.visualize_misalignment_images(
            image_1_data,
            image_2_data,
            [bounds_1, bounds_2],
            picked_left_points[point_distances_mean_idx],
            picked_right_points[point_distances_mean_idx],
            transform,
            f"Misalignment metric {name_1} {name_2} - Mean error {mean}",
        )

    return mean, median
