"""
Evaluate SmartSPIM datasets
"""
import logging
import multiprocessing
import os
import time
from pathlib import Path
from typing import Dict, Hashable, List, Optional, Sequence, Tuple, Union

import dask
import numpy as np
from aind_registration_evaluation.main_qa import (EvalStitching,
                                                  get_default_config)
from dask_image.imread import imread as daimread
from natsort import natsorted
from numcodecs import blosc

from ._shared.types import ArrayLike, PathLike
from .utils import utils


def add_leading_dim(data: ArrayLike) -> ArrayLike:
    """
    Adds a leading dimension

    Parameters
    ------------------------

    data: ArrayLike
        Input array that will have the
        leading dimension

    Returns
    ------------------------

    ArrayLike:
        Array with the new dimension in front.
    """
    return data[None, ...]


def pad_array_n_d(arr: ArrayLike, dim: int = 5) -> ArrayLike:
    """
    Pads a daks array to be in a 5D shape.

    Parameters
    ------------------------

    arr: ArrayLike
        Dask/numpy array that contains image data.
    dim: int
        Number of dimensions that the array will be padded

    Returns
    ------------------------
    ArrayLike:
        Padded dask/numpy array.
    """
    if dim > 5:
        raise ValueError("Padding more than 5 dimensions is not supported.")

    while arr.ndim < dim:
        arr = arr[np.newaxis, ...]
    return arr


def lazy_png_reader(filename: PathLike) -> ArrayLike:
    """
    Lazy PNG image reader

    Parameters
    ----------
    filename: PathLike
        Path where the image file is stored

    Returns
    -------
    ArrayLike
        Lazy array pointing to the data
    """
    return daimread(filename, arraytype="numpy")


def get_image_transformation(transform_dict: dict) -> np.matrix:
    """
    Get image transformation matrix based
    on the transformation dictionary

    Parameters
    ----------
    transform_dict: dict
        Dictionary that contains the numerical
        values for the transformations

    Returns
    -------
    np.matrix
        Registration matrix
    """
    identity_matrix = np.identity(4, dtype=np.int16)
    output_transform = {}
    for ori, transform in transform_dict.items():
        ori_transform = None
        if len(transform):
            ori_transform = identity_matrix.copy()
            ori_transform[-2, -1] = float(transform["X"])
            ori_transform[-3, -1] = float(transform["Y"])
            ori_transform[-4, -1] = float(transform["Z"])

        output_transform[ori] = ori_transform

    return output_transform


def get_stack_orientation(
    cols: List[str], rows: List[str], eval_col: str, eval_row: str
) -> dict:
    """
    Gets the neighbouring stacks of the evaluated stack
    based on its own orientation. These orientations
    are North, South, West and East.

    Parameters
    ----------
    cols: List[str]
        List of all columns in the dataset

    rows: List[str]
        List of all rows in the dataset

    eval_col: str
        Column to be evaluated

    eval_row: str
        Row to be evaluated

    Returns
    -------
    dict
        Dictionary with the neighbouring stacks
        organized per orientation, North, South
        West and East with respect of the evaluated
        stack
    """
    cols = natsorted(cols)
    rows = natsorted(rows)

    max_cols = len(cols)
    max_rows = len(rows)

    curr_col_index = cols.index(eval_col)
    curr_row_index = rows.index(eval_row)

    # Noth displacement
    north_dis = curr_row_index - 1

    # South displacement
    south_dis = curr_row_index + 1

    # West displacement
    west_dis = curr_col_index - 1

    # East displacement
    east_dis = curr_col_index + 1

    return {
        # Y axis
        "north": f"{eval_col}/{eval_col}_{rows[north_dis]}" if north_dis >= 0 else None,
        "south": f"{eval_col}/{eval_col}_{rows[south_dis]}"
        if south_dis < max_rows
        else None,
        # X axis
        "west": f"{cols[west_dis]}/{cols[west_dis]}_{eval_row}"
        if west_dis >= 0
        else None,
        "east": f"{cols[east_dis]}/{cols[east_dis]}_{eval_row}"
        if east_dis < max_rows
        else None,
    }


def get_stack_transforms(
    channel_transformations: dict,
    cols: List[str],
    rows: List[str],
    eval_col: str,
    eval_row: str,
) -> Tuple:
    """
    Gets the neighbouring stacks of the evaluated stack
    based on its own orientation and transformations.
    These orientations are North, South, West and East.

    Parameters
    ----------
    channel_transformations: dict
        Dictionary with stitching transformations between
        pairs of image stacks

    cols: List[str]
        List of all columns in the dataset

    rows: List[str]
        List of all rows in the dataset

    eval_col: str
        Column to be evaluated

    eval_row: str
        Row to be evaluated

    Returns
    -------
    Tuple[dict, str, dict]
        - Dict: Dictionary with the neighbouring stacks
        organized that contains information about
        the orientation (e.g., North, South West
        and East), transformation matrix and the
        original transformation (dictionary).
        - str: Relative path of the evaluated stack
        - dict: Stack orientation correspondence in
        North, South, West and East of the neighbouring
        stacks of the evaluated one
    """
    stack_eval = f"{eval_col}/{eval_col}_{eval_row}"

    stack_orientation_correspondence = get_stack_orientation(
        cols, rows, eval_col, eval_row
    )
    ori_transforms = channel_transformations["stacks_displacements"][stack_eval]
    transform_matrix = get_image_transformation(
        channel_transformations["stacks_displacements"][stack_eval]
    )

    # Mapping orientation transforms to stack transforms
    stack_transforms = {}
    for ori, stack in stack_orientation_correspondence.items():
        if stack:
            stack_transforms[stack] = {
                "ori": ori,
                "transform": transform_matrix[ori],
                "original": ori_transforms[ori],
            }

    return stack_transforms, stack_eval, stack_orientation_correspondence


def get_sample_img_png(directory_structure: dict, channel_dir: str) -> ArrayLike:
    """
    Gets the sample image for the dataset

    Parameters
    ---------------

    directory_structure: dict
        Whole brain volume directory structure

    Returns
    ---------------

    ArrayLike
        Array with the sample image
    """
    sample_img = None
    channel_dir = Path(channel_dir)
    for col_name, rows in directory_structure.items():
        for row_name, images in rows.items():
            sample_path = (
                channel_dir.joinpath(col_name).joinpath(row_name).joinpath(images[0])
            )

            if not isinstance(sample_img, dask.array.core.Array):
                sample_img = lazy_png_reader(str(sample_path))
            else:
                sample_img_2 = lazy_png_reader(str(sample_path))

                if sample_img.chunksize != sample_img_2.chunksize:
                    print("Changes ", sample_img, sample_img_2)
                    return sample_img

    return sample_img


def evaluate_stitching(
    channel_transformations: dict, channel_data_path: PathLike, logger: logging.Logger
):
    """
    Evaluates a SmartSPIM dataset channel

    Parameters
    ----------
    channel_transformations: dict
        Dictionary with stitching transformations between
        pairs of image stacks

    channel_data_path: PathLike
        Path where the dataset channel data is stored

    logger: logging.Logger
        Logging object
    """

    # Parameters to evaluate the SmartSPIM datasets
    overlap_ratio = 0.1
    n_keypoints = 200
    pad_width = 20
    orientation_map = {"west": "x", "east": "x", "north": "y", "south": "y"}

    # Get directory structure of the MIPs
    logger.info(f"Computing stitching evaluation for data in: {channel_data_path}")

    start_time = time.time()

    channel_folder_structure = utils.read_image_directory_structure(channel_data_path)

    channel_path = list(channel_folder_structure.keys())[0]
    cols = list(channel_folder_structure[channel_path].keys())
    rows = [row for row in list(channel_folder_structure[channel_path][cols[0]].keys())]
    n_cols = len(cols)
    n_rows = len(rows)
    len_stack = len(channel_folder_structure[channel_path][cols[0]][rows[0]])

    stacks_mips = {}
    n_stacks = 0
    for col, row in channel_folder_structure.items():
        for row, images in row.items():
            row_col = f"{col}/{row}"
            path = Path(channel_data_path).joinpath(row_col)
            print(f"[{n_stacks}] Reading stacks in {path}")
            stacks_mips[row_col] = daimread(f"{str(path)}/*.tiff")
            n_stacks += 1

    # Defining stack to evaluate
    mip_col_eval = "474410"
    mip_row_eval = "259920"
    mip_stack_eval = f"{mip_col_eval}/{mip_col_eval}_{mip_row_eval}"

    stack_transforms, stack_name, stack_orientations = get_stack_transforms(
        cols, rows, mip_col_eval, mip_row_eval
    )
    keypoint_metric = {}

    for close_stack, values in stack_transforms.items():
        xyz_ori = orientation_map[values["ori"]]
        transform = np.abs(values["transform"][1:, 1:])
        logger.info(f"Evaluating {stack_name} against {close_stack}")

        for curr_slide_idx in range(stacks_mips[stack_name].shape[-3]):
            logger.info(
                f"[{curr_slide_idx}] Orig stack: {stack_name} Dest stack: {close_stack} - orientation: {xyz_ori} ori: {values['ori']}"
            )
            img_1 = None
            img_2 = None

            if values["ori"] in ["north", "west"]:
                img_1 = stacks_mips[close_stack][curr_slide_idx, :, :].compute()
                img_2 = stacks_mips[stack_name][curr_slide_idx, :, :].compute()

            else:
                # south east
                img_1 = stacks_mips[stack_name][curr_slide_idx, :, :].compute()
                img_2 = stacks_mips[close_stack][curr_slide_idx, :, :].compute()

            mean, median = run_misalignment(
                image_1_data=img_1,
                image_2_data=img_2,
                transform=transform,
                n_keypoints=n_keypoints,
                pad_width=pad_width,
                overlap_ratio=overlap_ratio,
                orientation=xyz_ori,
                visualize=False,
            )

            if close_stack not in keypoint_metric:
                keypoint_metric[close_stack] = []

            keypoint_metric[close_stack].append({"mean": mean, "median": median})

    print(keypoint_metric)

    end_time = time.time()


def main():
    """
    Main evaluation function
    """
    pass


if __name__ == "__main__":
    main()
