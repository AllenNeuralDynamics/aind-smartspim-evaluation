"""
Evaluates SmartSPIM datasets in terms of
image quality and registration
"""

import multiprocessing
import os
from glob import glob

from aind_smartspim_evaluation.utils import utils


def run():
    """
    Run method to evaluate SmartSPIM dataset channels
    """
    data_folder = os.path.abspath("../data")
    results_folder = os.path.abspath("../results")

    stitch_channel_transforms = glob(f"{data_folder}/xml_merging_*.xml")

    if not len(stitch_channel_transforms):
        raise ValueError("There are no merging xmls in the data folder")

    logger = utils.create_logger(output_log_path=results_folder)
    utils.print_system_information(logger)

    # Tracking compute resources
    # Subprocess to track used resources
    manager = multiprocessing.Manager()
    time_points = manager.list()
    cpu_percentages = manager.list()
    memory_usages = manager.list()

    profile_process = multiprocessing.Process(
        target=utils.profile_resources,
        args=(
            time_points,
            cpu_percentages,
            memory_usages,
            20,
        ),
    )
    profile_process.daemon = True
    profile_process.start()

    # Main task
    for stitch_channel_transform in stitch_channel_transforms:
        channel_transformations = utils.TeraStitcherXMLParser().parse_terastitcher_xml(
            xml_path=stitch_channel_transform
        )

    # Getting tracked resources and plotting image
    utils.stop_child_process(profile_process)

    if len(time_points):
        utils.generate_resources_graphs(
            time_points,
            cpu_percentages,
            memory_usages,
            results_folder,
            "smartspim_evaluation",
        )


if __name__ == "__main__":
    run()
