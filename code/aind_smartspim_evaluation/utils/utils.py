"""
Utility functions
"""

import json
import logging
import multiprocessing
import os
import platform
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import psutil
import xmltodict
from natsort import natsorted

# IO types
PathLike = Union[str, Path]


def save_dict_as_json(
    filename: str,
    dictionary: dict,
) -> None:
    """
    Saves a dictionary as a json file.

    Parameters
    ------------------------
    filename: str
        Name of the json file.
    dictionary: dict
        Dictionary that will be saved as json.
    verbose: Optional[bool]
        True if you want to print the path where the file was saved.

    """

    if dictionary is None:
        dictionary = {}

    with open(filename, "w") as json_file:
        json.dump(dictionary, json_file, indent=4)


def create_folder(dest_dir: PathLike, verbose: Optional[bool] = False) -> None:
    """
    Create new folders.

    Parameters
    ------------------------

    dest_dir: PathLike
        Path where the folder will be created if it does not exist.

    verbose: Optional[bool]
        If we want to show information about the folder status. Default False.

    Raises
    ------------------------

    OSError:
        if the folder exists.

    """

    if not (os.path.exists(dest_dir)):
        try:
            if verbose:
                print(f"Creating new directory: {dest_dir}")
            os.makedirs(dest_dir)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise


def read_json_as_dict(filepath: str) -> dict:
    """
    Reads a json as dictionary.

    Parameters
    ------------------------

    filepath: PathLike
        Path where the json is located.

    Returns
    ------------------------

    dict:
        Dictionary with the data the json has.

    """

    dictionary = {}

    if os.path.exists(filepath):
        with open(filepath) as json_file:
            dictionary = json.load(json_file)

    return dictionary


def create_logger(output_log_path: PathLike) -> logging.Logger:
    """
    Creates a logger that generates
    output logs to a specific path.

    Parameters
    ------------
    output_log_path: PathLike
        Path where the log is going
        to be stored

    Returns
    -----------
    logging.Logger
        Created logger pointing to
        the file path.
    """
    CURR_DATE_TIME = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    LOGS_FILE = f"{output_log_path}/fusion_log_{CURR_DATE_TIME}.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s : %(message)s",
        datefmt="%Y-%m-%d %H:%M",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOGS_FILE, "a"),
        ],
        force=True,
    )

    logging.disable("DEBUG")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    return logger


def profile_resources(
    time_points: List,
    cpu_percentages: List,
    memory_usages: List,
    monitoring_interval: int,
):
    """
    Profiles compute resources usage.

    Parameters
    ----------
    time_points: List
        List to save all the time points
        collected

    cpu_percentages: List
        List to save the cpu percentages
        during the execution

    memory_usage: List
        List to save the memory usage
        percentages during the execution

    monitoring_interval: int
        Monitoring interval in seconds
    """
    start_time = time.time()

    while True:
        current_time = time.time() - start_time
        time_points.append(current_time)

        # CPU Usage
        cpu_percent = psutil.cpu_percent(interval=monitoring_interval)
        cpu_percentages.append(cpu_percent)

        # Memory usage
        memory_info = psutil.virtual_memory()
        memory_usages.append(memory_info.percent)

        time.sleep(monitoring_interval)


def generate_resources_graphs(
    time_points: List,
    cpu_percentages: List,
    memory_usages: List,
    output_path: str,
    prefix: str,
):
    """
    Profiles compute resources usage.

    Parameters
    ----------
    time_points: List
        List to save all the time points
        collected

    cpu_percentages: List
        List to save the cpu percentages
        during the execution

    memory_usage: List
        List to save the memory usage
        percentages during the execution

    output_path: str
        Path where the image will be saved

    prefix: str
        Prefix name for the image
    """
    time_len = len(time_points)
    memory_len = len(memory_usages)
    cpu_len = len(cpu_percentages)

    min_len = min([time_len, memory_len, cpu_len])
    if not min_len:
        return

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(time_points[:min_len], cpu_percentages[:min_len], label="CPU Usage")
    plt.xlabel("Time (s)")
    plt.ylabel("CPU Usage (%)")
    plt.title("CPU Usage Over Time")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time_points[:min_len], memory_usages[:min_len], label="Memory Usage")
    plt.xlabel("Time (s)")
    plt.ylabel("Memory Usage (%)")
    plt.title("Memory Usage Over Time")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{output_path}/{prefix}_compute_resources.png", bbox_inches="tight")


def stop_child_process(process: multiprocessing.Process):
    """
    Stops a process

    Parameters
    ----------
    process: multiprocessing.Process
        Process to stop
    """
    process.terminate()
    process.join()


def get_size(bytes, suffix: str = "B") -> str:
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'

    Parameters
    ----------
    bytes: bytes
        Bytes to scale

    suffix: str
        Suffix used for the conversion
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor


def get_code_ocean_cpu_limit():
    """
    Gets the Code Ocean capsule CPU limit

    Returns
    -------
    int:
        number of cores available for compute
    """
    # Checks for environmental variables
    co_cpus = os.environ.get("CO_CPUS")
    aws_batch_job_id = os.environ.get("AWS_BATCH_JOB_ID")

    if co_cpus:
        return co_cpus
    if aws_batch_job_id:
        return 1
    with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fp:
        cfs_quota_us = int(fp.read())
    with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
        cfs_period_us = int(fp.read())
    container_cpus = cfs_quota_us // cfs_period_us
    # For physical machine, the `cfs_quota_us` could be '-1'
    return psutil.cpu_count(logical=False) if container_cpus < 1 else container_cpus


def print_system_information(logger: logging.Logger):
    """
    Prints system information

    Parameters
    ----------
    logger: logging.Logger
        Logger object
    """
    co_memory = int(os.environ.get("CO_MEMORY"))
    # System info
    sep = "=" * 40
    logger.info(f"{sep} Code Ocean Information {sep}")
    logger.info(f"Code Ocean assigned cores: {get_code_ocean_cpu_limit()}")
    logger.info(f"Code Ocean assigned memory: {get_size(co_memory)}")
    logger.info(f"Computation ID: {os.environ.get('CO_COMPUTATION_ID')}")
    logger.info(f"Capsule ID: {os.environ.get('CO_CAPSULE_ID')}")
    logger.info(f"Is pipeline execution?: {bool(os.environ.get('AWS_BATCH_JOB_ID'))}")

    logger.info(f"{sep} System Information {sep}")
    uname = platform.uname()
    logger.info(f"System: {uname.system}")
    logger.info(f"Node Name: {uname.node}")
    logger.info(f"Release: {uname.release}")
    logger.info(f"Version: {uname.version}")
    logger.info(f"Machine: {uname.machine}")
    logger.info(f"Processor: {uname.processor}")

    # Boot info
    logger.info(f"{sep} Boot Time {sep}")
    boot_time_timestamp = psutil.boot_time()
    bt = datetime.fromtimestamp(boot_time_timestamp)
    logger.info(
        f"Boot Time: {bt.year}/{bt.month}/{bt.day} {bt.hour}:{bt.minute}:{bt.second}"
    )

    # CPU info
    logger.info(f"{sep} CPU Info {sep}")
    # number of cores
    logger.info(f"Physical node cores: {psutil.cpu_count(logical=False)}")
    logger.info(f"Total node cores: {psutil.cpu_count(logical=True)}")

    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    logger.info(f"Max Frequency: {cpufreq.max:.2f}Mhz")
    logger.info(f"Min Frequency: {cpufreq.min:.2f}Mhz")
    logger.info(f"Current Frequency: {cpufreq.current:.2f}Mhz")

    # CPU usage
    logger.info("CPU Usage Per Core before processing:")
    for i, percentage in enumerate(psutil.cpu_percent(percpu=True, interval=1)):
        logger.info(f"Core {i}: {percentage}%")
    logger.info(f"Total CPU Usage: {psutil.cpu_percent()}%")

    # Memory info
    logger.info(f"{sep} Memory Information {sep}")
    # get the memory details
    svmem = psutil.virtual_memory()
    logger.info(f"Total: {get_size(svmem.total)}")
    logger.info(f"Available: {get_size(svmem.available)}")
    logger.info(f"Used: {get_size(svmem.used)}")
    logger.info(f"Percentage: {svmem.percent}%")
    logger.info(f"{sep} Memory - SWAP {sep}")
    # get the swap memory details (if exists)
    swap = psutil.swap_memory()
    logger.info(f"Total: {get_size(swap.total)}")
    logger.info(f"Free: {get_size(swap.free)}")
    logger.info(f"Used: {get_size(swap.used)}")
    logger.info(f"Percentage: {swap.percent}%")

    # Network information
    logger.info(f"{sep} Network Information {sep}")
    # get all network interfaces (virtual and physical)
    if_addrs = psutil.net_if_addrs()
    for interface_name, interface_addresses in if_addrs.items():
        for address in interface_addresses:
            logger.info(f"=== Interface: {interface_name} ===")
            if str(address.family) == "AddressFamily.AF_INET":
                logger.info(f"  IP Address: {address.address}")
                logger.info(f"  Netmask: {address.netmask}")
                logger.info(f"  Broadcast IP: {address.broadcast}")
            elif str(address.family) == "AddressFamily.AF_PACKET":
                logger.info(f"  MAC Address: {address.address}")
                logger.info(f"  Netmask: {address.netmask}")
                logger.info(f"  Broadcast MAC: {address.broadcast}")
    # get IO statistics since boot
    net_io = psutil.net_io_counters()
    logger.info(f"Total Bytes Sent: {get_size(net_io.bytes_sent)}")
    logger.info(f"Total Bytes Received: {get_size(net_io.bytes_recv)}")


def read_image_directory_structure(folder_dir) -> dict:
    """
    Creates a dictionary representation of all the images
    saved by folder/col_N/row_N/images_N.[file_extention]

    Parameters
    ------------------------
    folder_dir:PathLike
        Path to the folder where the images are stored

    Returns
    ------------------------
    dict:
        Dictionary with the image representation where:
        {channel_1: ... {channel_n: {col_1: ... col_n: {row_1: ... row_n: [image_0, ..., image_n]} } } }
    """

    directory_structure = {}
    folder_dir = Path(folder_dir)

    channel_paths = natsorted(
        [
            folder_dir.joinpath(folder)
            for folder in os.listdir(folder_dir)
            if os.path.isdir(folder_dir.joinpath(folder))
        ]
    )

    cols = natsorted(os.listdir(channel_paths[0]))
    column_example = channel_paths[0].joinpath(cols[0])
    rows = natsorted(os.listdir(column_example))
    images = natsorted(os.listdir(column_example.joinpath(rows[0])))

    for channel_idx in range(len(channel_paths)):
        directory_structure[channel_paths[channel_idx]] = {}

        for col in cols:
            possible_col = channel_paths[channel_idx].joinpath(col)

            if os.path.isdir(possible_col):
                directory_structure[channel_paths[channel_idx]][col] = {}

                for row in rows:
                    possible_row = (
                        channel_paths[channel_idx].joinpath(col).joinpath(row)
                    )

                    if os.path.isdir(possible_row):
                        directory_structure[channel_paths[channel_idx]][col][
                            row
                        ] = images

    return directory_structure


class TeraStitcherXMLParser:
    """
    Class to parse from XML to JSON format
    """

    def __init__(self) -> None:
        """
        Class constructor
        """

        # terastitcher reference order
        self.terastitcher_reference_order = {
            1: ["Y", "V"],
            2: ["X", "H"],
            3: ["D", "Z"],
        }

        # Terastitcher string reference relation
        self.terastitcher_str_reference = {
            "H": ["X", 1],
            "V": ["Y", 2],
            "D": ["Z", 3],
        }

    @staticmethod
    def parse_xml(xml_path: PathLike, encoding: str = "utf-8") -> dict:
        """
        Static method to parse XML to dictionary

        Parameters
        --------------
        xml_path: PathLike
            Path where the XML is stored

        encoding: str
            XML encoding system. Default: utf-8

        Returns
        --------------
        Dict
            Dictionary with the parsed XML
        """
        with open(xml_path, "r", encoding=encoding) as xml_reader:
            xml_file = xml_reader.read()

        xml_dict = xmltodict.parse(xml_file)

        return xml_dict

    def __map_terastitcher_volume_info(self, teras_dict: dict) -> Tuple[Dict]:
        """
        Maps the terastitcher image volume information

        Parameters
        --------------
        teras_dict: dict
            Dictionary with the information extracted
            from the XML

        Returns
        --------------
        dict
            Dictionary with the volume information
        """
        # Mapping reference system
        refs = {}
        for ref, val in teras_dict["TeraStitcher"]["ref_sys"].items():
            val = int(val)
            negative = ""
            if val < 0:
                negative = "-"

            refs[
                ref.replace("@", "")
            ] = f"{negative}{self.terastitcher_reference_order[abs(val)][0]}"

        # Mapping voxel dims
        voxels = {}
        voxels["unit"] = "microns"
        for ref, val in teras_dict["TeraStitcher"]["voxel_dims"].items():
            ref = ref.replace("@", "")
            voxels[self.terastitcher_str_reference[ref][0]] = float(val)

        # Mapping volume origin for computations
        origin = {}
        origin["unit"] = "milimeters"
        for ref, val in teras_dict["TeraStitcher"]["origin"].items():
            ref = ref.replace("@", "")
            origin[self.terastitcher_str_reference[ref][0]] = float(val)

        return refs, voxels, origin

    def __map_terastitcher_stacks_displacements(self, teras_dict: dict) -> dict:
        """
        Map the terastitcher stacks displacements

        Parameters
        --------------
        teras_dict: dict
            Dictionary with the information extracted
            from the XML

        Returns
        --------------
        dict
            Dictionary with the displacements per
            stacks of tiles
        """

        def map_displ(displacement: dict) -> dict:
            """
            Helper function to map displacements in each
            direction

            Parameters
            --------------
            displacement: dict
                Dictionary with the displacements in each
                direction. (e.g. NORTH, SOUTH, WEST, EAST)

            Returns
            --------------
            dict
                Dictionary with the parsed displacements
                per stack of tiles
            """
            data = {}

            if displacement is not None:
                for key, val in displacement["Displacement"].items():
                    if key in ["V", "H", "D"]:
                        axis = self.terastitcher_str_reference[key][0]
                        data[axis] = val["@displ"]

            return data

        stack_displacements = teras_dict["TeraStitcher"]["STACKS"]["Stack"]
        stacks = {}

        for idx in range(len(stack_displacements)):
            # print(stack_displacements[idx])
            if stack_displacements[idx]["@STITCHABLE"] == "yes":
                dir_name = stack_displacements[idx]["@DIR_NAME"]
                stacks[dir_name] = {
                    "north": map_displ(stack_displacements[idx]["NORTH_displacements"]),
                    "east": map_displ(stack_displacements[idx]["EAST_displacements"]),
                    "south": map_displ(stack_displacements[idx]["SOUTH_displacements"]),
                    "west": map_displ(stack_displacements[idx]["WEST_displacements"]),
                }

        return stacks

    def parse_terastitcher_xml(
        self, xml_path: PathLike, encoding: str = "utf-8"
    ) -> dict:
        """
        Parses the terastitcher XML file

        Parameters
        --------------
        teras_dict: dict
            Dictionary with the information extracted
            from the XML

        Returns
        --------------
        Dictionary with the parsed terastitcher xml
        """
        teras_dict = TeraStitcherXMLParser.parse_xml(xml_path, encoding)

        stitch_dict = {}

        stitch_dict["dataset_path"] = teras_dict["TeraStitcher"]["stacks_dir"]["@value"]
        refs, voxels, origin = self.__map_terastitcher_volume_info(teras_dict)

        # Getting important info to identify dataset
        stitch_dict["reference_axis"] = refs
        stitch_dict["voxels_size_"] = voxels
        stitch_dict["origin"] = origin

        stacks_displacement = self.__map_terastitcher_stacks_displacements(teras_dict)
        stitch_dict["stacks_displacements"] = stacks_displacement

        return stitch_dict
