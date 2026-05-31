import copy
import datetime
import os
import sys
from itertools import product

import yaml


def gen_configs(config_file, suffix=None):
    """Generate models to fit.

    Args:
        config_file (str):
            Path to config file for generating models to sweep through.
        suffix (str):
            Text to add to label models folder and copy of config file.

    Returns:
        models_folder (str):
            Path to the folder containing models as determined
            by the config file.

    Saves model configurations and a copy of the config fileas used.
    """

    # load in configuration
    with open(config_file, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # Use datestamp as suffix if not given
    if suffix is None:
        suffix = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Set output locations
    config_folder, _ = os.path.split(config_file)
    models_folder = os.path.join(config_folder, f"model_configs_{suffix}")
    os.mkdir(models_folder)

    # Save the current version of the config file
    parent, _ = os.path.split(config_file)
    config_copy_path = os.path.join(parent, f"models_config_{suffix}.yaml")
    with open(config_copy_path, "w") as outfile:
        yaml.dump(config, outfile)

    # Load in params list
    model_direction = config["model_direction"]
    if model_direction == "xyz":
        dimension = 3
    elif model_direction in ("xy", "xz", "yz"):
        dimension = 2
    elif model_direction in ("xx", "yy", "zz"):
        dimension = 1
    else:
        print(f"model_direction in config file ({model_direction}) is invalid.")
        sys.exit(1)
    backgrounds = config["background"]
    n_peaks = config["n_peaks"]
    peak_amps = config["peak_amps"]
    dist_ratios = config["dist_ratios"]
    custom_ratios_list = config["custom_ratios_list"]
    repeats = config["repeats"]
    offsets = config["offset"]
    normalises = config["normalise"]
    params_initial = config["params_initial"]
    params_lower = config["params_lower"]
    params_upper = config["params_upper"]

    if isinstance(params_initial["characteristic_distance_1"], list):
        if len(params_initial["characteristic_distance_1"]) != len(dist_ratios):
            raise ValueError(
                "There is a list of distances for the first peak."
                " There must be the same number of entries in this list"
                " as in dist_ratios in the config file"
                " (types of relationship between characteristic distances)."
            )
        print(
            "Multiple characteristic distances present for first peak"
            ". Therefore, assuming peak distances "
            "are given respectively for each option given for dist_ratios"
            " in the config file (types of relationship between"
            " characteristic distances)."
        )
        multiple_dist_ratios = True
    else:
        multiple_dist_ratios = False

    # generate all possible model configurations
    for index, params in enumerate(
        product(
            backgrounds,
            n_peaks,
            peak_amps,
            dist_ratios,
            # custom_ratios_lists,
            repeats,
            offsets,
            normalises,
        )
    ):
        params_initial_copy = copy.deepcopy(params_initial)
        params_lower_copy = copy.deepcopy(params_lower)
        params_upper_copy = copy.deepcopy(params_upper)

        model_config = {
            "dimension": dimension,
            "background": params[0],
            "n_peaks": params[1],
            "peak_amps": params[2],
            "dist_ratios": params[3],
            # "custom_ratios_list": custom_ratios_list,
            "repeats": params[4],
            "offset": params[5],
            "normalise": params[6],
            "params_initial": params_initial_copy,
            "params_lower": params_lower_copy,
            "params_upper": params_upper_copy,
        }

        if params[3] == "custom_ratios":
            model_config["custom_ratios_list"] = custom_ratios_list
        else:
            model_config["custom_ratios_list"] = None

        # If multiple characteristic distance types present
        # (e.g. sweeping through both models using multiples of a unit distance
        # and models using independent distances)
        if multiple_dist_ratios:
            # change params_values
            for name, file in zip(
                ["params_initial", "params_lower", "params_upper"],
                [params_initial_copy, params_lower_copy, params_upper_copy],
                strict=True,
            ):
                idx = dist_ratios.index(params[3])
                model_config[name]["characteristic_distance_1"] = file[
                    "characteristic_distance_1"
                ][idx]

        # save yaml file
        model_config_save_loc = os.path.join(models_folder, f"model_{index}.yaml")
        with open(model_config_save_loc, "w") as outfile:
            yaml.dump(model_config, outfile)

    return models_folder
