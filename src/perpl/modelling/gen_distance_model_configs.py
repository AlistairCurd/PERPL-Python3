import copy
import datetime
import os
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
    # with open(os.path.join(config_folder, "config.yaml"), "r") as ymlfile:
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
    dimension = config["dimension"]
    backgrounds = config["background"]
    n_peaks = config["n_peaks"]
    peak_ampss = config["peak_amps"]
    dist_mults = config["dist_mults"]
    custom_mults_list = config["custom_mults_list"]
    repeats = config["repeats"]
    offsets = config["offset"]
    normalises = config["normalise"]
    params_initial = config["params_initial"]
    params_lower = config["params_lower"]
    params_upper = config["params_upper"]

    if isinstance(params_initial["characteristic_distance_1"], list):
        if len(params_initial["characteristic_distance_1"]) != len(dist_mults):
            raise ValueError(
                "There is a list of distances for the first peak."
                " There must be the same number of entries in this list"
                " as in charac_dist in the config file"
                " (types of relationship between characteristic distances)."
            )
        print(
            "Multiple characteristic distances present for first peak"
            ". Therefore, assuming peak distances "
            "are given respectively for each option given for charac_dist"
            " in the config file (types of relationship between"
            " characteristic distances)."
        )
        multiple_dist_mults = True
    else:
        multiple_dist_mults = False

    # generate all possible model configurations
    for index, params in enumerate(
        product(
            dimension,
            backgrounds,
            n_peaks,
            peak_ampss,
            dist_mults,
            # custom_mults_lists,
            repeats,
            offsets,
            normalises,
        )
    ):
        params_initial_copy = copy.deepcopy(params_initial)
        params_lower_copy = copy.deepcopy(params_lower)
        params_upper_copy = copy.deepcopy(params_upper)

        model_config = {
            "dimension": params[0],
            "background": params[1],
            "n_peaks": params[2],
            "peak_amps": params[3],
            "characteristic_distance": params[4],
            "characteristic_distance_ratio": custom_mults_list,
            "repeats": params[5],
            "offset": params[6],
            "normalise": params[7],
            "params_initial": params_initial_copy,
            "params_lower": params_lower_copy,
            "params_upper": params_upper_copy,
        }

        # If multiple characteristic distance types present
        # (e.g. sweeping through both models using multiples of a unit distance
        # and models using independent distances)
        if multiple_dist_mults:
            # change params_values
            for name, file in zip(
                ["params_initial", "params_lower", "params_upper"],
                [params_initial_copy, params_lower_copy, params_upper_copy],
                strict=True,
            ):
                idx = dist_mults.index(params[4])
                model_config[name]["characteristic_distance_1"] = file[
                    "characteristic_distance_1"
                ][idx]

        # save yaml file
        model_config_save_loc = os.path.join(models_folder, f"model_{index}.yaml")
        with open(model_config_save_loc, "w") as outfile:
            yaml.dump(model_config, outfile)

    return models_folder
