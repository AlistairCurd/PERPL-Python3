import argparse
import copy
from itertools import product
import os
import warnings

import yaml


def gen_configs(config_file):
    """Generate models to fit.

    Args:
        config_file (str):
            Path to config file for generating models to sweep through.
    
    Returns:
        Nothing
    
    Saves model configurations in parent folder of the config file.
    """

    # load in configuration
    # with open(os.path.join(config_folder, "config.yaml"), "r") as ymlfile:
    with open(config_file, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    
    # Set output locations
    config_folder, _ = os.path.split(config_file)
    models_folder = os.path.join(config_folder, "models")
    os.mkdir(models_folder)

    ## load in params for particular direction
    # direction_params = config[direction]  # Was axial/transverse

    # load in params list
    dimension = config["dimension"]
    backgrounds = config["background"]
    n_peaks = config["n_peaks"]
    peak_types = config["peak_type"]
    charac_dists = config["charac_dist"]
    charac_dist_ratio = config["charac_dist_ratio"]
    repeats = config["repeats"]
    offsets = config["offset"]
    normalises = config["normalise"]
    params_initial = config["params_initial"]
    params_lower = config["params_lower"]
    params_upper = config["params_upper"]

    if type(params_initial["characteristic_distance_1"]) is list:
        assert len(params_initial["characteristic_distance_1"]) == len(charac_dists)
        warnings.warn(
            "Multiple characteristic distances for first peak "
            ". Therefore, assuming peak distances "
            "are for each type of characteristic distance."
        )
        multiple_charac_dists = True
    else:
        multiple_charac_dists = False

    # generate all possible model configurations
    for index, params in enumerate(
        product(
            dimension,
            backgrounds,
            n_peaks,
            peak_types,
            charac_dists,
            # charac_dist_ratios,
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
            "peak_type": params[3],
            "characteristic_distance": params[4],
            "characteristic_distance_ratio": charac_dist_ratio,
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
        if multiple_charac_dists:
            # change params_values
            for name, file in zip(
                ["params_initial", "params_lower", "params_upper"],
                [params_initial_copy, params_lower_copy, params_upper_copy],
            ):
                idx = charac_dists.index(params[4])
                model_config[name]["characteristic_distance_1"] = file[
                    "characteristic_distance_1"
                ][idx]

        # save yaml file
        model_config_save_loc = os.path.join(
            models_folder, f"model_{index}.yaml"
        )
        with open(model_config_save_loc, "w") as outfile:
            yaml.dump(model_config, outfile)


def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with"""

    # parse arguments
    parser = argparse.ArgumentParser(
        description="Generate configuration files for parameter sweep"
    )

    # parser.add_argument(
    #    "-e",
    #    "--experiment",
    #    action="store",
    #    type=str,
    #    help="name of the experiment",
    #    required=True,
    #)

    parser.add_argument(
        "-cf",
        "--config_file",
        action="store",
        type=str,
        help="path to the config file for building the model sweep",
        required=True,
    )

    args = parser.parse_args(argv)

    # config_folder = os.path.join("experiments", args.experiment, "perpl_config")

    gen_configs(args.config_file)



if __name__ == "__main__":
    main()
