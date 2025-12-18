import argparse
import copy
from itertools import product
import os
import warnings

import yaml


def gen_configs(config_folder, direction):

    # load in configuration
    with open(os.path.join(config_folder, "config.yaml"), "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    # load in params for particular direction
    direction_params = config[direction]

    # load in params list
    dimension = direction_params["dimension"]
    backgrounds = direction_params["background"]
    n_peaks = direction_params["n_peaks"]
    peak_types = direction_params["peak_type"]
    charac_dists = direction_params["charac_dist"]
    charac_dist_ratio = direction_params["charac_dist_ratio"]
    repeats = direction_params["repeats"]
    offsets = direction_params["offset"]
    normalises = direction_params["normalise"]
    params_initial = direction_params["params_initial"]
    params_lower = direction_params["params_lower"]
    params_upper = direction_params["params_upper"]

    if type(params_initial["characteristic_distance_1"]) is list:
        assert len(params_initial["characteristic_distance_1"]) == len(charac_dists)
        warnings.warn(
            "Multiple characteristic distances for first peak "
            f"for {direction} direction. Therefore, assuming peak distances "
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

        # if something
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
            config_folder, f"{direction}_models/model_{index}.yaml"
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

    parser.add_argument(
        "-e",
        "--experiment",
        action="store",
        type=str,
        help="name of the experiment",
        required=True,
    )

    parser.add_argument(
        "-a",
        "--only_axial",
        action="store_true",
        help="Fit only the axial direction",
        required=False,
    )

    parser.add_argument(
        "-t",
        "--only_transverse",
        action="store_true",
        help="Fit only the transverse plane",
        required=False,
    )

    args = parser.parse_args(argv)

    config_folder = os.path.join("experiments", args.experiment, "perpl_config")

    folders = ["axial_models", "transverse_models"]

    if args.only_axial:
        folders.remove("transverse_models")
    
    if args.only_transverse:
        folders.remove("axial_models")

    for f in folders:
        folder = os.path.join(config_folder, f)
        if os.path.exists(folder):
            raise ValueError(f"Cannot proceed as {folder} folder already exists")

    for f in folders:
        folder = os.path.join(config_folder, f)
        os.makedirs(folder)

    # generate axial and transverse config files
    if not args.only_transverse:
        gen_configs(config_folder, "axial")
    if not args.only_axial:
        gen_configs(config_folder, "transverse")


if __name__ == "__main__":
    main()
