import argparse
import datetime
import os
import sys
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from perpl.io import plotting
from perpl.modelling.gen_distance_model_configs import gen_configs
from perpl.modelling.modelling_general import PERPLModel


def get_experimental_rpd(
    distances, rpd_type, fitlength, bin_size, kde_kernel_size, model_config
):
    """
    Args:
        ...
        rpd_type (str):
            "distance_histogram" or "distance_kde"

    Returns:
        x_expt, y_expt (1D numpy arrays):
            Values of the distance histogram or KDE (y_expt)
            at the distances in x_expt.
    """
    if rpd_type == "distance_histogram":
        hist_values, edges = np.histogram(
            distances, bins=np.arange(0, fitlength + 1, bin_size)
        )
        centres = (edges[:-1] + edges[1:]) / 2
        x_expt = centres
        y_expt = hist_values

    elif rpd_type == "distance_kde":
        increment = max(1, round(fitlength / len(distances)))
        x_expt = np.arange(0, fitlength + 1.0, increment)

        if model_config["dimension"] == 3:
            raise ValueError(
                "dimension: 3 (from config file) is not implemented"
                " for KDE generation"
            )

        churchman_map = {
            1: plotting.estimate_rpd_churchman_1d,
            2: plotting.estimate_rpd_churchman_2d,
        }
        fn = churchman_map.get(model_config["dimension"])

        y_expt = fn(
            input_distances=distances,
            calculation_points=x_expt,
            combined_precision=kde_kernel_size,
        )

    else:
        raise ValueError("RPD type currently unsupported.")

    # Avoid zero distance for dividing by distance
    # Actual normalisation is done later in
    # fit_to_experiment
    if model_config["normalise"]:
        mask = x_expt > 0
        x_expt = x_expt[mask]
        y_expt = y_expt[mask]

    return x_expt, y_expt


def model_the_data(
    distances,  # 1D numpy array of distances
    rpd_type,  # "distance_histogram" or "distance_kde"
    model_file,
    model_config,
    kde_kernel_size,
    bin_size,
    fitlength,
    output_folder,
):
    model_name = model_file.rstrip(".yaml")

    x_expt, y_expt = get_experimental_rpd(
        distances,
        rpd_type,
        fitlength,
        bin_size,
        kde_kernel_size,
        model_config,
    )

    perpl_model = PERPLModel(
        dimension=model_config["dimension"],
        background=model_config["background"],
        n_peaks=model_config["n_peaks"],
        peak_amps=model_config["peak_amps"],
        dist_ratios=model_config["dist_ratios"],
        custom_ratios_list=model_config["custom_ratios_list"],
        repeats=model_config["repeats"],
        offset=model_config["offset"],
        normalise=model_config["normalise"],
        params_initial=model_config["params_initial"],
        params_lower=model_config["params_lower"],
        params_upper=model_config["params_upper"],
        name=model_name,
    )

    if (
        model_config["background"] is None
        and model_config["n_peaks"] == 0
        and model_config["repeats"] is False
    ):
        print(
            f"Skipping {model_name}:"
            " contains no characteristic distances,"
            " repeated localisations or background"
        )
        return None

    perpl_model.fit_to_experiment(
        x_expt,
        y_expt,
    )

    if rpd_type == "distance_histogram":
        # plot distance hist and fit
        bin_centres = x_expt
        bin_width = bin_centres[1] - bin_centres[0]
        bin_edges = x_expt - bin_width / 2
        bin_edges = np.append(bin_edges, bin_edges[-1] + bin_width)

        fig = perpl_model.plot_distance_hist_and_fit(
            distances,
            bin_edges,
            bin_centres,
            fitlength,
        )
        figname = os.path.join(
            output_folder,
            (f"{model_name}_fitlength_{fitlength}_binsize_{bin_size}_histandfit.svg"),
        )

    elif rpd_type == "distance_kde":
        # plot kde and fit
        fig = perpl_model.plot_distance_kde_and_fit(x_expt, y_expt, fitlength)
        figname = os.path.join(
            output_folder,
            (f"{model_name}_fitlength_{fitlength}_kdeandfit.svg"),
        )

    if fig is not None:
        fig.savefig(figname)
        plt.close(fig)

    # plot model components
    fig2 = perpl_model.plot_model_components(fitlength)
    if rpd_type == "distance_histogram":
        figname = os.path.join(
            output_folder,
            (
                f"{model_name}_fitlength_{fitlength}_binsize_{bin_size}_modelcomponents.svg"
            ),
        )
    elif rpd_type == "distance_kde":
        figname = os.path.join(
            output_folder,
            (f"{model_name}_fitlength_{fitlength}_modelcomponents.svg"),
        )
    if fig2 is not None:
        fig2.savefig(figname)
        plt.close(fig2)

    # save model params and err
    if rpd_type == "distance_histogram":
        opt_param_path = os.path.join(
            output_folder,
            f"{model_name}_fitlength_{fitlength}_binsize_{bin_size}_optparams.txt",
        )
    elif rpd_type == "distance_kde":
        opt_param_path = os.path.join(
            output_folder,
            f"{model_name}_fitlength_{fitlength}_optparams.txt",
        )
    with open(opt_param_path, "w") as f:
        f.write("Optimal params +- Error\n")
        f.write("-----------------------\n")
        if perpl_model.params_optimised is None:
            f.write("Model failed to fit")
        else:
            for row in zip(
                perpl_model.param_names,
                perpl_model.params_optimised,
                perpl_model.params_err,
                strict=True,
            ):
                f.write(f"{row[0]}: {row[1]} +- {row[2]}\n")

    return x_expt, y_expt, perpl_model


def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with"""

    # parse arugments
    parser = argparse.ArgumentParser(description="Model the data using PERPL")

    parser.add_argument(
        "-rp",
        "--rel_posns_file",
        action="store",
        type=str,
        help="path to the file containing the relative positions",
        required=True,
    )

    parser.add_argument(
        "-cf",
        "--config_file",
        action="store",
        type=str,
        help="path to the config file to use for generating models",
        required=True,
    )

    parser.add_argument(
        "-fh",
        "--fit_histograms",
        action="store_true",
        help="fit histograms; choose at least one of -fh and -fkde",
        required=False,
    )

    parser.add_argument(
        "-fkde",
        "--fit_kdes",
        action="store_true",
        help="fit kdes; choose at least one of -fh and -fkde",
        required=False,
    )

    parser.add_argument(
        "-nofit",
        "--no_fitting",
        action="store_true",
        help="Only generate model configuration files, do not fit to data",
        required=False,
    )

    args = parser.parse_args(argv)

    if not (args.fit_histograms or args.fit_kdes or args.no_fitting):
        parser.error("Must specify at least one of" " --fh, --fkde or -nofit")

    print(f"Fit hists: {args.fit_histograms}")
    print(f"Fit KDEs: {args.fit_kdes}")

    config_file = args.config_file

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Collect RPD types to run
    run_modes = []

    if args.fit_histograms:
        run_modes.append("distance_histogram")
    if args.fit_kdes:
        run_modes.append("distance_kde")

    # Collect data pre-processing settings
    with open(config_file, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    model_direction = config["model_direction"]
    limits = config["limits"]
    fitlength_lst = config["fitlength"]

    if "distance_histogram" in run_modes:
        bin_size_lst = config["bin_sizes"]
    if "distance_kde" in run_modes:
        kde_kernel_size_lst = config["kde_kernel_size"]

    # load in relative positions and configuration
    relpos = pd.read_csv(args.rel_posns_file)
    if relpos.empty:
        raise ValueError("Relative positions dataframe must not be empty.")

    # Limit the relative positions in X, Y and Z if desired
    relpos = relpos.loc[abs(relpos["xx_separation"]) < limits["x_limit"]]
    relpos = relpos.loc[abs(relpos["yy_separation"]) < limits["y_limit"]]
    relpos = relpos.loc[abs(relpos["zz_separation"]) < limits["z_limit"]]

    # Select the data in the selected direction
    distances = relpos[f"{model_direction}_separation"].to_numpy()

    # Ensure absolute distance and remove duplicates
    distances = abs(distances)
    distances = np.sort(distances)[::2]

    # Generate models to use and load in
    try:
        models_folder = gen_configs(config_file, timestamp)
    except ValueError as e:
        print(f"Error {e}")
        sys.exit(1)
    model_files = os.listdir(models_folder)
    print(f"{len(model_files)} models are being tested")

    model_configs = []
    for model_file in model_files:
        with open(os.path.join(models_folder, model_file), "r") as ymlfile:
            config = yaml.safe_load(ymlfile)
            model_configs.append(config)

    # Stop if fitting not required, only model config files
    if args.no_fitting:
        return

    # Set up output parent folder
    ## Contains table of results and subdirectory(ies) for fits
    parent, _ = os.path.split(args.rel_posns_file)

    output_modelling_folder = os.path.join(
        parent, f"modelling_output_{model_direction}_{timestamp}"
    )
    os.makedirs(output_modelling_folder)

    # For histoagrmmed and kde data separately
    for rpd_type in run_modes:
        # Create output locations;
        if rpd_type == "distance_histogram":
            out_folder = "histogram_fits"
            results_file = "results_histograms.csv"
        elif rpd_type == "distance_kde":
            out_folder = "kde_fits"
            results_file = "results_kde.csv"
        out_folder_path = os.path.join(output_modelling_folder, out_folder)
        os.makedirs(out_folder_path)
        results_path = os.path.join(output_modelling_folder, results_file)

        print(f"Output folder: {out_folder_path}")

        # Initialise results
        if rpd_type == "distance_histogram":
            x_kernel_col = "Bin width"
        else:
            x_kernel_col = "KDE kernel width"
        model_results = []

        # Iterate through models
        if rpd_type == "distance_histogram":
            bin_or_kernel_list = bin_size_lst
            kde_kernel_size = None
        else:
            bin_or_kernel_list = kde_kernel_size_lst
            bin_size = None

        count = 0

        for bin_or_kernel, fitlength in product(
            bin_or_kernel_list,
            fitlength_lst,
        ):
            if rpd_type == "distance_histogram":
                bin_size = bin_or_kernel
                kde_kernel_size = None
            else:
                bin_size = None
                kde_kernel_size = bin_or_kernel

            for model_file, model_config in zip(
                model_files, model_configs, strict=True
            ):
                model_out = model_the_data(
                    distances,
                    rpd_type,
                    model_file,
                    model_config,
                    kde_kernel_size,
                    bin_size,
                    fitlength,
                    out_folder_path,
                )
                if model_out is None:
                    continue
                x_expt, y_expt, fitted_model = model_out

                model_results.append(
                    {
                        "Name": model_file.rstrip(".yaml"),
                        "Fit length": fitlength,
                        x_kernel_col: bin_or_kernel,
                        "N Peaks": model_config["n_peaks"],
                        "Peak ratios": model_config["peak_amps"],
                        "Model distance ratios": model_config["dist_ratios"],
                        "Repeated localisations": model_config["repeats"],
                        "Background model": model_config["background"],
                        "AICc": fitted_model.aic_corrected,
                        "AIC": fitted_model.aic,
                        "SSR": fitted_model.sum_of_squares_error,
                        "N params": fitted_model.n_params,
                        "N calculation points": len(x_expt),
                        "N measured distances": len(distances),
                        "Check: bg < 0": fitted_model.bgbelowzero,
                        "Check: params reached limits": fitted_model.popt_at_bound,
                        "Check: large param uncertainty": (
                            fitted_model.large_uncertainty,
                        ),
                    }
                )

                if (count + 1) % 10 == 0 and count > 0:
                    print(f"{count + 1} models run out of {len(model_files)}...")
                count += 1

        results_df = pd.DataFrame(model_results)
        results_df.to_csv(results_path, index=None)
