import argparse
import datetime
from itertools import product
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import yaml

from perpl.io import plotting
from perpl.modelling.gen_distance_model_configs import gen_configs
from perpl.modelling.modelling_general import PERPLModel



def get_experimental_rpd(distances, rpd_type, fitlength,
                          bin_size, kde_kernel_size, model_config):
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
    models,
    model_configs,
    kde_kernel_size,
    bin_size,
    fitlength,
    output_folder,
    results,
):
    # for each model...
    for i, model in enumerate(models):

        if i == 0:
            print(f"Output folder: {output_folder}")

        model_name = model.rstrip(".yaml")
        model_config = model_configs[i]

        x_expt, y_expt = get_experimental_rpd(
            distances, rpd_type, fitlength,
            bin_size, kde_kernel_size, model_config,
        )

        perpl_model = PERPLModel(
            dimension=model_config["dimension"],
            background=model_config["background"],
            n_peaks=model_config["n_peaks"],
            peak_type=model_config["peak_type"],
            characteristic_distance=model_config["characteristic_distance"],
            characteristic_distance_ratio=model_config["characteristic_distance_ratio"],
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
            print(f"Skipping {model_name}:"
                  " contains no distances,"
                  " repeated localisations or background"
            )
            continue

        # print("Model name ", model_name) Debug

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
                (
                    f"{model_name}_fitlength_{fitlength}_binsize_{bin_size}_histandfit.svg"
                ),
            )

        elif rpd_type == "distance_kde":
            # plot kde and fit
            fig = perpl_model.plot_distance_kde_and_fit(x_expt, y_expt, fitlength)
            figname = os.path.join(
                output_folder,
                (
                    f"{model_name}_fitlength_{fitlength}_kdeandfit.svg"
                ),
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
                (
                    f"{model_name}_fitlength_{fitlength}_modelcomponents.svg"
                ),
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
                ):
                    f.write(f"{row[0]}: {row[1]} +- {row[2]}\n")

        # save ssr, aic, aiccorr, setup
        results["ssrs"].append(perpl_model.sum_of_squares_error)
        results["aics"].append(perpl_model.aic)
        results["aiccorrs"].append(perpl_model.aic_corrected)
        if rpd_type == "distance_histogram":
            results["setups"].append(
                f"{model_name}_fitlength_{fitlength}_binsize_{bin_size}"
            )
        elif rpd_type == "distance_kde":
            results["setups"].append(
                f"{model_name}_fitlength_{fitlength}"
            )
        results["fitlengths"].append(fitlength)
        results["bgbelowzeros"].append(perpl_model.bgbelowzero)
        results["nparams"].append(perpl_model.n_params)
        results["ndatapoints"].append(len(x_expt))
        results["ndistances"].append(len(distances))
        results["popt_at_bounds"].append(perpl_model.popt_at_bound)
        results["large_uncertainties"].append(perpl_model.large_uncertainty)


def make_results():
    return {k: [] for k in [
        "ssrs","aics","aiccorrs","setups","fitlengths",
        "bgbelowzeros","nparams","ndatapoints","ndistances",
        "popt_at_bounds","large_uncertainties"
    ]}


def save_results(path, results):
    rows = zip(
        results["setups"],
        results["aiccorrs"],
        results["aics"],
        results["ssrs"],
        results["fitlengths"],
        results["bgbelowzeros"],
        results["nparams"],
        results["ndatapoints"],
        results["ndistances"],
        results["popt_at_bounds"],
        results["large_uncertainties"],
    )

    with open(path, "w") as f:
        f.write("Model,AICcorr,AIC,SSR,Fitlength,BGbelowzero,Nparams,"
                "Ndatapoints,Ndistances,POptAtBounds,LargeUncertainty\n")
        for row in rows:
            f.write(",".join(map(str, row)) + "\n")


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

    print(f"Fit hists: {args.fit_histograms}")
    print(f"Fit KDEs: {args.fit_kdes}")

    config_file = args.config_file

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Generate models to use
    models_folder = gen_configs(config_file, timestamp)

    # Collect data pre-processwing settings
    with open(config_file, "r") as ymlfile:
        config = yaml.safe_load(ymlfile)
    model_direction = config["model_direction"]
    limits = config["limits"]
    bin_size_lst = config["bin_sizes"]
    kde_kernel_size_lst = config["kde_kernel_size"]
    fitlength_lst = config["fitlength"]

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

    # load in models
    model_files = os.listdir(models_folder)
    print(f"{len(model_files)} models are being tested")

    model_configs = []
    for model in model_files:
        with open(
            os.path.join(models_folder, model), "r"
        ) as ymlfile:
            config = yaml.safe_load(ymlfile)
            model_configs.append(config)

    # Stop if fitting not required
    if args.no_fitting:
        return

    # Set up output parent folder
    ## Contains table of results and subdirector(ies) for fits 
    parent, _ = os.path.split(args.rel_posns_file)

    output_modelling_folder = os.path.join(
        parent, f"modelling_output_{model_direction}_{timestamp}"
    )
    if not os.path.exists(output_modelling_folder):
        os.makedirs(output_modelling_folder)

    # +++ FIT...

    # .... histograms

    if args.fit_histograms:
        
        # Create output folder
        output_folder_hists = os.path.join(output_modelling_folder, "histogram_fits")
        if not os.path.exists(output_folder_hists):
            os.makedirs(output_folder_hists)

        results = make_results()

        for preproc_param in list(
            product(
                bin_size_lst,
                fitlength_lst,
            )
        ):
            bin_size, fitlength = preproc_param

            model_the_data(
                distances,
                "distance_histogram",  # rpd_type
                model_files,
                model_configs,
                None,  # kde_kernel_size
                bin_size,
                fitlength,
                output_folder_hists,
                results,
            )

        output_path = os.path.join(
            output_modelling_folder, "results_histograms.csv"
        )

        save_results(output_path, results)

    # ... KDEs
    if args.fit_kdes:
        
        # Create output folder
        output_folder_kdes = os.path.join(output_modelling_folder, "kde_fits")
        if not os.path.exists(output_folder_kdes):
            os.makedirs(output_folder_kdes)

        results = make_results()

        for preproc_param in list(
            product(
                kde_kernel_size_lst,
                fitlength_lst,
            )
        ):
            kde_kernel_size, fitlength = preproc_param

            print(f"kde_kernel, fitlength: {kde_kernel_size}, {fitlength}")

            model_the_data(
                distances,
                "distance_kde",  # rpd_type
                model_files,
                model_configs,
                kde_kernel_size,
                None,  # bin_size
                fitlength,
                output_folder_kdes,
                results,
            )

        output_path = os.path.join(
            output_modelling_folder, "results_kdes.csv"
        )

        save_results(output_path, results)


if __name__ == "__main__":
    main()
