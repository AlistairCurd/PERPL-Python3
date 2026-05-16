import argparse
import datetime
from itertools import product
import matplotlib.pyplot as plt
import os
import sys

import numpy as np
import pandas as pd

import yaml

from perpl.io import plotting
from perpl.gen_model_sweep_configs import gen_configs
from perpl.modelling.modelling_general import PERPLModel


def model_the_data(
    distances,  # 1D numpy array of distances
    plot_type,  # "hist" or "kde"
    models,
    model_configs,
    kde_kernel_size,
    bin_size,
    fitlength,
    output_folder,
    ssrs,
    aics,
    aiccorrs,
    setups,
    fitlengths,
    bgbelowzeros,
    nparams,
    ndatapoints,
    ndistances,
    popt_at_bounds,
    large_uncertainties,
):
    print("Modelling...")

    # for each model...
    for i, model in enumerate(models):

        if i == 0:
            print(f"Output folder: {output_folder}")

        model_name = model.rstrip(".yaml")
        model_config = model_configs[i]

        if plot_type == "histogram":

            # Get the histogram data up to distance = fitlength
            hist_values, bin_edges = np.histogram(
                distances, bins=np.arange(0, fitlength + 1, bin_size)
            )
            bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

            x_expt = bin_centres
            y_expt = hist_values

        elif plot_type == "kde":

            if len(distances) == 0:
                print(f"Skipping {model_name} as no distances to fit")
                continue

            increment = np.round(fitlength / len(distances))
            if increment == 0:
                increment = 1
            calculation_points = np.arange(0, fitlength + 1.0, increment)

            print("Found x points.")

            if model_config["dimension"] == 1:
                churchman = plotting.estimate_rpd_churchman_1d
            elif model_config["dimension"] == 2:
                churchman = plotting.estimate_rpd_churchman_2d
            elif model_config["dimension"] == 3:
                print("3D KDE function not yet implemented")
                sys.exit()
                # churchman = plotting.estimate_rpd_churchman_3d

            rpd = churchman(
                input_distances=distances,
                calculation_points=calculation_points,
                combined_precision=kde_kernel_size,
            )

            if model_config["normalise"]:
                y_expt = rpd[calculation_points > 0]
                x_expt = calculation_points[calculation_points > 0]
            else:
                y_expt = rpd
                x_expt = calculation_points
            
            print("Set up expt RPD.")

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

        if plot_type == "histogram":
            # plot distance hist and fit
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

        elif plot_type == "kde":
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
        if plot_type == "histogram":
            figname = os.path.join(
                output_folder,
                (
                    f"{model_name}_fitlength_{fitlength}_binsize_{bin_size}_modelcomponents.svg"
                ),
            )
        elif plot_type == "kde":
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
        if plot_type == "histogram":
            opt_param_path = os.path.join(
                output_folder,
                f"{model_name}_fitlength_{fitlength}_binsize_{bin_size}_optparams.txt",
            )
        elif plot_type == "kde":
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
        ssrs.append(perpl_model.sum_of_squares_error)
        aics.append(perpl_model.aic)
        aiccorrs.append(perpl_model.aic_corrected)
        if plot_type == "histogram":
            setups.append(
                f"{model_name}_fitlength_{fitlength}_binsize_{bin_size}"
            )
        elif plot_type == "kde":
            setups.append(
                f"{model_name}_fitlength_{fitlength}"
            )
        fitlengths.append(fitlength)
        bgbelowzeros.append(perpl_model.bgbelowzero)
        nparams.append(perpl_model.n_params)
        ndatapoints.append(len(x_expt))
        ndistances.append(len(distances))
        popt_at_bounds.append(perpl_model.popt_at_bound)
        large_uncertainties.append(perpl_model.large_uncertainty)


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

        ssrs = []
        aics = []
        aiccorrs = []
        setups = []
        fitlengths = []
        bgbelowzeros = []
        nparams = []
        ndatapoints = []
        ndistances = []
        popt_at_bounds = []
        large_uncertainties = []

        for preproc_param in list(
            product(
                bin_size_lst,
                fitlength_lst,
            )
        ):
            bin_size, fitlength = preproc_param

            model_the_data(
                distances,
                "histogram",  # plot_type
                model_files,
                model_configs,
                None,  # kde_kernel_size
                bin_size,
                fitlength,
                output_folder_hists,
                ssrs,
                aics,
                aiccorrs,
                setups,
                fitlengths,
                bgbelowzeros,
                nparams,
                ndatapoints,
                ndistances,
                popt_at_bounds,
                large_uncertainties,
            )

        (
            aiccorrs,
            aics,
            ssrs,
            setups,
            fitlengths,
            bgbelowzeros,
            nparams,
            ndatapoints,
            ndistances,
            popt_at_bounds,
            large_uncertainties,
        ) = zip(
            *sorted(
                zip(
                    aiccorrs,
                    aics,
                    ssrs,
                    setups,
                    fitlengths,
                    bgbelowzeros,
                    nparams,
                    ndatapoints,
                    ndistances,
                    popt_at_bounds,
                    large_uncertainties,
                )
            )
        )

        with open(os.path.join(output_modelling_folder, "results_histograms.csv"), "w") as f:
            f.write(
                "Model,AICcorr,AIC,SSR,Fitlength,BGbelowzero,Nparams,Ndatapoints,Ndistances,POptAtBounds,LargeUncertainty\n"
            )
            for row in zip(
                setups,
                aiccorrs,
                aics,
                ssrs,
                fitlengths,
                bgbelowzeros,
                nparams,
                ndatapoints,
                ndistances,
                popt_at_bounds,
                large_uncertainties,
            ):
                f.write(",".join(map(str, row)) + "\n")

    # ... KDEs
    if args.fit_kdes:
        
        # Create output folder
        output_folder_kdes = os.path.join(output_modelling_folder, "kde_fits")
        if not os.path.exists(output_folder_kdes):
            os.makedirs(output_folder_kdes)

        ssrs = []
        aics = []
        aiccorrs = []
        setups = []
        fitlengths = []
        bgbelowzeros = []
        nparams = []
        ndatapoints = []
        ndistances = []
        popt_at_bounds = []
        large_uncertainties = []

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
                "kde",  # plot_type
                model_files,
                model_configs,
                kde_kernel_size,
                None,  # bin_size
                fitlength,
                output_folder_kdes,
                ssrs,
                aics,
                aiccorrs,
                setups,
                fitlengths,
                bgbelowzeros,
                nparams,
                ndatapoints,
                ndistances,
                popt_at_bounds,
                large_uncertainties,
            )

            print("Modelled once.")

        (
            aiccorrs,
            aics,
            ssrs,
            setups,
            fitlengths,
            bgbelowzeros,
            nparams,
            ndatapoints,
            ndistances,
            popt_at_bounds,
            large_uncertainties,
        ) = zip(
            *sorted(
                zip(
                    aiccorrs,
                    aics,
                    ssrs,
                    setups,
                    fitlengths,
                    bgbelowzeros,
                    nparams,
                    ndatapoints,
                    ndistances,
                    popt_at_bounds,
                    large_uncertainties,
                )
            )
        )

        with open(os.path.join(output_modelling_folder, "results_kdes.csv"), "w") as f:
            f.write(
                "Model,AICcorr,AIC,SSR,Fitlength,BGbelowzero,Nparams,Ndatapoints,Ndistances,POptAtBounds,LargeUncertainty\n"
            )
            for row in zip(
                setups,
                aiccorrs,
                aics,
                ssrs,
                fitlengths,
                bgbelowzeros,
                nparams,
                ndatapoints,
                ndistances,
                popt_at_bounds,
                large_uncertainties,
            ):
                f.write(",".join(map(str, row)) + "\n")


if __name__ == "__main__":
    main()
