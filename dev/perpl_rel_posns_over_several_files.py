import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from perpl.io import plotting
# from perpl.relative_positions import main as calculate_relative_positions
from perpl.relative_positions import getdistances, get_vectors, save_relative_positions


def main(argv=None):
    """Main script for the module with variable arguments

    Args:
        argv : Custom arguments to run script with"""

    # parse arugments
    parser = argparse.ArgumentParser(
        description="Calculate relative positions using PERPL"
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
        "-lf",
        "--localisation_filter",
        action="store",
        type=int,
        help="ensure each point cloud has over this number of localisations",
        default=100,
    )

    parser.add_argument(
        "-f",
        "--filter_distance",
        action="store",
        type=int,
        default=150,
        help="filter distance",
    )

    parser.add_argument(
        "-nn",
        "--nearest_neighbours",
        action="store",
        type=int,
        default=0,
        help="Number of nearest neighbours to find within the filter distance, "
        "if desired. 0 (default) means no limit on the number of "
        "neighbours used within the filter distance.",
    )

    parser.add_argument(
        "-b",
        "--bin_size",
        action="store",
        type=int,
        default=1,
        help="Size of the bins for plotting the relative positions in histogram",
    )

    parser.add_argument(
        "-v",
        "--visualise",
        action="store_true",
        default=False,
        help="Whether to visualise plots etc.",
    )

    args = parser.parse_args(argv)

    folder = os.path.join("experiments", args.experiment)

    input_folder = os.path.join(
        folder, "data"
    )
    output_folder = os.path.join(folder, "output", "perpl_relative_posns")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # input folder
    loc_prec_folders = [
        os.path.join(input_folder, x)
        for x in os.listdir(input_folder)
        if os.path.isdir(os.path.join(input_folder, x))
    ]
    if len(loc_prec_folders) == 0:
        raise ValueError("No input folders")

    for input_folder in loc_prec_folders:

        localisation_precision_filter = input_folder.split("/")[-1].replace(
            "nm_filter", ""
        )

        print("Input folder: ", input_folder)
        print("Localisation precision filter: ", localisation_precision_filter, "nm")

        files = os.listdir(input_folder)

        d_values_list = []

        mean_precision_list = []
        dropped_files = []
        input_locs = 0
        output_locs = 0

        for file in files:

            file_path = os.path.join(input_folder, file)

            df = pl.read_csv(file_path)

            if len(df) < args.localisation_filter:
                print(f"Insufficient localisations for file {file}")
                dropped_files.append(file)
                continue

            input_locs += len(df)

            mean_x_precision = (
                df.select(
                    pl.col("x_err")
                    .str.strip_chars_start(" ")
                    .cast(pl.Float64)
                )
                .mean()
                .item()
            )
            mean_y_precision = (
                df.select(
                    pl.col("y_err")
                    .str.strip_chars_start(" ")
                    .cast(pl.Float64)
                )
                .mean()
                .item()
            )
            mean_z_precision = (
                df.select(
                    pl.col("z_err").str.strip_chars_start(" ").cast(pl.Float64)
                )
                .mean()
                .item()
            )

            # Based on lines 845-910 in perpl.relative_positions

            xyz_values = df[["x", "y", "z"]].to_numpy()

            if args.visualise:

                fig = plt.figure()
                ax = fig.add_subplot(projection="3d")
                ax.scatter(
                    xyz_values[:, 0],
                    xyz_values[:, 1],
                    xyz_values[:, 2],
                    s=1,
                    marker="x",
                )
                ax.set_xlabel("X [nm]")
                ax.set_ylabel("Y [nm]")
                ax.set_zlabel("Z [nm]")
                plt.gca().set_aspect("equal", adjustable="box")
                plt.show()

            d_values = getdistances(
                xyz_values, args.filter_distance, args.nearest_neighbours, verbose=False
            )[1]

            # Get distances in 2D and 3D for relative positions.
            # Note, get_vectors() is an unhelpful name as it takes the vectors we already have
            # and calculates distances.
            if len(d_values) > 0:
                d_values = get_vectors(d_values, dims=3)
                d_values_list.append(d_values)
                mean_precision_list.append(
                    (mean_x_precision + mean_y_precision + mean_z_precision) / 3
                )
                output_locs += len(df)
            else:
                print(f"No distances within filter for {file}")
                dropped_files.append(file)

        if not len(d_values_list) == 0:

            d_values = np.concatenate(d_values_list, axis=0)

            results_dir = os.path.join(
                folder,
                "perpl_relative_posns",
                f"{args.filter_distance}filterdistance_{localisation_precision_filter}precisionfilter_{args.localisation_filter}numberoflocalisations",
            )

            if not os.path.exists(results_dir):
                os.makedirs(results_dir)

            info = {
                "results_dir": results_dir,
                "short_names": False,
                "in_file_no_extension": f"all_z_disks_{localisation_precision_filter}precisionfilter_{args.localisation_filter}numberoflocalisations",
            }

            ## Plot vector component results
            plotting.plot_histograms(
                d_values,
                dims=3,
                filter_distance=args.filter_distance,
                info=info,
                binsize=args.bin_size,
            )

            # change so relative positions files saved dir higher than hists
            info["results_dir"] = results_dir = os.path.join(
                folder, "perpl_relative_posns"
            )

            ## Save relative positions and vector components.
            xyz_filename = save_relative_positions(
                d_values,
                args.filter_distance,
                dims=3,
                info=info,
                nns=args.nearest_neighbours,
            )

            file_name = (
                xyz_filename.replace("relpos", "locprec").rstrip(".csv") + ".txt"
            )

            print("Mean precision (after filtering): ", np.mean(mean_precision_list))

            np.savetxt(file_name, np.atleast_1d(np.mean(mean_precision_list)))

            print("Dropped files: ", dropped_files)
            print("Number of dropped files: ", len(dropped_files))
            print("Number of retained files: ", len(files) - len(dropped_files))
            print("% of localisations kept: ", 100 * output_locs / input_locs)

        else:

            print("No output files")


if __name__ == "__main__":
    main()
