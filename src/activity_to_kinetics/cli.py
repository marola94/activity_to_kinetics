# src/activity_to_kinetics/cli.py
import argparse
from . import utility_functions as uf  # well_arg bor her hos dig

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Proccess activity data and plot, determine rate or kinetics.'
    )

    parser.add_argument("-m", "--measurements_file", type=str,
                        help="Path to the input Excel file with activity measurements.")

    parser.add_argument("-w", "--wells", type=uf.well_arg,
                        help=("Specify which wells to analyze. You can specifiy specific wells (e.g. A1; -w A1) "
                              "or whole rows and columns of the 96MTP (e.g. row B and/or line 4; -w B, 4). "
                              "It is also possible to specify wells along with columns or rows "
                              "(e.g. row C, 2 and well B7; -w C, 2, B7 ). "))

    parser.add_argument("-f", "--fitting", action='store_true',
                        help="Activates manual rate fitting if called together with the -r flag.")

    parser.add_argument("-k", "--estimate_kinetics", action='store_true',
                        help="Activates estimation of kinetic parameters.")

    parser.add_argument("-r", "--estimate_rates", action='store_true',
                        help="Activates automatic rate estimation.")

    parser.add_argument("-s", "--rates_input_file", type=str,
                        help="Path to the input Excel file with rates from activity experiments.")

    parser.add_argument("-q", "--rates_output_file", type=str,
                        help="Name of output excel file for intial rates.")

    parser.add_argument("-o", "--kinetics_output_file", type=str,
                        help="Name of output excel file for kinetics.")

    parser.add_argument("-d", "--output_directory", type=str,
                        help="The directory to store the output files. Default is in working directory.")

    parser.add_argument("-n", "--create_output", action='store_true',
                        help="Saves the results as defualt Excel file.")

    parser.add_argument("-p", "--plot_activity", action='store_true',
                        help="Plotting the activity measurements.")

    return parser.parse_args(argv)
