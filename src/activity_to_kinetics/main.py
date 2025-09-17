import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openpyxl import load_workbook

# internal packages
from . import utility_functions as uf
from . import rate_calculation as rc
from . import file_loading as fl
from . import plot_generator as pg
from . import validation
from . import kinetic_model_fitting as kmf
from .units import ureg, map_concentration_unit

def main(args):

    #make it a possibility to skip the interface if specific flags are called

    abs_conversion_dict = {
        "convert_absorbances": args.convert_absorbances,
        "conversion_unit": args.conversion_unit,
        "slope": args.slope,
        "intercept": args.intercept,
        "extinction_coefficient": args.extinction_coefficient 
    }
    


    # Path to store output files and figures
    storing_path = uf.create_path(args.output_directory)

    # Path to store results files
    rate_result_path = uf.generate_file_path(args.rates_output_file, storing_path, kind="rates")
    kinetics_result_path = uf.generate_file_path(args.kinetics_output_file, storing_path, kind="kinetics")

    # Load measurements, units and titles for individual reactions
    if args.measurements_file:
        conversion_dict, data_df, titles_df = fl.load_measurements_file(args.measurements_file)

    # Load specified datapoint intervals for slope fitting
    if args.fitting and args.estimate_rates and args.measurements_file:
        slope_fitting_df = fl.load_manual_rate_fitting(args.measurements_file)

    # Load settings for kinetics calculations
    if args.estimate_kinetics and args.measurements_file and (args.rates_input_file or args.estimate_rates):

        # Check conversion_dict has settings for converting absorbances (i.e. AU) to a unit
        if conversion_dict.get('Measurement unit') == 'AU' and args.estimate_kinetics:
            if not conversion_dict.get('Conversion method'):
                sys.exit(
                    "\n \t \u274C Failed to Execute Commands \n"
                    " Issue: Kinetics can not be computed without rates being in molar units.\n"
                    " Actions:\n"
                    " \u2192 Insert parameters for conversion of absorbances to molar units with either an extioncion coefficient and path lenght\n"
                    " or a calibration curve, in the Measurements sheet of the Template_data_file.\n"
                    )

        kinetics_settings_dict = fl.load_kinetics_settings(args.measurements_file)
        if kinetics_settings_dict:
            if args.rates_input_file and not args.estimate_rates:

                loaded_rates_df = fl.load_rate_data_file(args.rates_input_file) # Loading Excel file with rates
                
                kinetics_settings_dict = validation.validate_wells(kinetics_settings_dict, loaded_rates_df, wells_in = 'index')

                loaded_rates_df = uf.assign_units_to_values_and_update_columns(loaded_rates_df)

            elif args.rates_input_file and args.estimate_rates:
                sys.exit(
                    "\n \t \u274C Failed to Execute Commands \n"
                    "Issue:\n"
                    "  You attempted to run kinetics using BOTH uploaded rates and in-program rate estimation.\n"
                    "\n"
                    "Do exactly ONE of the following:\n"
                    "  \u2192 Option A — Use uploaded rates with kinetics:\n"
                    "     - Provide the rates Excel:        -s <rates.xlsx>\n"
                    "     - Enable kinetics computation:    -k\n"
                    "     - Provide the measurements file:  -m <Template_data_file.xlsx>\n"
                    "     Example:  myprog -k -s rates.xlsx -m Template_data_file.xlsx\n"
                    "\n"
                    "  \u2192 Option B — Estimate rates in the program and run kinetics:\n"
                    "     - Enable kinetics computation:    -k\n"
                    "     - Enable rate estimation:         -r\n"
                    "     - Provide the measurements file:  -m <Template_data_file.xlsx>\n"
                    "     Example:  myprog -k -r -m Template_data_file.xlsx\n"
                    "\n"
                    "Additional notes:\n"
                    "  - Always upload the measurements file (Template_data_file.xlsx) via -m and follow the instructions\n"
                    "    in the 'Kinetics settings' sheet to define models before computing kinetics.\n"
                )

            elif not args.rates_input_file and args.estimate_rates:
                
                kinetics_settings_dict = validation.validate_wells(kinetics_settings_dict, data_df, wells_in = 'columns')

                print(f'Calculate kinetics based on rates estimated from ')
        else:
            if args.estimate_rates:
                choice = input(
                    "\n \t \u274C Failed to Execute Kinetics Command \n"
                    " Issue: No kinetics models were defined or all were discarded due to improper input in the Kinetics settings sheet of the Excel file.\n"
                    " Action:\n"
                    " \u2192 Follow the instructions in the Kinetics settings sheet of the Excel file to calculate kinetics.\n"
                    " \n Do you want to continue estimating the rates? (Yes/No): "
                    ).strip().lower()
                if choice in {"y", "yes"}:
                    print("Continuing with rate estimation.")
                    args.estimate_kinetics = False
                else:
                    sys.exit(
                        "Program stopped..."
                    )
            else:
                sys.exit(
                    "\n \t \u274C Failed to Execute Kinetics Command \n"
                    " The kinetics flag was raised but the Kinetics settings sheet in the Excel file was not defined correctly.\n"
                    " Actions:\n" \
                    " \u2192 Follow the instructions in the Kinetics settings sheet of the Excel file in order to calculate kinetics.\n"
                )


    # Pruning dataframes, if only specific wells are chosen to analyze 
    if args.wells:
        data_df = uf.filter_columns_by_well_list(data_df, args.wells)
        titles_df = uf.filter_columns_by_well_list(titles_df, args.wells)
        if args.fitting and args.estimate_rates and args.measurements_file:
            slope_fitting_df = uf.filter_columns_by_well_list(slope_fitting_df, args.wells)
    
    if args.measurements_file and (args.estimate_rates or args.plot_activity):

        rates_results_list = []
        for WELL in data_df:
            
            print(f'\n \t ************** Processing well {WELL} **************')

            # Removes row with NaN values or strings to secure only numbers are in columns 
            well_data = pd.to_numeric(data_df[WELL], errors='coerce').dropna()

            # The 'Desired unit' key is decisive for wether the measurements will be converted or not
            if conversion_dict.get('Desired unit'):
            
                if conversion_dict.get('Measurement unit') == 'AU':
                    # Converting absorbances to concentrations
                    x = uf.convert_absorbance_to_concentration(Abs=np.array(well_data), conversion_parameters=conversion_dict)
                    canonical_unit = map_concentration_unit(str(x.units))
                    if canonical_unit:
                        x = x.to(canonical_unit)


                else:
                    # Define data to the supplied unit of the measurements and convert to desired unit
                    x = ureg.Quantity(np.array(well_data),conversion_dict['Measurement unit']).to({conversion_dict['Desired unit']})
                    canonical_unit = map_concentration_unit(str(x.units))
                    if canonical_unit:
                        x = x.to(canonical_unit)

            else:
                if conversion_dict.get('Measurement unit') == 'AU':
                    x = ureg.Quantity(np.array(well_data), ureg.AU)
                else:
                    x = ureg.Quantity(np.array(well_data),conversion_dict['Measurement unit'])
                    canonical_unit = map_concentration_unit(str(x.units))
                    if canonical_unit:
                        x = x.to(canonical_unit)
            
            # Retrieve times
            timestamps = uf.convert_times(data_df.index,conversion_dict)

            kwargs = {}
            if args.estimate_rates:

                # Retrieve the defined datapoint interval for the correpsonding well if defined 
                datapoint_interval = (
                                    {'Start datapoint': slope_fitting_df.loc['Start datapoint', WELL],
                                    'End datapoint':   slope_fitting_df.loc['End datapoint', WELL]}
                                    if args.fitting and WELL in slope_fitting_df.columns else {}
                                )
                
                # ---------------- Calculate initial rate, yield and smoothed data ----------------
                # Retrives a dictionary of either four or five keys; 'Filtered curve', 'Initial rate', 'Intercept', 'Yield' and 'R2', 
                # where 'Filtered curve' is included if the automatic rate determination is launched
                rate_dict = rc.determine_initial_rate(timestamps.magnitude, x.magnitude, **datapoint_interval)

                rate_dict['Well'] = WELL
                rate_dict['Initial rate'] = ureg.Quantity(rate_dict['Initial rate'],f'{x.units}/{timestamps.units}')
                rate_dict['Yield'] = ureg.Quantity(rate_dict['Yield'],x.units)
                rate_dict['Intercept'] = ureg.Quantity(rate_dict['Intercept'],x.units)
                if 'Filtered curve' in rate_dict:
                    rate_dict['Filtered curve'] = ureg.Quantity(rate_dict['Filtered curve'], x.units)
                
                if WELL in titles_df:
                    rate_dict['Experiment title'] = titles_df.loc['Title',WELL]
                else:
                    rate_dict['Experiment title'] = ""

                kwargs = rate_dict

                print('\n \u2705 Initial rate determined.')

                if (args.create_output or args.rates_output_file or args.output_directory or args.estimate_kinetics):
                    rates_results_list.append(rate_dict)

            if args.plot_activity:

                plot1, _ = pg.activity_plot(x=uf.rescale_times(timestamps), y=x, well=WELL, **kwargs)

                if WELL in titles_df:
                    title = titles_df.loc['Title',WELL]
                    filename_plot1 = f'activity_{title}_{WELL}' 
                else:
                    filename_plot1 = f'activity_{WELL}'

                path = f'{storing_path}/activity_plots'
                if not os.path.isdir(path):
                    os.makedirs(path)
                    
                plot1.savefig(f'{path}/'+f'{filename_plot1}'+'.png', format='png', bbox_inches='tight', dpi=500)



        rates_results_df = pd.DataFrame(rates_results_list) 
        if args.estimate_rates and (args.create_output or args.rates_output_file or args.output_directory):
            

            columns_to_drop = ['Filtered curve']
            rates_output_df = rates_results_df.drop(columns=columns_to_drop) 

            first_cols = [c for c in ["Well", "Experiment title"] if c in rates_output_df.columns]
            order = first_cols + [c for c in rates_output_df.columns if c not in first_cols]
            rates_output_df = rates_output_df[order]

            pint_columns = [col for col in rates_output_df.columns if any(isinstance(x, ureg.Quantity) for x in rates_output_df[col])]
            rates_output_df, _ = uf.convert_to_most_common_unit(rates_output_df, pint_columns) 

            # Check if the file exists
            if not os.path.exists(rate_result_path):
                print('Converting rate data to excel file')
                rates_output_df.to_excel(rate_result_path, index=False)
                print('File saved\n')
            
            else:
                print(f'Updating {args.rates_output_file}, with new data' )
                uf.update_excel(file_path=rate_result_path, df_update=rates_output_df, key='Well')
                print('Rate results file updated\n') 

    if args.estimate_kinetics:
        if args.rates_input_file:
            rates_df = loaded_rates_df
        else:
            rates_results_df = rates_results_df.set_index('Well')
            rates_df = uf.assign_units_to_values_and_update_columns(rates_results_df)

        for model_number, model_settings in kinetics_settings_dict.items():

            print(f'\n \t ************** Processing {model_number} **************')

            data_df = model_settings['data'].copy()
            data_df['Wells'] = data_df['Wells'].astype(str).str.strip()

            # Extract initial rates from matching wells
            rates_col = rates_df.loc[rates_df.index.intersection(model_settings['data']['Wells']), "Initial rate"]
            rates_col.index = rates_col.index.astype(str).str.strip()

            # Join uses data_df's index, and matches 'Wells' column on the right's index
            merged = data_df.join(rates_col, on='Wells', how='inner')
            
            # Sort by substrate concentration index in descending order
            merged = merged.sort_index(ascending=True)

            # Move index out and assign units to substrate concentrations and extract as array. And convert numpy array with Quantity elements, to Quantity array
            sub_conc = uf.to_quantity_vector(uf.assign_units_to_values_and_update_columns(merged.reset_index()),['Substrate concentration'])

            # Convert numpy array with Quantity elements, to Quantity array 
            initial_rates = uf.to_quantity_vector(merged,['Initial rate'])

            # get enzyme concentration key
            enz_key = next(key for key in model_settings if str(key).startswith('Enzyme concentration'))

            # Get unit from enzyme concentration key
            unit = re.search(r'\[(.*?)\]', enz_key).group(1)
            enz_conc = ureg.Quantity(model_settings[enz_key],unit)

            base_enz_key = re.sub(r'\s*\[.*?\]\s*', '', enz_key); model_settings[base_enz_key] = enz_conc; model_settings.pop(enz_key, None)         

            # Attach model no. for kinetics and plotting  
            model_settings['Model No.'] = model_number

            # Generate models
            model_collection = kmf.model_fitting(sub_conc, initial_rates, **model_settings)
            
            if not model_collection:
                continue
            
            for key, model in model_collection.items():

                fig = pg.kinetic_plot(model, key)

                model_number = model.get('Model No.')

                filename = f'{key}'
                path = f'{storing_path}/kinetic_plots/{model_number.replace(" ","")}'
                if not os.path.isdir(path):
                    os.makedirs(path)
                    
                fig.savefig(f'{path}/'+f'{filename}'+'.png', format='png', bbox_inches='tight', dpi=500)

            print('\n \u2705 Kinetics for {model_number} determined.')

            kinetics_output_df = uf.collect_models_to_dataframe(model_collection)

            if (args.create_output or args.kinetics_output_file or args.output_directory):
                # Check if the file exists
                if not os.path.exists(kinetics_result_path):
                    print('Converting rate data to excel file')
                    kinetics_output_df.to_excel(kinetics_result_path, index=False)
                    print('File saved\n')
                        
                else:
                    print(f'Updating {args.kinetics_output_file}, with new data' )
                    uf.update_excel(file_path=kinetics_result_path, df_update=kinetics_output_df, key='Well')
                    print('Rate results file updated\n') 


            # --- If time allows ---
            #   Update rate calculation code -> faster and better


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Proccess activity data and plot, determine rate or kinetics.')

    parser.add_argument("-m", "--measurements_file", type=str, 
                        help="Path to the input Excel file with activity measurements.")
    parser.add_argument("-w", "--wells", type=uf.well_arg, help="Specify which wells to analyze. You can specifiy specific wells (e.g. A1; -w A1) " \
                        "or whole rows and columns of the 96MTP (e.g. row B and/or line 4; -w B, 4). It is also possible to specify wells along with columns" \
                        "or rows (e.g. row C, 2 and well B7; -w C, 2, B7 ). ")

    parser.add_argument("-c", "--convert_absorbances", type=str, choices=['Standard','extinction','none'],
                        help="Convert absorbances to concentrations and type either 'Standard' or 'Extinction' to specify the conversion method. Or type 'None' to continue without conversion.")
    parser.add_argument("-u", "--conversion_unit", type=str, choices=["mol/L", "mmol/L", "umol/L", "nmol/L"], 
                        help="The desired unit to convert absorbances to.")
    parser.add_argument("-a", "--slope", type= float,
                        help="The slope of the standard curve given in the same unit as the specified unit with the '-u' flag.")
    parser.add_argument("-b", "--intercept", type= float, 
                        help="The intercept of the standard curve given in the same unit as the specified unit with the '-u' flag.")
    parser.add_argument("-e", "--extinction_coefficient", type=float, 
                        help=r"The extinction coefficient, given in M$^{-1}$cm${-1}$.")
    
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
    parser.add_argument("-d", "--output_directory", type = str,
                        help = "The directory to store the output files. Default is in working directory.")
    parser.add_argument("-n", "--create_output", action='store_true',
                        help = "Saves the results as defualt Excel file.")

    parser.add_argument("-p", "--plot_activity", action='store_true',
                        help="Plotting the activity measurements.")
    
    args = parser.parse_args()
    main(args)
    