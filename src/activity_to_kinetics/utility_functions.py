import numpy as np
import pandas as pd
import argparse
import re
import sys
import datetime
from pathlib import Path
import os
from collections import Counter
from .units import ureg, map_concentration_unit


def filter_columns_by_well_list(df, well_list):
    """
    Filter a DataFrame to only include columns that match elements in well_list.
    Supports full rows (e.g., 'A'), full columns (e.g., '2'), or exact wells (e.g., 'A2').

    Args:
        df (pd.DataFrame): Input dataframe with 96-well format columns (e.g., A1, B7).
        well_list (List[str]): List of validated well inputs.

    Returns:
        pd.DataFrame: Filtered DataFrame with only matching columns.
    """
    filtered_cols = set()

    for entry in well_list:
        entry = entry.strip().upper()

        if re.fullmatch(r'[A-H]', entry):  # Match full row (e.g., 'A')
            matching = [col for col in df.columns if re.fullmatch(rf'{entry}(?:[1-9]|1[0-2])', col)]
            filtered_cols.update(matching)

        elif re.fullmatch(r'[1-9]|1[0-2]', entry):  # Match full column (e.g., '2')
            matching = [col for col in df.columns if re.fullmatch(rf'[A-H]{entry}', col)]
            filtered_cols.update(matching)

        elif re.fullmatch(r'[A-H](?:[1-9]|1[0-2])', entry):  # Match exact well (e.g., 'B7')
            if entry in df.columns:
                filtered_cols.add(entry)

    return df.loc[:, sorted(filtered_cols)]

def well_arg(value):
    """Parse and validate -w input for 96 MTP wells. Returns a list of valid entries."""
    if isinstance(value, list):
        entries = value
    elif isinstance(value, str):
        entries = value.split(',') if ',' in value else [value]
    else:
        raise argparse.ArgumentTypeError("Input must be a string or a list.")

    validated_entries = []
    for entry in entries:
        entry = entry.strip().upper()
        if re.fullmatch(r'[A-H]', entry):  # Valid row (A–H)
            validated_entries.append(entry)
        elif re.fullmatch(r'[1-9]|1[0-2]', entry):  # Valid column (1–12)
            validated_entries.append(entry)
        elif re.fullmatch(r'[A-H](?:[1-9]|1[0-2])', entry):  # Valid well (e.g., A1–H12)
            validated_entries.append(entry)
        else:
            raise argparse.ArgumentTypeError(f"The well input '{entry}' is not a valid 96 MTP coordinate.")
    
    return validated_entries

def convert_absorbance_to_concentration(Abs, conversion_parameters):

    Abs = ureg.Quantity(Abs,ureg.dimensionless)
    
    conversion_method = conversion_parameters['Conversion method']
    canonical_unit = map_concentration_unit(conversion_parameters['Desired unit'])
    if canonical_unit:
        output_unit = ureg(canonical_unit).units
    else:
        output_unit = ureg(conversion_parameters['Desired unit']).units

    print(f'\n Converting absorbances to {output_unit} with {conversion_method.lower()}.')

    if conversion_method == 'Calibration curve':
            
        intercept = conversion_parameters['Calibration curve intercept']
        slope_value = conversion_parameters['Calibration curve slope']
        slope_unit = ureg(conversion_parameters['Calibration curve slope unit'])

        if slope_unit != 1/output_unit:

            # Convert the slope of the calibration curve to 
            slope = ureg.Quantity(slope_value,slope_unit).to(1/output_unit)

        else:
            slope = ureg.Quantity(slope_value, slope_unit)
            
        c = calibration_curve(Abs, slope, intercept)

    elif conversion_method == 'Lambert-Beers':
        
        extinction_coefficient_value = conversion_parameters['Extinction coefficient']
        extinction_coefficient_unit = ureg(conversion_parameters['Extinction coefficient unit'])
        extinction_coefficient = ureg.Quantity(extinction_coefficient_value, extinction_coefficient_unit)
        
        path_length_value = conversion_parameters['Path length']
        path_length_unit = ureg(conversion_parameters['Path length unit'])
        path_length = ureg.Quantity(path_length_value, path_length_unit)

        if not extinction_coefficient.check((1 / (output_unit * path_length.units)).dimensionality): # 1 / (output_unit * path_length) gives the extinction coefficient in desired unit format
            sys.exit(
                "\n \t \u274C Error in unit conversion \n"
                " Unit conversion was requested, but the desired unit and the extinction coefficient unit are incompatible.\n"
                f" Desired unit is {output_unit}, while extinction coefficient unit is {extinction_coefficient_unit}.\n"
                " Actions:\n"
                " \u2192 Select a desired unit and extinction coefficient which are compatible.\n"
                " \u2192 Set 'Desired unit' to 'No conversion' and skip the unit conversion of the uploaded measurements."            
            )

        
        if extinction_coefficient.units != (1 / (output_unit * path_length.units)):
            extinction_coefficient = extinction_coefficient.to(1 / (output_unit * path_length.units))

        c = lambert_beers(Abs, extinction_coefficient, path_length)

    return c

# Function for calculating concentration from absorbance by a calibration curve
def calibration_curve(absorbance, slope, intercept):
    return ((absorbance-intercept)/slope)

# Function for converting absorbances to concentrations with Lambert-Beers law
def lambert_beers(absorbance, extincion_coefficient, path_length):
    return absorbance / (extincion_coefficient * path_length)

# Function that convert datetime.time array into seconds
def datetime_to_seconds(datetime_array):
    seconds_array = []
    for i in range(len(datetime_array)):
        seconds = int(datetime.timedelta(hours=datetime_array[i].hour, minutes=datetime_array[i].minute, seconds=datetime_array[i].second).total_seconds())
        seconds_array.append(seconds)

    seconds_array = np.array(seconds_array).astype('float64').T
    return seconds_array

# Function that convert an array of seconds to an array of datetime.time objects
def seconds_to_datetime(seconds_array):
    datetime_array = []
    for i in range(len(seconds_array)):
        dt = str(datetime.timedelta(seconds=seconds_array[i]))
        dt = datetime.datetime.strptime("%H:%M:%S",dt)
        datetime_array.append(dt)
    
    datetime_array = np.array(datetime_array)
    return datetime_array

def convert_times(index, conversion_dictionary):
    """
    Convert the times in tehe index of the dataframe to pint.Quantity.
    """
    if all(isinstance(x, datetime.time) for x in index):
        # Case 1: Excel time format
        seconds_array = datetime_to_seconds(index)
        t = ureg.Quantity(seconds_array, 's')

    elif all(isinstance(x, (int, float)) for x in index):
        # Case 2: numeric + time unit
        time_unit = conversion_dictionary.get('Time unit')
        if not time_unit:
            sys.exit(
                "\n \t \u274C Error reading file with measurements\n"
                " Issue: Times are numeric but no 'Time unit' was provided.\n"
                " Actions:\n"
                " \u2192 Specify a time unit in the Excel sheet from the dropdown.\n"
            )
        t = ureg.Quantity(index.to_numpy(dtype=float), time_unit)

    else:
        sys.exit(
            "\n \t \u274C Error reading file with measurements\n"
            " Issue: Unsupported time format in 'Time' column.\n"
            " Actions:\n"
            " \u2192 Report times either as Excel time format (HH:MM:SS) or as numbers along with a unit next to the 'Time unit' cell.\n"
        )

    return t


def rescale_times(t):
    """
    Convert a time Quantity array to one of: s, min, h, d
    Thresholds:
      - span_sec >= 500        -> minutes
      - span_sec >= 300*60     -> hours
      - span_sec >  72*3600    -> days
      else                     -> seconds

    Times are converted according to having 6 ticks on the x-axis. 
    """
    t = t.to("s")
    a = np.asarray(t.magnitude) # Array with numbers

    if a.size == 0:
        return t  # nothing to do

    span_sec = float(np.nanmax(a) - np.nanmin(a))

    if span_sec > 72 * 3600:          # over 72 hours -> days
        return t.to("d")
    elif span_sec >= 300 * 60:        # 300 minutes or more -> hours
        return t.to("h")
    elif span_sec >= 500:             # 500 seconds or more -> minutes
        return t.to("min")
    else:
        return t                     # keep seconds
    
def create_path(path: str | None = None) -> Path:

    """
    Return an absolute OUTPUT directory path.

    - If `path` is None or empty -> use the current working directory (CWD).
    - Relative paths are interpreted relative to CWD.
    - Absolute paths are used as-is.
    - '~' and environment variables are expanded.
    - If the path exists and is a FILE -> raise error.
    - If the path does not exist and looks like a file name (has a suffix) -> raise error.
    - If the path does not exist and looks like a directory -> create it.
    """

    if path is None or str(path).strip() == "":
        # No user input: default to the current working directory
        p = Path.cwd() / "data" / "processed"
    else:
        # Expand environment variables and the user's home (~)
        s = os.path.expandvars(str(path).strip())
        p = Path(s).expanduser()
        # If it's a relative path, anchor it at the current working directory
        if not p.is_absolute():
            p = Path.cwd() / p

    if p.exists():
        # If it exists, it must be a directory
        if p.is_file():
            raise NotADirectoryError(f"Output path points to a file, not a directory: {p}")
    else:
        # Reject clearly file-like names when the path doesn't exist (e.g., 'results.csv')
        if p.suffix and not p.name.startswith("."):
            raise ValueError(f"'{p.name}' looks like a file name. Provide a DIRECTORY path instead.")
        # Create the directory (including parents)
        p.mkdir(parents=True, exist_ok=True)

    # Return a canonical absolute path (nice to print/store/use later)
    return p.resolve()

def generate_file_path(user_input, storing_path, kind) -> Path:

    """
    Build the final .xlsx file path.
    - Relative inputs are resolved under `storing_path`.
    - Absolute inputs are used as-is (user is responsible).
    - If extension is not '.xlsx', print an info message and force '.xlsx'.
    - If no filename is given, use defaults per `kind`.
    - Returns a Path you can write to directly (parent dirs are created).
    """

    # Defaults + message template
    if kind == "rates":
        default_stem = "rate_results"
        msg_tpl = ("The program can only save files to .xlsx, "
                   "the rate results will be saved as {filename}.xlsx")
    elif kind == "kinetics":
        default_stem = "kinetic_results"
        msg_tpl = ("The program can only save files to .xlsx, "
                   "the kinetics results will be saved as {filename}.xlsx")
    else:
        raise ValueError("kind must be 'rates' or 'kinetics'")

    # No filename provided -> use default in storing_path
    if user_input is None or str(user_input).strip() == "":
        final_path = (storing_path / f"{default_stem}.xlsx").resolve()
        final_path.parent.mkdir(parents=True, exist_ok=True)
        return final_path

    # Expand env vars and ~, keep any folders the user typed
    s_in = os.path.expandvars(str(user_input).strip())
    p_in = Path(s_in).expanduser()

    # Resolve relative vs absolute against storing_path
    base_path = p_in if p_in.is_absolute() else (storing_path / p_in)

    # Determine stem + suffix from the *filename part*
    stem = base_path.stem if base_path.suffix else base_path.name
    suffix = base_path.suffix.lower()

    # Handle extension rules (+ print info if needed)
    if suffix == ".xlsx":
        final_name = base_path.name
    elif suffix == "":
        final_name = f"{stem}.xlsx"
    else:
        # Non-xlsx -> force .xlsx and inform the user, showing the user's stem
        print(msg_tpl.format(filename=stem))
        final_name = f"{stem}.xlsx"

    final_path = base_path.with_name(final_name).resolve()
    final_path.parent.mkdir(parents=True, exist_ok=True)  # ensure folders exist
    return final_path

def update_excel(
    file_path: str,
    df_update: pd.DataFrame,
    key: str
    ):

    # 1) Load the existing sheet into a DataFrame
    df_old = pd.read_excel(file_path, dtype={key: str})
    
    # 2) Ensure the key column is treated as string and strip whitespace
    df_old[key] = df_old[key].astype(str).str.strip()
    df_update[key] = df_update[key].astype(str).str.strip()
    
    # 3) Align columns (union of old and new) 
    #    so that new columns from df_update are also included if needed
    all_cols = list(dict.fromkeys(list(df_old.columns) + list(df_update.columns)))
    df_old = df_old.reindex(columns=all_cols)
    df_update = df_update.reindex(columns=all_cols)
    
    # 4) Set the key as index for both DataFrames 
    #    so we can update rows directly
    df_old_idx = df_old.set_index(key)
    df_upd_idx = df_update.set_index(key)
    
    # 5) Update rows where the key already exists
    #    Only non-NaN values from df_update will overwrite
    df_old_idx.update(df_upd_idx)
    
    # 6) Add rows for keys that do not exist in the old DataFrame
    new_keys = df_upd_idx.index.difference(df_old_idx.index)
    df_final = pd.concat([df_old_idx, df_upd_idx.loc[new_keys]], axis=0)
    
    # 7) Save back to the same file and sheet 
    #    (other sheets in the Excel file remain untouched)
    with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as w:
        df_final.reset_index().to_excel(w, index=False)

# Function to convert all values in a column to the most common unit
def convert_to_most_common_unit(df, columns):
    
    """
    Converts specified columns in a DataFrame from pint.Quantity to float64 in the most common unit, and append the unit in the column name.
    
    Parameters:
    df (pandas.DataFrame): The dataframe containing the data.
    column_name (list): A list of column names to convert.

    Returns:
    pandas.DataFrame: The dataframe with the specified columns converted to their most common units, and values as float64.
    dict: A dictionary with the most common unit for each specified column.
    
    Example:
    >>> df, unit = convert_to_most_common_unit(df, 'Rate')
    >>> print(f"The most common unit in the 'Rate' column is: {unit}")
    """

    common_units = {}
    for column in columns:
        # Extract the units
        units = [value.units for value in df[column] if isinstance(value, ureg.Quantity)]
        most_common_unit = Counter(units).most_common(1)[0][0]

        common_units[column] = most_common_unit
        
        # Convert all values to the most common unit
        df[column] = df[column].apply(lambda x: x.to(most_common_unit).magnitude if isinstance(x, ureg.Quantity) else x)
        
        # Rename the column to include the unit
        df.rename(columns={column: f'{column} [{most_common_unit}]'}, inplace=True)
    
    return df, most_common_unit

def assign_units_to_values_and_update_columns(df:pd.DataFrame) -> pd.DataFrame:
    '''
    Extract unit from column name in af dataframe (e.g. Initial rate [µM/s] -> µM/s) and assign the unit to the column.
    '''
    columns_to_drop = []

    for col in df.columns:
        match = re.search(r'\[(.*?)\]', col)
        if match:
            # Extract the unit from the column name
            unit = match.group(1)
            col_name = re.sub(r'\s*\[.*?\]', '', col)
            
            # Assign units to the values in the column
            df[col_name] = df[col].apply(lambda x: ureg.Quantity(x, unit))
            
            # Add the original column to the drop list
            columns_to_drop.append(col)
    
    # Drop the old columns
    df.drop(columns=columns_to_drop, inplace=True)
    
    return df

def to_quantity_vector(df, column_names):
    for column in column_names:
        if column in df.columns:
            # Check if the column contains pint quantities
            #if all(isinstance(x, ureg.Quantity) for x in df[column]):
                units = [value.units for value in df[column] if isinstance(value, ureg.Quantity)]
                most_common_unit = Counter(units).most_common(1)[0][0]

                # Convert all values to the most common unit and get magnitudes
                magnitudes = [value.to(most_common_unit).magnitude for value in df[column]]

                # Extract magnitudes into a list
                magnitudes = df[column].apply(lambda x: x.magnitude).tolist()

                # Combine the list with the unit
                # Create a single Quantity array with the most common unit
                quantity_array = ureg.Quantity(magnitudes, most_common_unit)

    return quantity_array

def collect_models_to_dataframe(model_collection:dict) -> pd.DataFrame:

    # Flatten the nested dictionaries
    flattened_data = []
    for outer_key, inner_dict in model_collection.items():
        flat_dict = {'Model ID': outer_key}
        flat_dict.update(inner_dict)
        flattened_data.append(flat_dict)

    # Convert to DataFrame
    df = pd.DataFrame(flattened_data)

    # Define the desired order of columns
    print(df.columns)
    columns_order = ['Model No.', 'Title', 'Model ID', 'Model type', 'Enzyme concentration',  
                     'Vmax', 'std. Vmax', 'Km', 'std. Km', 'kcat', 'std. kcat', 'std. Ki', 'Ki',
                     'Removed indices', 'RMSE', 'r_squared', 'MAE' 
                     ]

    # Reorder the columns based on the defined order
    # Fill missing columns with NaN
    for col in columns_order:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[columns_order]

    # List to store column names that contain pint.Quantity
    pint_columns = [col for col in df.columns if any(isinstance(x, ureg.Quantity) for x in df[col])]
    df, _ = convert_to_most_common_unit(df, pint_columns)
    
    return df