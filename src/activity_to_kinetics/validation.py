import pandas as pd
import sys
import sys
import datetime
from units import ureg, map_concentration_unit

def validate_parameters(all_params):
    
    # --- Check that 'Measurement unit' is defined ---
    if pd.isna(all_params.get('Measurement unit')):
        sys.exit(
            "\n \t \u274C Error reading file with measurements\n" 
            " Issue: Measurements unit must be defined along with the uploaded measurements.\n"
            " Actions:\n"
            " \u2192 Please provide a unit in the uploaded Excel sheet by entering it in the cell to the right of 'Measurement unit'.\n"
        )

    # --- If 'Desired unit' is not defined, keep only 'Measurement unit' and 'Time unit' (if not NaN/None) ---
    if pd.isna(all_params.get('Desired unit')):
        allowed_keys = ['Measurement unit', 'Time unit']
        return {k: v for k, v in all_params.items()
                if k in allowed_keys and not pd.isna(v)}

    # --- Remove undefined or empty values (None/NaN) ---
    all_params = {k: v for k, v in all_params.items() if not pd.isna(v)}

    # --- Check for conflict: both conversion methods are fully defined ---
    has_calibration_curve = all(
        key in all_params for key in [
            'Calibration curve slope',
            'Calibration curve intercept',
            'Calibration curve slope unit'
        ]
    )
    has_lambert_beers = all(
        key in all_params for key in [
            'Extinction coefficient',
            'Extinction coefficient unit',
            'Path length',
            'Path length unit'
        ]
    )

    if has_calibration_curve and has_lambert_beers:
        sys.exit(
            "\n \t \u274C Error reading file with measurements\n"
            " Both calibration curve and Lambert-Beers parameters have been defined.\n"
            " Actions:\n"
            " \u2192 Choose only one conversion method in the uploaded Excel file.\n"
        )

    elif not has_calibration_curve and not has_lambert_beers:
        sys.exit(
            "\n \t \u274C Error reading file with measurements\n"
            " Unit conversion was requested, but neither calibration curve or Lambert-Beers parameters were defined.\n"
            " Actions:\n"
            " \u2192 Choose include conversion parameters for one of the conversion methods in the uploaded Excel file.\n"
            " \u2192 Set 'Desired unit' to 'No conversion' and skip the unit conversion of the uploaded measurements.\n"            
        )

    # --- If only calibration curve is defined: remove extinction-related keys ---
    if has_calibration_curve and not has_lambert_beers:
        for key in ['Extinction coefficient', 'Extinction coefficient unit', 'Path length', 'Path length unit']:
            all_params.pop(key, None)
        all_params['Conversion method'] = 'Calibration curve'

    # --- If only Lambert-Beers method is defined: remove calibration-related keys ---
    elif has_lambert_beers and not has_calibration_curve:
        for key in ['Calibration curve slope', 'Calibration curve intercept', 'Calibration curve slope unit']:
            all_params.pop(key, None)
        all_params['Conversion method'] = 'Lambert-Beers'

    # --- Return cleaned and validated parameters ---
    return all_params


def validate_wells(models: dict, rates_df: pd.DataFrame, wells_in: str = "index") -> dict:
    """
    Keep only models whose 'data'['Wells'] are all present in rates_df.
    
    wells_in: 
      - "index"   -> wells are in rates_df.index
      - "columns" -> wells are in rates_df.columns
    
    Assumes:
      - models[model_key]["data"] is a DataFrame
      - column "Wells" exists in that DataFrame
    Prints a message for every removed model.
    Returns a cleaned copy of the models dict.
    """

    def _norm_well(x):
        if pd.isna(x):
            return None
        s = str(x).strip().upper()
        return s if s else None

    # --- Reference wells ---
    if wells_in == "index":
        ref_wells = {w for w in (_norm_well(idx) for idx in rates_df.index) if w is not None}
    elif wells_in == "columns":
        ref_wells = {w for w in (_norm_well(col) for col in rates_df.columns) if w is not None}
    else:
        raise ValueError("wells_in must be 'index' or 'columns'")

    cleaned = {}
    for model_key, inner in models.items():
        df = inner["data"]
        model_wells = {w for w in (_norm_well(v) for v in df["Wells"]) if w is not None}

        # Subset check
        if not model_wells.issubset(ref_wells):
            missing = sorted(model_wells - ref_wells)
            msg = (
                f"The wells defined in the Kinetics settings sheet for {model_key} "
                f"do not match with the wells listed in the uploaded file.\n"
            )
            if missing:
                msg += f" (missing: {', '.join(missing)})\n"
            msg += f" Kinetics will not be calculated for {model_key}.\n"
            print(msg)
            continue

        cleaned[model_key] = inner
    
    if not cleaned and wells_in=='index':
        sys.exit(
            "\n \t \u274C Failed to Execute Kinetics Command \n"
            " Issue: All kinetics models were discarded due to improper misalignment between wells in the inserted\n"
            " Kinetics settings sheet of the Excel file and the uploaded file with rates.\n"
            " Action:\n"
            " \u2192 Make sure the wells in the uploaded file with rates match those in the Kinetics settings sheet."
            " Follow the instructions in the Kinetics settings sheet of the Excel file."
        )
        
    if not cleaned and wells_in=='columns':
        sys.exit(
            "\n \t \u274C Failed to Execute Kinetics Command \n"
            " Issue: All kinetics models were discarded due to improper misalignment between wells in the inserted\n"
            " Kinetics settings sheet of the Excel file and the uploaded file with measurements.\n"
            " Action:\n"
            " \u2192 Make sure the wells in the uploaded file with measurements match those in the Kinetics settings sheet."
            " Follow the instructions in the Kinetics settings sheet of the Excel file."
        )

    return cleaned


def validating_kinetics_dictionary(models: dict):
    print('\n\t *** Validating kinetics settings ***\n')
    
    cleaned = {}
    k = 0
    for model_key, inner in models.items():
        df  = inner.get("data", None)
        enz_key = next((k for k in inner.keys() if str(k).startswith("Enzyme concentration")), None) 
        # the key for the enzyme concentration can vary depending on the unit provided in the key for the enzyme concentration
        enz = inner.get(enz_key, None)

        # Ensure DF exists with correct structure
        if not isinstance(df, pd.DataFrame) or "Wells" not in df.columns:
            df = pd.DataFrame(columns=["Wells"]).rename_axis("Substrate concentration")

        # Build masks: index must be non-empty, Wells must be non-empty
        idx_series = pd.Series(df.index, index=df.index)
        idx_ok   = idx_series.notna() & (idx_series.astype(str).str.strip() != "")
        wells_ok = df["Wells"].notna() & (df["Wells"].astype(str).str.strip() != "")

        both_ok   = idx_ok & wells_ok
        n_before  = len(df)
        n_valid   = int(both_ok.sum())
        dropped   = n_before - n_valid

        # Clean dataframe
        cleaned_df = df.loc[both_ok].copy()

        # Track fatal vs non-fatal
        err_msgs = []
        fatal = False

        # (2) Enzyme conc missing
        if enz is None:
            err_msgs.append(
            " \u2022 An enzyme concentration as well as a unit has to be defined.\n"
            " Actions:\n"
            " \u2192 Provide the enzyme concentration for the series of substrate concentration and well pairs in the Excel sheet with Kinetics settings.\n"
            )
            fatal = True

        # (3) Misalignment found and <4 remain
        if dropped > 0 and n_valid < 4:
            err_msgs.append(
                " \u2022 Misalignment between well and corresponding substrate concentration found and the element was discarded.\n"
                " Furthermore less than 4 substrate concentration and well pairs are left and kinetics can't be computed for this model.\n"
                " Actions:\n"
                " \u2192 Align the given substrate concentration with a well where an initial rate has been calculated\n"
                " and provide at least 4 substrate concentration and well pairs in the Excel sheet with Kinetics settings.\n"
            )
            fatal = True

        # (4) Misalignment found and >=4 remain
        if dropped > 0 and n_valid >= 4:
            err_msgs.append(
                " \u2022 Misalignment between well and corresponding substrate concentration found and the element was discarded.\n"
                " Actions:\n"
                " \u2192 Align the given substrate concentration with a well where an initial rate has been calculated in \n"
                " the Excel sheet with Kinetics settings.\n"
            )

        # (1) Not enough data points (only if no misalignment happened)
        if dropped == 0 and n_valid < 4: 
            err_msgs.append(
                " \u2022 At least 4 data points are needed to fit a kinetic model.\n"
                " Actions:\n"
                " \u2192 Insert at least 4 wells and their corresponding substrate concentrations in the Excel sheet with Kinetics settings.\n"
            )
            fatal = True

        # Print messages if any
        if err_msgs:
            k+=1
            if k ==1:
                print(f"---------------------------------------------------------------")
            
            print(f"\n \u2757 Improper settings for {model_key} \u2757 \n\n"+f"\n".join(err_msgs))

            if fatal:
                print(f"\n Kinetics will not be computed for {model_key}\n")
                print(f"---------------------------------------------------------------")
            else:
                print(f"\n Kinetics will be computed for {model_key} with valid pairs\n")
                print(f"---------------------------------------------------------------")

        # Keep model only if not fatal
        if not fatal:
            cleaned[model_key] = {**inner, "data": cleaned_df}

    return cleaned

def validate_unit(unit_str: str, var_type: str):
    """
    Checking the unit for subtrate and enzyme concentration,
    and returns the unit in pint format.
    var_type is either 'enzyme' or 'substrate'.
    """

    valid_units = ["nM", "µM", "mM", "M"]

    canonical_unit_format = map_concentration_unit(unit_str)

    try:
        # Try to parse unit in pint
        unit = ureg(canonical_unit_format).units
    except Exception:
        if var_type == "enzyme":
            sys.exit(
                "\n \t \u274C Error in kinetics settings\n"
                " Issue: The provided unit for the enzyme concentration is not supported.\n"
                " Actions:\n"
                " \u2192 Specify a unit from the dropdown in the Excel sheet in the cell to the right of 'Enzyme concentration unit'.\n"
                )
        elif var_type == "substrate":
            sys.exit(
                "\n \t \u274C Error in kinetics settings\n"
                " Issue: The provided unit for the substrate concentration is not supported.\n"
                " Actions:\n"
                " \u2192 Specify a unit from the dropdown in the Excel sheet in the cell to the right of 'Substrate concentration unit'.\n"
                )
        
    # Convert to pint unit format
    unit_str_norm = f"{unit}"

    if unit_str_norm not in valid_units:
        if var_type == "enzyme":
            raise ValueError("Invalid unit for enzyme concentration")
        elif var_type == "substrate":
            raise ValueError("Invalid unit for substrate concentration")

    return unit

def checking_time_format(data_df, conversion_dictionary):
    
    if all(isinstance(x, datetime.time) for x in data_df.index):
        return
    
    else:    
        if not conversion_dictionary.get('Time unit'):
            # No time unit, which is required for converting array of number and
            # times are not datetime.time object
            sys.exit(
                "\n \t \u274C Error reading file with measurements\n" 
                " Issue: Times are reported as numeric values, but no time unit was provided.\n"
                " Actions:\n"
                " \u2192 Please specify a time unit in the Excel sheet by entering it in the cell to the right of 'Time unit:'.\n"
                " \u2192 Alternatively, report times using Excel's time format (HH:MM:SS).\n"
            )
        elif not all(isinstance(x, (int, float)) for x in data_df.index):
            # Time unit exists, but the uploaded time format is not all float or integers  
            sys.exit(
                "\n \t \u274C Error reading file with measurements\n" 
                " Issue: Times are not numeric values, even though a 'Time unit' was provided.\n"
                " Actions:\n"
                " \u2192 Check format of the cells with the reported times are either 'General' or 'Number',"
                " when reporting the timestamps as numbers together with the time format.\n"
                " \u2192 Alternatively, report times using Excel's time format (HH:MM:SS).\n"
            )

        elif conversion_dictionary.get('Time unit') and all(isinstance(x, (int, float)) for x in data_df.index):
            return

        else:
            sys.exit(
                "\n \t \u274C Error reading file with measurements\n" 
                " Issue: Times were not reported correctly. The time unit of the measurements were not provided\n"
                " and timestamps were not readable.\n"
                " Actions:\n"
                " \u2192 Specify a time unit in the Excel sheet by entering it in the cell to the right of 'Time unit:'.\n"
                " \u2192 Check format of the cells with the reported times are either 'General' or 'Number',"
                " when reporting the timestamps as numbers together with the time format.\n"
                " \u2192 Alternatively, report times using Excel's time format (HH:MM:SS).\n"
            )