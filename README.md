# GT activity to kinetics
Python program for calculating reaction rates from activity experiments measured with UV-vis spectrometry and analyzing enzyme kinetics. Developed as part of the study "Characterizing family 1 glycosyltransferases (GT1, UGT) by reverse glycosylation: fast determination of acceptor specificity, donor specificity, hydrolysis, and enzyme stability".   

#### Disclaimer
This code has been developed (and only used) to determine rates and kintics from UV-vis experiments with Glycosyltransferase family 1 enzymens (GT1) in combination with CNP-\beta-glycosides, this program can in principle be used in other relations. In addition, the software has primarily been tested by the developer. Feedback, bug reports, and contributions are very welcome.

## Installation

##### Prerequisites:
- Python ≥ 3.11
- Git (Optional - If you plan to clone the repository)
- Conda/Miniconda (Optional)

To run the program it is required that Python 3.11 or above is installed. If you don’t have Python, install it here: **[Download Python](https://www.python.org/downloads/)**
 
> **Windows note**, during setup, check “Add Python to PATH”.

#### 1) Getting the code

Without Git:
- On GitHub, click the green **Code** button → **Download ZIP**.
- Unzip it. You’ll get a folder called `activity_to_kinetics`.

Using Git:

In your terminal and folder that you want the code to reside:
```bash
git clone https://github.com/marola/activity_to_kinetics.git
cd activity_to_kinetics
```

#### 2) Enter the downloaded folder in terminal
Open a terminal and move to the downloaded folder
- Windows: open the folder in File Explorer → click the address bar → type `powershell` and press **Enter**.

- macOS/Linux: right-click the folder → “Open in Terminal”.

#### 3) Create a virtual environment (Optional)  
##### Windows (PowerShell)
```bash
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

##### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```
or with conda:
```bash
conda create -n activity_to_kinetics python=3.11 -y
conda activate activity_to_kinetics
```

#### 4) Install the python package
Run these **two** commands in the terminal you just opened:
```bash
python -m pip install --upgrade pip
python -m pip install .
```

Now you are good to go!

## Input File & Settings

The program expects a single Excel workbook with **three sheets**:

- **Measurements** – raw activity data per 96-well plate position (A1–H12), plus optional unit-conversion setup and optional plot titles.
- **Rate settings** – optional manual data-point ranges for initial-rate fitting (per well).
- **Kinetics settings** – model definition (Michaelis–Menten or substrate-inhibition), enzyme concentration, and which wells/initial rates to use with their substrate concentrations. *(Only required if you plan to estimate kinetics.)*

Open `data/raw/Template_data_file.xlsx` to see the expected layout.  
Recommended workflow: **copy** `Template_data_file.xlsx`, **rename** it, and **fill in** your own activity data and settings for your run(s).

---

### 1) `Measurements` sheet

**What to put here**
- **Data grid:** Paste your time-series activity data into the 96-well layout so each column is a well (e.g., `A1`, `B7`, …) and each row a time point.
- **Time:**  
  - Use either Excel time format (`HH:MM:SS`) **or** numeric timestamps.  
  - If numeric, specify a **Time unit** (e.g., `s`, `min`) in the sheet.
- **Measurement unit:** Provide the unit of your measurements (e.g., `AU` for absorbance or a concentration unit).
- **Optional conversion to concentration:** If your data are **absorbances** and you want concentrations:
  - Choose **one** method: **Standard (calibration curve)** *or* **Extinction (Lambert–Beer)**.
  - Fill the required parameters:
    - **Calibration curve:** slope, intercept, and slope unit.
    - **Extinction:** extinction coefficient *(M⁻¹·cm⁻¹)* and path length *(cm)*.
  - Pick a **Desired unit** from the dropdown (e.g., `mol/L`, `mmol/L`, `μmol/L`, `nmol/L`).
- **Plot titles (optional):** Provide a title per well to appear on the activity plots.

> **Important:** Define **only one** conversion method. Defining both calibration and extinction parameters at the same time is invalid.

---

### 2) `Rate settings` sheet (optional)

Use this if the automatic initial-rate finder isn’t satisfactory for specific wells.

- For any well (e.g., `C2`), specify **Start datapoint** and **End datapoint** (row indices in your time series) that the slope should be fitted over.
- These manual ranges are applied when you run with `-r -f` (estimate rates **and** enable manual fitting).

---

### 3) `Kinetics settings` sheet

To calculate kinetics, define how kinetics should be computed from initial rates.

- **Enzyme concentration:** Provide the value **and** the unit (e.g., `nM`, `μM`, `mM`, `M`).
- **Model:** Choose one of:
  - **Michaelis–Menten**
  - **Substrate inhibition**
- **Which data to use:** List the **Wells** to include (must match wells for which you have initial rates) and, for each, the corresponding **Substrate concentration** (with unit).
- **Plot title (optional):** A custom title for the kinetics plot.

> **Guidelines:**  
> - Provide **≥ 4** valid well–substrate pairs per model to enable a reliable fit.  
> - Well IDs must match exactly (e.g., `A1`, `B7`).  
> - Units must be consistent and valid (the tool will validate them).

---

### Quick checklist

- [ ] Data pasted into `Measurements` under correct well headers (A1–H12).  
- [ ] Time in Excel format **or** numeric with **Time unit** set.  
- [ ] **Measurement unit** provided; if converting absorbance → concentration, **only one** method filled and **Desired unit** chosen.  
- [ ] (Optional) Manual ranges in `Rate settings` for wells that need it.  
- [ ] To estimate kinetics, provide the enzyme concentration (+ unit), model choice, wells + corresponding substrate concentrations (≥ 4 pairs) in to the `Kinetics settings` sheet.

## Usage

You will run the tool from a **terminal** inside the project folder.  
> **Windows Note**: if `python` doesn’t work, try `py` instead.

### 1) Open a terminal in the project folder
- Windows: open the folder in File Explorer → click the address bar → type `powershell` → Enter.
- macOS/Linux: right-click the folder → “Open in Terminal”.

### 2) Quick check
Show the built-in help with the commands that can be run:
```bash
atk --help
```

### Commands reference

`-m, --measurements_file <path>`
  Path to the input Excel with the three sheets (`Measurements`, `Rate settings`, `Kinetics settings`). Always required.

`-w, --wells <spec>`
  Which wells to process: single wells (e.g. `-w A1`), rows/columns (`-w B` or `-w 4`), or a mix (`-w C,2,B7`).
  If omitted, all available wells with data in the `Measurements` file will be processed.

`-r, --estimate_rates`
  Estimate initial rates automatically.

`-f, --fitting`
  Use manual start/end datapoints specified in the `Rate settings` sheet (must be combined with `-r`).

`-k, --estimate_kinetics`
  Compute kinetic parameters using models definitions created in the `Kinetics settings` sheet.
  Needs initial rates (from `-s` or computed with `-r` in the same run).

`-s, --rates_input_file <path>`
  Excel with precomputed initial rates (if you don’t use `-r`).

`-q, --rates_output_file <name>`
  Output Excel filename for initial rates (saved under `-d`).

`-o, --kinetics_output_file <name>`
  Output Excel filename for kinetics results.

`-d, --output_directory <path>`  
  Where outputs (plots and Excel files) are written.  
  If omitted, the tool defaults to `<CWD>/data/processed/` and creates it automatically.  
  *(CWD = your current working directory when you run `atk`.)*

`-n, --create_output`
  Create default output files/folders even if explicit names are not provided.

`-p, --plot_activity`
  Generate and save activity plots per well.

### Usefuls commands

##### Plot activity of all wells in the `Measurements` sheet
```bash
atk -m "data/raw/<measurements_filename.xlsx>" -p
```
> **Note:** plots are saved in `/data/processed/activity_plots`

##### Plot activity for row B, column 4 and well F9 of the 96 well micro titer plate
```bash
atk -m "data/raw/<measurements_filename.xlsx>" -w "B,4,F9" -p 
```  

#### Estimating rates
##### Plot activity and estimate rates of all wells in the `Measurements` sheet of the uploaded file and save the rate results.
```bash
atk -m "data/raw/<measurements_filename.xlsx>" -p -r -n
```
> **Note:** the default directory for rate results is `/data/processed` with the default file name `rate_results.xlsx`.

##### In case you want to save results in a specific folder.
```bash
atk -m "data/raw/<measurements_filename.xlsx>" -p -r -d "<folder_name>"
```
> **Note:** the results will be saved in the `activity_to_kinetics` folder within the given folder name. To place the results in `/data/processed` repository write: `/data/processed/<folder_name>`.

The default name of results Excel files computed for rate and kinetics are `rate_results` and `kinetic_results`
##### To determine the name of the Excel file with rate results
```bash
atk -m "data/raw/<measurements_filename.xlsx>" -p -r -q "<results_file>"
```

##### To re-calculate a specific well and update the results file
```bash
atk -m "data/raw/<measurements_filename.xlsx>" -p -r -w "D12" -q "<path/to/results_file>"
```
or
```bash
atk -m "data/raw/<measurements_filename.xlsx>" -p -r -w "D12" -n
```
if the file to overwrite is the default output file `/data/processed/rate_results.xlsx`

#### Estimating kinetics
##### 