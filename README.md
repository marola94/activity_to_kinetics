# activity_to_kinetics
Python script for calculating reaction rates from activity experiments and analyzing enzyme kinetics. Developed as part of the study "Characterizing family 1 glycosyltransferases (GT1, UGT) by reverse glycosylation: fast determination of acceptor specificity, donor specificity, hydrolysis, and enzyme stability".   

#### Disclaimer
While this code has been developed (and only used) to determine rates and kintics from UV-vis experiments with Glycosyltransferase family 1 enzymens (GT1) in combination with CNP-\beta-glycosides, this program can in principle be used in other relations. In addition, the software has primarily been tested by the developer. Feedback, bug reports, and contributions are very welcome.

## Installation

Prerequisites:
\bullet Python ≥ 3.11
\bullet Git (Optional - If you plan to clone the repository)
\bullet Conda/Miniconda (Optional)

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
## User instructions



