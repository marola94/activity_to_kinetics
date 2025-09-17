import pint
ureg = pint.UnitRegistry()
ureg.default_format = "~P"



# --- Absorbance definition --
ureg.define("absorbance_unit = [] = AU")



# --- Molarity ---
ureg.define("molar = mole / liter = M")
ureg.define("millimolar = 1e-3 * molar = mM")
ureg.define("micromolar = 1e-6 * molar = µM = uM")
ureg.define("nanomolar  = 1e-9 * molar = nM")

pint.set_application_registry(ureg)

# --- module level mapping (lowercase keys) ---
UNIT_MAP_CONC = {
    "mol/l":  "M",
    "mmol/l": "mM",
    "µmol/l": "uM",   # use 'uM' alias; Pint will *display* µM with ~P
    "umol/l": "uM",
    "nmol/l": "nM",
    # already-canonical forms allowed too (after lowercasing)
    "m": "M", "mm": "mM", "µm": "uM", "um": "uM", "nm": "nM",
}

def map_concentration_unit(unit_str: str) -> str:
    """
    Map an Excel concentration unit to a canonical alias: 'M', 'mM', 'uM', or 'nM'.
    Raises ValueError if the input cannot be mapped.
    This function performs *mapping only* (no Pint parsing/validation).
    """
    # normalize: trim, remove spaces, unify micro symbol, lowercase
    key = (unit_str or "").strip().replace(" ", "")
    key = key.replace("μ", "µ").lower()  # normalize Greek mu variants and lowercase

    alias = UNIT_MAP_CONC.get(key)

    return alias