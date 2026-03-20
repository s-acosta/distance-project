import numpy as np
from scipy.io import loadmat
from scipy.io.matlab import MatlabOpaque


def clean_matlab_struct(val):
    """Recursively convert a MATLAB structured array to a nested Python dict.

    MatlabOpaque fields (MATLAB string objects) cannot be decoded by scipy
    and will be returned as None.
    """
    if isinstance(val, MatlabOpaque):
        return None
    elif isinstance(val, np.ndarray):
        if val.dtype.names:  # structured array (MATLAB struct)
            result = {}
            for name in val.dtype.names:
                field = val[name]
                if field.shape == ():
                    field = field.item()
                result[name] = clean_matlab_struct(field)
            return result
        elif val.dtype == object:  # cell array
            return [clean_matlab_struct(v) for v in val]
        elif val.shape == ():
            return val.item()
        else:
            return val
    elif isinstance(val, tuple):
        return tuple(clean_matlab_struct(v) for v in val)
    else:
        return val


def load_mat_struct(path, squeeze_me=True):
    """Load a .mat file and return a cleaned dict of all top-level variables."""
    raw = loadmat(str(path), squeeze_me=squeeze_me)
    data = {k: clean_matlab_struct(v) for k, v in raw.items() if not k.startswith("__")}
    return data if len(data) > 1 else next(iter(data.values()))
