from typing import Union, Dict, List, Optional, Any, Tuple

from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Union
from difflib import get_close_matches
import inspect


from utils import  ROOT, DEFAULT_CFG_DICT,  LOGGER, IterableSimpleNamespace, yaml_load, colorstr


def cfg2dict(cfg):
    """
    Convert a configuration object to a dictionary, whether it is a file path, a string, or a SimpleNamespace object.

    Inputs:
        cfg (str) or (Path) or (SimpleNamespace): Configuration object to be converted to a dictionary.

    Returns:
        cfg (dict): Configuration object in dictionary format.
    """
    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)  # load dict
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)  # convert to dict
    return cfg


def get_cfg(cfg: Union[str, Path, Dict, SimpleNamespace] = DEFAULT_CFG_DICT, overrides: Dict = None):
    """
    Load and merge configuration data from a file or dictionary.

    Args:
        cfg (str) or (Path) or (Dict) or (SimpleNamespace): Configuration data.
        overrides (str) or (Dict), optional: Overrides in the form of a file name or a dictionary. Default is None.

    Returns:
        (SimpleNamespace): Training arguments namespace.
    """
    cfg = cfg2dict(cfg)

    # Merge overrides
    if overrides:
        overrides = cfg2dict(overrides)
        #check_cfg_mismatch(cfg, overrides)
        cfg = {**cfg, **overrides}  # merge cfg and overrides dicts (prefer overrides)

    # Special handling for numeric project/name
    for k in 'project', 'name':
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = str(cfg[k])
    if cfg.get('name') == 'model':  # assign model to 'name' arg
        cfg['name'] = cfg.get('model', '').split('.')[0]
        LOGGER.warning(f"WARNING ⚠️ 'name=model' automatically updated to 'name={cfg['name']}'.")

    # Type and Value checks
#    for k, v in cfg.items():
#        if v is not None:  # None values may be from optional args
#            if k in CFG_FLOAT_KEYS and not isinstance(v, (int, float)):
#                raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
#                                f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')")
#            elif k in CFG_FRACTION_KEYS:
#                if not isinstance(v, (int, float)):
#                    raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
#                                    f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')")
#                if not (0.0 <= v <= 1.0):
#                    raise ValueError(f"'{k}={v}' is an invalid value. "
#                                     f"Valid '{k}' values are between 0.0 and 1.0.")
#            elif k in CFG_INT_KEYS and not isinstance(v, int):
#                raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
#                                f"'{k}' must be an int (i.e. '{k}=8')")
#            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
#                raise TypeError(f"'{k}={v}' is of invalid type {type(v).__name__}. "
#                                f"'{k}' must be a bool (i.e. '{k}=True' or '{k}=False')")

    # Return instance
    return IterableSimpleNamespace(**cfg)


def check_cfg_mismatch(base: Dict, custom: Dict, e=None):
    """
    This function checks for any mismatched keys between a custom configuration list and a base configuration list.
    If any mismatched keys are found, the function prints out similar keys from the base list and exits the program.

    Inputs:
        - custom (Dict): a dictionary of custom configuration options
        - base (Dict): a dictionary of base configuration options
    """
    base, custom = (set(x.keys()) for x in (base, custom))
    mismatched = [x for x in custom if x not in base]
    if mismatched:
        string = ''
        for x in mismatched:
            matches = get_close_matches(x, base)  # key list
            matches = [f'{k}={DEFAULT_CFG_DICT[k]}' if DEFAULT_CFG_DICT.get(k) is not None else k for k in matches]
            match_str = f'Similar arguments are i.e. {matches}.' if matches else ''
            string += f"'{colorstr('red', 'bold', x)}' is not a valid YOLO argument. {match_str}\n"
        raise SyntaxError(string) from e



def print_args(args: Optional[dict] = None, show_file=True, show_func=False):
    """Print function arguments (optional args dict)."""


    x = inspect.currentframe().f_back  # previous frame
    file, _, func, _, _ = inspect.getframeinfo(x)
    if args is None:  # get args automatically
        args, _, _, frm = inspect.getargvalues(x)
        args = {k: v for k, v in frm.items() if k in args}
    try:
        file = Path(file).resolve().relative_to(ROOT).with_suffix('')
    except ValueError:
        file = Path(file).stem
    s = (f'{file}: ' if show_file else '') + (f'{func}: ' if show_func else '')
    LOGGER.info(colorstr(s) + ', '.join(f'{k}={v}' for k, v in args.items()))