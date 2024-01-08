from enum import Enum, auto


class Mode(Enum):
    """Mode of the program."""
    TRAIN = auto()
    TEST = auto()
    PREDICT = auto()
    DEBUG = auto()

class DatasetType(Enum):
    """Type of dataset."""
    BOP = "bop"
    LM = "lm"
    LMO = "lmo"
    YCBV = "ycbv"
    ABLATION = "lm_ablation"
    CUSTOM = "custom"


class ImageType(Enum):
    """Type of image. Real, PBR, or synthetic."""
    REAL = auto()
    PBR = auto()
    SYNTH = auto()

   

class BBoxType(Enum):
    OBJECT = auto()
    VISIBLE = auto()
    JITTER = auto() 

class MetricType(Enum):
    ANG_DIST = auto()
    EUCL_DIST = auto()
    ADD = auto()
    ADI = auto()

class AttentionType(Enum):
    pass
 
class DistType(Enum):
    NORMAL = auto()
    UNIFORM = auto()