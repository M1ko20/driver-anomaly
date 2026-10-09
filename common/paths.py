"""Single place for the shared data root all pipeline scripts read/write under."""
import os

DATA_ROOT = os.environ.get("DRIVER_DATA_ROOT", "/home/fei/mik0542")


def p(*parts):
    return os.path.join(DATA_ROOT, *parts)
