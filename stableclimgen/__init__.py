__version__ = "0.1.0"

from .src.train import train
from .src.test_healpix import test

__all__ = ['train', 'test']
