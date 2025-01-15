"""
This module consolidates the imports of key classes from the `data_loader.py` 
and `summarizer.py` files located in the `training` folder. By including this 
file in the package, you can simplify imports throughout the project.

Example Usage:
--------------
You can import the `SummarizationDataLoader` and `Summarizer` classes from the 
`training` package with a single import statement:
    from training import SummarizationDataLoader, Summarizer
"""

# Importing the SummarizationDataLoader class from the data_loader module
from .data_loader import SummarizationDataLoader

# Importing the Summarizer class from the summarizer module
from .summarizer import Summarizer

# Defining the public API of this module
__all__ = ["SummarizationDataLoader", "Summarizer"]
