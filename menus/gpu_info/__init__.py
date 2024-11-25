"""
This is the package for the menu shown when GPU Info is selected in the main menu.
"""

# general from modules can be implemented here.
# pylint: disable=E0401

import os
from modules import general

def menu():
    """
    The menu for the GPU Info menu, which simply runs "nvidia-smi" in the terminal.
    :return:
    """
    general.clear_screen()
    os.system("nvidia-smi")
