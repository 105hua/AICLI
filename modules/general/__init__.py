"""
General Functions for tasks that are used frequently.
"""

import os

def clear_screen():
    """
    Clears the screen by running "cls" on Windows and "clear" on Linux.
    :return:
    """
    if os.name == "nt":
        os.system("cls")
    else:
        os.system("clear")
