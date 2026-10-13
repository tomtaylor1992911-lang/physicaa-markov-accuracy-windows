"""Compatibility entry point for the revised NumPy-only numerical workflow.

Run python code/generate_figures.py to regenerate the revised data and figures.
This entry point runs both the baseline and additional analyses.
"""
from revision_numerics import main as base_main
from polish_analysis import main as polish_main

if __name__ == "__main__":
    base_main()
    polish_main()
