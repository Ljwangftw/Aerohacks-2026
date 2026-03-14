"""
main.py — Entry point for the AeroHover autonomous hover system.

Run: python main.py
"""

import sys
import os

# Ensure project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.app import AeroHoverApp


def main():
    app = AeroHoverApp(config_name="default")
    app.run()


if __name__ == "__main__":
    main()
