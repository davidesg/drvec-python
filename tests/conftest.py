"""Run against the source tree without installing (src/ on sys.path)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "src"))
DATA = os.path.join(HERE, "data")
FIXTURES = os.path.join(DATA, "fixtures")
PRE = [os.path.join(FIXTURES, "mmpre.muskrat.pre"), os.path.join(FIXTURES, "mmpre.mink.pre")]
