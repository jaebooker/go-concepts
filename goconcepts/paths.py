"""Where the networks and external game records live.

Set GOCONCEPTS_NETS to the directory holding the network files (default: ./nets in the
repository root) and GOCONCEPTS_SGF to the professional-game SGF directory (default:
./data/sgf). See the README for the expected file names and where to get them.
"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NETS_DIR = os.environ.get("GOCONCEPTS_NETS", os.path.join(ROOT, "nets"))
SGF_DIR = os.environ.get("GOCONCEPTS_SGF", os.path.join(ROOT, "data", "sgf"))
