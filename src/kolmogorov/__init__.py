# Kolmogorov Detector: Object Detection via Program Synthesis
#
# Core Insight: Objects are regions with low Kolmogorov complexity.
# Instead of learning features, we search for the shortest program
# that generates each object.
#
# This is a fundamentally different paradigm from neural network detection.

from .dsl import *
from .renderer import *
from .search import *
from .detector import *
