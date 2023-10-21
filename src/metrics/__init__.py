from .kurtosis import Kurtosis, RotatedKurtosis
from .distribution import Distribution
from .outliers import Outliers
from .topk import Top1, Top5
from .base import Metric
from .rms import RMS

METRICS = Metric.registry