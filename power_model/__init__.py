"""Per-die energy/power traces for PanAgretti simulations.

Pipeline: ``PowerCollector`` (a ``MetricsCollector`` subclass) bins per-device
activity counters during a simulation; ``price`` turns them into per-die power
with the coefficients in ``power.yaml``. ``agentic_sim`` never imports this
package.
"""

from .binner import ACTIVITY_KINDS, CATEGORIES, ActivityBinner, BinnedActivity
from .coefficients import PowerConfig, load_power_config
from .collector import PowerCollector
from .layout import DeviceLayout, Die
from .pricing import PowerTrace, price, write_outputs

__all__ = [
    "ACTIVITY_KINDS",
    "CATEGORIES",
    "ActivityBinner",
    "BinnedActivity",
    "DeviceLayout",
    "Die",
    "PowerCollector",
    "PowerConfig",
    "PowerTrace",
    "load_power_config",
    "price",
    "write_outputs",
]
