"""decider-2b-coherent: Mapika/decider-2b + a learned coupling (and a sims-calibrated marginal adapter behind a domain
gate), answered from one coherent joint. See README.md."""
from .joint import MixtureJoint
from .modeling import BASE, BASE_REVISION, CoherentDecider, Prediction, ipf_mixture, load

__all__ = ["load", "CoherentDecider", "Prediction", "MixtureJoint", "ipf_mixture", "BASE", "BASE_REVISION"]
__version__ = "1.1.0"
