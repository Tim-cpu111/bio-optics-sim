# src/biooptics/mc/__init__.py
import warnings

def simulate_with_scattering(*args, **kwargs):
    warnings.warn(
        "biooptics.mc.simulate_with_scattering is deprecated; "
        "use biooptics.simulation.s1_semif.simulate_with_scattering instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from ..simulation.s1_seminf import simulate_with_scattering as _f
    return _f(*args, **kwargs)
