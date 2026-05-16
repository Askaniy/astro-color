from .spectral_objects import Spectrum, SpectralSet, SpectralCube
from .photospectral_objects import Photospectrum, PhotospectralSet, PhotospectralCube
from .spectral_reconstruction import ReconstructedSpectrum, ReconstructedSpectralSet, ReconstructedSpectralCube, spectral_reconstruction
from .filters import Filter, FilterSet
from .convolution import observe, scale_spectrum
from .color import ColorSystem, ColorPoint, ColorLine, ColorImage
from .physical_models import sun_CALSPEC, vega_CALSPEC, BlackBodyModel


# API namespace
__all__ = (
    'Spectrum',
    'SpectralSet',
    'SpectralCube',
    'Photospectrum',
    'PhotospectralSet',
    'PhotospectralCube',
    'ReconstructedSpectrum',
    'ReconstructedSpectralSet',
    'ReconstructedSpectralCube',
    'spectral_reconstruction',
    'Filter',
    'FilterSet',
    'observe',
    'scale_spectrum',
    'ColorSystem',
    'ColorPoint',
    'ColorLine',
    'ColorImage',
    'sun_CALSPEC',
    'vega_CALSPEC',
    'BlackBodyModel'
)
