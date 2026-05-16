from typing import overload, Tuple
import numpy as np
import numpy.typing as npt
from collections.abc import Sequence

from .auxiliary import integrate
from .algebra import mul_error, mul_value
from .core import Item, Set, Cube, RealObject
from .spectral_objects import Spectrum
from .photospectral_objects import Photospectrum, PhotospectralSet, PhotospectralCube
from .filters import Filter, FilterSet


@overload
def observe(
        target: Item | Set | Cube,
        bandpass: Filter
    ) -> Tuple[float, float | None]:
    ...

@overload
def observe(
        target: Item,
        bandpass: FilterSet
    ) -> Photospectrum:
    ...

@overload
def observe(
        target: Set,
        bandpass: FilterSet
    ) -> PhotospectralSet:
    ...

@overload
def observe(
        target: Cube,
        bandpass: FilterSet
    ) -> PhotospectralCube:
    ...

def observe(
        target: RealObject,
        bandpass: Filter | FilterSet
    ):
    """
    Implementation of convolution between a (photo)spectral object and a filter or a filter set.
    Ignores the uncertainty of filter profiles.
    """
    target = target.determine_at_wavelengths(bandpass.wavelength_nm, strictly=True)
    match bandpass:
        case Filter():
            value = integrate(mul_value(target.spectral_dist, bandpass.spectral_dist), target.nm_step)
            error = mul_error(target.spectral_dist, target.covariance_matrix, bandpass.spectral_dist, None)
            if error is not None:
                error = integrate(error, target.nm_step)
            return value, error
        case FilterSet():
            value = np.einsum('ij, j... -> i...', bandpass.matrix, target.spectral_dist)
            # compare! 1D value = integrate(mul_value(target.spectral_dist, bandpass.spectral_dist), nm_step)
            # compare! 2D value = integrate(target.spectral_dist[:, :, np.newaxis] * bandpass.spectral_dist[:, np.newaxis, :], nm_step).T
            # 3D: value = np.empty((len(bandpass), *target.spatial_shape))
                # for i in range(len(bandpass)):
                #     profile = bandpass.spectral_dist[:,i]
                #     br[i] = integrate((target.spectral_dist.T * profile).T, nm_step)
            error = None
            if target.covariance_matrix is not None:
                error = np.einsum('ij, jk..., lk -> il...', bandpass.matrix, target.covariance_matrix, bandpass.matrix)
            match target:
                case Item():
                    return Photospectrum(bandpass, value, error, name=target.name)
                case Set():
                    return PhotospectralSet(bandpass, value, error, name=target.name)
                case Cube():
                    return PhotospectralCube(bandpass, value, error, name=target.name)


def scale_spectrum(
        target: Spectrum,
        bandpass: Filter | FilterSet,
        requested_value: int | float | tuple[float, float] = 1
    ) -> Spectrum:
    """
    Returns a new spectrum that matches the query brightness value (1 by default)
    at the specified filter.
    """
    current_value, std = observe(target, bandpass)
    if current_value <= 0:
        # Prevents errors of dividing by zero and inversion
        return target
    if isinstance(requested_value, Sequence):
        requested_value = requested_value[0] # likely a [value, std]
    # TODO: process std?
    return target * (requested_value / current_value)
