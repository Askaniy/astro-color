import numpy as np
import numpy.typing as npt
from collections.abc import Callable
from typing import Any, Generic, TypeVar
from scipy.linalg import solve
from scipy.optimize import minimize
from copy import deepcopy

from .auxiliary import smoothness_matrix
from .core import Item, Set, Cube
from .spectral_objects import SpectralObject, Spectrum, SpectralSet, SpectralCube
from .photospectral_objects import PhotospectralObject, Photospectrum, PhotospectralSet, PhotospectralCube
from .errors import UnsupportedDimensionError


# For type checkers, this type specifies that each reconstructed class
# can only store hyperspectral objects of its own dimension.
PhotospectralType = TypeVar('PhotospectralType', bound='PhotospectralObject')


class ReconstructedSpectralObject(SpectralObject, Generic[PhotospectralType]):

    def __init__(self,
            wavelength_nm: npt.ArrayLike,
            spectral_dist: npt.ArrayLike,
            uncertainty: npt.ArrayLike | None = None,
            name: Any = None,
            photospectral_obj: PhotospectralType | None = None
        ):

        """
        Creates a ReconstructedSpectralObject from arrays of wavelength, brightness and (optionally) uncertainty.
        Performs checks for data type and uniformity; interpolates and extrapolates if it is needed.

        Args:
        - `wavelength_nm` (ArrayLike): list of wavelengths in nanometers on an arbitrary grid
        - `spectral_dist` (ArrayLike): array of "brightness" in energy density units (not a photon counter)
        - `uncertainty`: (ArrayLike): optional array of standard deviations or covariance matrix
        - `name` (Any): object identifier
        - `is_emission_spectrum` (bool): if `True`, creates an emission spectral object from the spectral lines
        - `photospectral_obj` (PhotospectralObject): optional, a way to store the pre-reconstructed data
        """
        super().__init__(wavelength_nm, spectral_dist, uncertainty, name)
        self.photospectral_obj = photospectral_obj

    def _determine_at_trusted_wavelengths(self, requested_wavelengths: npt.NDArray):
        """
        Directly uses the provided wavelength grid to create a new object.
        See `determine_at_wavelengths()` for the general case.
        """
        if self.photospectral_obj is None:
            extrapolated = super()._determine_at_trusted_wavelengths(requested_wavelengths)
        else:
            # Repeating the spectral reconstruction on the new wavelength range
            extrapolated = self.photospectral_obj._determine_at_trusted_wavelengths(requested_wavelengths)
        return extrapolated

    def _apply_scalar_operation(self, operand, value_handling: Callable, error_handling: Callable):
        """
        Returns a new object of the same class transformed according to the linear operator.
        Operand is assumed to be a number or an array along the spectral axis.
        Linearity is needed because values and uncertainty are handled uniformly.
        """
        output = super()._apply_scalar_operation(operand, value_handling, error_handling)
        if self.photospectral_obj is not None:
            output.photospectral_obj = self.photospectral_obj._apply_scalar_operation(operand, value_handling, error_handling)
        return output


class ReconstructedSpectrum(ReconstructedSpectralObject[Photospectrum], Item):
    """
    Class to work with a single reconstructed spectrum (1D SpectralObject).

    Attributes:
    - `wavelength_nm` (npt.NDArray): spectral axis, list of wavelengths in nanometers on a uniform grid
    - `spectral_dist` (npt.NDArray): array of "brightness" in energy density units (not a photon counter)
    - `covariance_matrix`: (npt.NDArray): optional matrix that stores uncertainty and its correlations
    - `name` (Any): object identifier
    - `photospectral_obj` (PhotospectralObject): optional, a way to store the pre-reconstructed data
    """


class ReconstructedSpectralSet(ReconstructedSpectralObject[PhotospectralSet], Set):
    """
    Class to work with a line of continuous spectra (2D SpectralObject).

    Attributes:
    - `wavelength_nm` (npt.NDArray): spectral axis, list of wavelengths in nanometers on a uniform grid
    - `spectral_dist` (npt.NDArray): array of "brightness" in energy density units (not a photon counter)
    - `covariance_matrix`: (npt.NDArray): optional matrix that stores uncertainty and its correlations
    - `name` (Any): object identifier
    - `photospectral_obj` (PhotospectralObject): optional, a way to store the pre-reconstructed data
    - `size` (int): spatial axis length
    """


class ReconstructedSpectralCube(ReconstructedSpectralObject[PhotospectralCube], Cube):
    """
    Class to work with an image of continuous spectra (3D SpectralObject).

    Attributes:
    - `wavelength_nm` (npt.NDArray): spectral axis, list of wavelengths in nanometers on a uniform grid
    - `spectral_dist` (npt.NDArray): array of "brightness" in energy density units (not a photon counter)
    - `covariance_matrix`: (npt.NDArray): optional matrix that stores uncertainty and its correlations
    - `name` (Any): object identifier
    - `photospectral_obj` (PhotospectralObject): optional, a way to store the pre-reconstructed data
    - `width` (int): horizontal spatial axis length
    - `height` (int): vertical spatial axis length
    - `size` (int): number of pixels
    """

    def flatten(self):
        """ Returns a SpectralSet with linearized spatial axis """
        output = super().flatten()
        if self.photospectral_obj is not None:
            output.photospectral_obj = self.photospectral_obj.flatten()
        return output


def spectral_reconstruction(
        photospectral_obj: PhotospectralObject,
        requested_wavelengths: npt.ArrayLike,
        spectral_reconstruction_mode: str = '',
        attach_photospectral_obj: bool = True
    ) -> SpectralObject | ReconstructedSpectralObject:
    """
    Reconstructs a SpectralObject from photospectral data on the wavelength array.

    Interpolation is not used because it is not a solution to the inverse ill-posed problem
    (i.e., looking at the spectrum through the filters does not give exactly the original photospectral_obj).

    The function uses the Tikhonov regularization method, with a combination of first-order
    and second-order differential operators for the Tikhonov matrix.
    That is, it tries to minimize height variations and curvature in the spectrum.

    Confidence bands for spectral sets and cubes are not computed by default.
    """
    br0 = photospectral_obj.spectral_dist
    cov0 = None if photospectral_obj.ignore_uncertainty_forCubes and photospectral_obj.ndim == 3 else photospectral_obj.covariance_matrix
    cov1 = None
    if len(photospectral_obj.filter_set) == 1:
        # single-point PhotospectralObject support
        nm_min, nm_max = photospectral_obj._get_extremal_grid_endpoints(requested_wavelengths)
        nm1 = photospectral_obj._grid(nm_min, nm_max)
        br1 = np.full((nm1.size, 1, 1)[:photospectral_obj.ndim], br0) # not tested
    else:
        filter_set = photospectral_obj.filter_set
        nm1 = filter_set.wavelength_nm
        filter_matrix = filter_set.matrix
        #L = smoothness_matrix(T.shape[1], order=2)
        #A = filter_matrix.T @ filter_matrix + 0.05 * L.T @ L
        order1_matrix = smoothness_matrix(filter_matrix.shape[1], order=1)
        order2_matrix = smoothness_matrix(filter_matrix.shape[1], order=2)
        # TODO: research on some known spectra to find which ratios (0.005, 1) fit best
        alpha = 0.005
        beta = 1
        tikhonov_matrix_covar = alpha * order1_matrix.T @ order1_matrix + beta * order2_matrix.T @ order2_matrix
        right_matrix = filter_matrix.T @ filter_matrix + tikhonov_matrix_covar
        if photospectral_obj.ndim == 3:
            # scipy supports batch mode for 2d arrays, but not for 3D arrays
            br0 = br0.reshape(filter_matrix.shape[0], -1)
        left_vector = filter_matrix.T @ br0
        br1 = solve(right_matrix, left_vector) # x1.5 faster than np.linalg.inv(A) @ b
        if photospectral_obj.ndim == 3:
            # Reshape spectral cube back from square
            br1 = br1.reshape(-1, *photospectral_obj.spectral_dist.shape[1:])
        if photospectral_obj.ndim == 1 and br1.min() < 0:
            # To avoid negative spectra, a lower bound is set and iterative
            # optimization is performed using quadratic programming methods.
            # The processing speed drops by a factor of about five,
            # so the use is blocked for spectral squares and cubes:
            # background noise near zero can be most of the pixels.
            # TODO: RECHECK! MAY CONTAIN ERRORS!
            def objective(vector):
                # Tikhonov-regularized quadratic objective: 0.5 * Y^T A Y - b^T Y
                return 0.5 * vector @ right_matrix @ vector - left_vector @ vector
            def gradient(vector):
                # Gradient of the objective
                return right_matrix @ vector - left_vector
            bounds = ((0, None) for _ in range(right_matrix.shape[1]))
            result = minimize(
                fun=objective,
                x0=np.maximum(br1, 0),
                jac=gradient,
                bounds=bounds,
                method='L-BFGS-B',
            )
            if not result.success:
                raise ValueError(f'Optimization failed: {result.message}')
            br1 = result.x
        if photospectral_obj.ndim == 1 and cov0 is not None:
            # Measurement confidence band calculation
            # Confidence bands for spectral squares and cubes are not computed to save computational resources
            right_matrix_inv = np.linalg.inv(right_matrix)
            cov1 = right_matrix_inv @ filter_matrix.T @ cov0 @ filter_matrix @ right_matrix_inv
            # TODO: write the result covariance matrix to the class instance! Uncertainty of spectrum is self-correlated
            # I mean, it's obvious that e.g. 555 nm and 560 nm data points depend on each other:
            # if a spectrum is smooth, their values can't be too different, even if their std allows one to go up and the other to go down
            # that means uncertainty of a spectrum of N data points should be described by NxN covariance matrix
            # instead of N length std array (which is just a root of diagonal of that matrix)
            #std1 = np.sqrt(np.diag(cov1))
            # An attempt to account for the sensitivity confidence band of the method
            #std1 = np.sqrt(std1**2 + (0.01 * np.median(br1))**2 * np.diag(right_matrix_inv))
            # TODO: needs research, the reconstructed std scale `0.01` is arbitrary!
    if attach_photospectral_obj:
        match photospectral_obj:
            # An implementation suitable for type checking
            case Photospectrum():
                spectral_obj = ReconstructedSpectrum(
                    nm1, br1, cov1,
                    name=photospectral_obj.name,
                    photospectral_obj=deepcopy(photospectral_obj)
                )
            case PhotospectralSet():
                spectral_obj = ReconstructedSpectralSet(
                    nm1, br1, cov1,
                    name=photospectral_obj.name,
                    photospectral_obj=deepcopy(photospectral_obj)
                )
            case PhotospectralCube():
                spectral_obj = ReconstructedSpectralCube(
                    nm1, br1, cov1,
                    name=photospectral_obj.name,
                    photospectral_obj=deepcopy(photospectral_obj)
                )
            case _:
                raise ValueError(f'For {photospectral_obj.name} to be reconstructed, it must be of class Photospectrum, PhotospectralSet or PhotospectralCube')
    else:
        try:
            target_class = (Spectrum, SpectralSet, SpectralCube)[photospectral_obj.ndim - 1]
        except IndexError:
            raise UnsupportedDimensionError(photospectral_obj.ndim, photospectral_obj.name)
        spectral_obj = target_class(nm1, br1, cov1, name=photospectral_obj.name)
    return spectral_obj
