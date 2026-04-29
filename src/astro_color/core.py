import numpy as np
import numpy.typing as npt
from math import prod
from collections.abc import Callable
from typing import Self, Any, Final, ClassVar, TypeAlias
from copy import deepcopy

from .auxiliary import repr_generator, spatial_downscaling
from .algebra import add_value, add_error, sub_value, sub_error, mul_value, mul_error, div_value, div_error
from .errors import InconsistentAxesError, InconsistentUncertaintySizeError


class BaseObject:
    """ Internal class for inheriting spectral data properties """
    wavelength_nm: npt.NDArray = NotImplemented # the own spectral axis or the wavelength range of the filter set
    spectral_dist: npt.NDArray = NotImplemented
    covariance_matrix: npt.NDArray | None = None
    name: Any = None

    ndim: ClassVar[int] = NotImplemented

    # For the sake of simplifying work with the spectrum,
    # its discretization step is fixed and frozen.
    nm_step: Final[int] = 5 # nm

    # Maximum wavelength, the clipping level
    nm_red_limit: Final[int] = 65535 # nm
    # it is possible to set the red limit to 327 675 nm
    # with compression by 5 nm step, but it was not implemented

    # When processing images through spectral cubes, performance is prioritized,
    # and uncertainty is not saved (yet). Therefore it is disabled by default.
    ignore_uncertainty_forCubes = True

    # Wavelength and brightness axis storage data type
    _wavelength_nm_dtype: Final = np.uint16
    _spectral_dist_dtype: Final = np.float64

    @property
    def spectral_size(self):
        """ Returns the spectral axis length """
        return self.spectral_dist.shape[0]

    @property
    def spatial_size(self):
        """ Returns the total number of (photo)spectra stored in the object """
        return prod(self.spatial_shape)

    @property
    def spatial_shape(self):
        """ Returns the spatial axes shape: length of the set or (width, height) """
        return self.spectral_dist.shape[1:]

    @property
    def standard_deviation(self):
        """ Calculates an array of standard deviations from the covariance matrix """
        if self.covariance_matrix is None:
            return None
        else:
            # TODO: support for sets and cubes
            return np.sqrt(np.diag(self.covariance_matrix))

    @classmethod
    def stub(cls, name: Any = None) -> Self:
        """ Initializes an object in case of the data problems """
        raise NotImplementedError('Implemented in the inherited classes.')

    def _get_extremal_grid_endpoints(self, requested_wavelengths: npt.ArrayLike):
        """
        Wavelength grid generation pipeline.
        Getting the minimum and maximum values of an untrusted array.
        """
        if isinstance(requested_wavelengths, np.ndarray):
            nm_min = requested_wavelengths.min()
            nm_max = requested_wavelengths.max()
        else:
            nm_min = np.min(requested_wavelengths)
            nm_max = np.max(requested_wavelengths)
        nm_min = max(nm_min, 0)
        nm_max = min(nm_max, self.nm_red_limit)
        return nm_min, nm_max

    def _grid_endpoints_preprocessing(self, start: int | float, end: int | float) -> tuple[int, int]:
        """
        Wavelength grid generation pipeline.
        Maps the endpoints to a standard grid (wavelengths are multiples of the grid step).
        """
        if (shift := start % self.nm_step) != 0:
            start += self.nm_step - shift
        if end % self.nm_step == 0:
            end += self.nm_step # to include the last point
        return int(start), int(end)

    def _grid(self, start: int | float, end: int | float):
        """
        Wavelength grid generation pipeline.
        Returns a uniform grid array with the points being multiples of the grid step (endpoints included)
        """
        start, end = self._grid_endpoints_preprocessing(start, end)
        return np.arange(start, end, self.nm_step, dtype=self._wavelength_nm_dtype)

    def determine_at_wavelengths(self, requested_wavelengths: npt.ArrayLike, strictly: bool = False):
        """
        Returns a new SpectralObject, guaranteeing that the specified wavelength range
        has been determined or reconstructed for it.
        If `strictly=True`, then the new object is defined exclusively
        on the specified wavelength range.
        Only the minimum and maximum wavelengths are extracted from the specified range,
        based on which a uniform grid is constructed.

        Example: `spectrum = photospectrum.determine_at_wavelengths([400, 700])`
        """
        nm_min, nm_max = self._get_extremal_grid_endpoints(requested_wavelengths)
        requested_wavelengths = self._grid(nm_min, nm_max)
        spectral_obj = self._determine_at_trusted_wavelengths(requested_wavelengths)
        # Spectral range clipping
        if strictly and spectral_obj.wavelength_nm is not requested_wavelengths:
            spectral_obj.spectral_dist = spectral_obj.get_spectral_dist_at_wavelengths(nm_min, nm_max)
            spectral_obj.covariance_matrix = spectral_obj.get_covariance_matrix_at_wavelengths(nm_min, nm_max)
            spectral_obj.wavelength_nm = requested_wavelengths
        # Sanity checks
        if (len_nm := spectral_obj.wavelength_nm.size) != (len_values := len(spectral_obj.spectral_dist)):
            raise InconsistentAxesError(len_nm, len_values, spectral_obj.name)
        if spectral_obj.covariance_matrix is not None and (len_error := len(spectral_obj.covariance_matrix)) != len_nm:
            raise InconsistentUncertaintySizeError(len_error, len_values, spectral_obj.name)
        return spectral_obj

    def _determine_at_trusted_wavelengths(self, requested_wavelengths: npt.NDArray):
        """
        Directly uses the provided wavelength grid to create a new object.
        See `determine_at_wavelengths()` for the general case.
        """
        raise NotImplementedError('Implemented in the inherited classes.')

    def convert_from_photon_spectral_density(self):
        """
        Returns a new BaseObject converted from photon spectral density
        to energy spectral density, using the fact that E = h c / λ.
        """
        raise NotImplementedError('Implemented in the inherited classes.')

    def convert_from_energy_spectral_density_per_frequency(self):
        """
        Returns a new BaseObject converted from frequency spectral density
        to energy spectral density, using the fact that f_λ = f_ν c / λ².
        """
        raise NotImplementedError('Implemented in the inherited classes.')

    def _apply_element_wise_operation(self, operand: 'BaseObject', value_handling: Callable, error_handling: Callable) -> Self:
        """ Returns a new object formed from element-wise operation """
        raise NotImplementedError('Implemented in the inherited classes.')

    def _apply_scalar_operation(self, operand: npt.ArrayLike, value_handling: Callable, error_handling: Callable) -> Self:
        """ Returns a new object of the same class transformed according to the operator """
        output = deepcopy(self)
        output.spectral_dist = value_handling(self.spectral_dist, operand)
        output.covariance_matrix = error_handling(self.spectral_dist, self.covariance_matrix, operand, None)
        return output

    def __add__(self, other) -> Self:
        if isinstance(other, BaseObject):
            return self._apply_element_wise_operation(other, add_value, add_error)
        else:
            return self._apply_scalar_operation(other, add_value, add_error)

    def __sub__(self, other) -> Self:
        if isinstance(other, BaseObject):
            return self._apply_element_wise_operation(other, sub_value, sub_error)
        else:
            return self._apply_scalar_operation(other, sub_value, sub_error)

    def __mul__(self, other) -> Self:
        if isinstance(other, BaseObject):
            return self._apply_element_wise_operation(other, mul_value, mul_error)
        else:
            return self._apply_scalar_operation(other, mul_value, mul_error)

    def __truediv__(self, other) -> Self:
        if isinstance(other, BaseObject):
            return self._apply_element_wise_operation(other, div_value, div_error)
        else:
            return self._apply_scalar_operation(other, div_value, div_error)

    def __hash__(self) -> int:
        """ Returns the hash value based on the object's name """
        return hash(self.name)

    def __eq__(self, other) -> bool:
        """ Checks equality with another BaseObject instance """
        if isinstance(other, BaseObject):
            return np.array_equal(self.wavelength_nm, other.wavelength_nm) and np.array_equal(self.spectral_dist, other.spectral_dist)
        return False

    def __repr__(self):
        output = f'{self.__class__.__name__}('
        output += f'\n\twavelength_nm = [{repr_generator(self.wavelength_nm, is_int=True)}],'
        if self.ndim == 1:
            output += f'\n\tspectral_dist = [{repr_generator(self.spectral_dist)}], '
        elif self.ndim == 2:
            if self.spatial_shape[0] == 1:
                f'\tspectral_dist = [[{self.spectral_dist[0]:.3f}, {self.spectral_dist[1]:.3f}, ..., {self.spectral_dist[-1]:.3f}]],'
        return output + '\n)'



class Item(BaseObject):
    """ Internal class for inheriting spatial data properties (1D) """

    ndim: ClassVar[int] = 1


class Set(BaseObject):
    """ Internal class for inheriting spatial data properties (2D) """

    ndim: ClassVar[int] = 2

    def __len__(self) -> int:
        """ Returns the spatial axis length (alias for .spatial_size) """
        return self.spatial_size

    def __getitem__(self, item: slice):
        """ Returns the spatial axis slice """
        if isinstance(item, slice):
            output = deepcopy(self)
            output.spectral_dist = output.spectral_dist[:,item]
            if output.covariance_matrix is not None:
                output.covariance_matrix = output.covariance_matrix[:,:,item]
            return output


class Cube(BaseObject):
    """ Internal class for inheriting spatial data properties (3D) """

    ndim: ClassVar[int] = 3

    def downscale(self, pixels_limit: int):
        """ Brings the spatial resolution of the cube to approximately match the number of pixels """
        output = deepcopy(self)
        output.spectral_dist, output.covariance_matrix = \
            spatial_downscaling(output.spectral_dist, output.covariance_matrix, pixels_limit)
        return output

    def flatten(self):
        """ Returns a (photo)spectral set with linearized spatial axis """
        raise NotImplementedError('Implemented in the inherited classes.')

    @property
    def width(self):
        """ Returns horizontal spatial axis length """
        return self.spatial_shape[0]

    @property
    def height(self):
        """ Returns vertical spatial axis length """
        return self.spatial_shape[1]


RealObject: TypeAlias = Item | Set | Cube
