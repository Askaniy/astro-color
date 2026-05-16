import numpy as np
import numpy.typing as npt
from copy import deepcopy
from typing import Self, Any, Callable, cast

from .core import BaseObject, Item, Set, Cube
# No dependency on .spectral_objects to avoid a cycle!
from .filters import FilterSet
from .errors import nan_values_warning, InconsistentDimensionError, \
    InconsistentAxesError, InconsistentUncertaintySizeError, InconsistentUncertaintyShapeError


class PhotospectralObject(BaseObject):
    """
    Internal parent class for Photospectrum (1D), PhotospectralSet (2D) and PhotospectralCube (3D).

    Attributes:
    - `filter_set` (FilterSet): instance of the class storing filter profiles
    - `wavelength_nm` (npt.NDArray): shortcut for filter_set.wavelength_nm, the definition range
    - `spectral_dist` (npt.NDArray): array of "brightness" in energy density units (not a photon counter)
    - `covariance_matrix`: (npt.NDArray): optional matrix that stores uncertainty and its correlations
    - `name` (Any): object identifier
    """

    def __init__(self,
            filter_set: FilterSet,
            spectral_dist: npt.ArrayLike,
            uncertainty: npt.ArrayLike | None = None,
            name: Any = None
        ):
        """
        Args:
        - `filter_set` (FilterSet): instance of the class storing filter profiles
        - `spectral_dist` (ArrayLike): array of "brightness" in energy density units (not a photon counter)
        - `uncertainty` (ArrayLike): optional array of standard deviations or a covariance matrix
        - `name` (Any): object identifier
        """
        self.name = name
        # Spatial axis check
        self.spectral_dist = np.array(spectral_dist, dtype=self._spectral_dist_dtype)
        if self.ndim != self.spectral_dist.ndim:
            raise InconsistentDimensionError(self.ndim, self.spectral_dist.ndim, self.name)
        if np.any(np.isnan(self.spectral_dist)):
            self.spectral_dist = np.nan_to_num(self.spectral_dist)
            nan_values_warning('br', self.name)
        # Spectral axis check
        if not isinstance(filter_set, FilterSet):
            raise ValueError('`filter_set` argument is not a FilterSet instance')
        self.filter_set = filter_set
        if (len_filters := len(self.filter_set)) != (len_values := len(self.spectral_dist)):
            raise InconsistentAxesError(len_filters, len_values, self.name)
        # Uncertainty check
        if self.ignore_uncertainty_forCubes and self.ndim == 3:
            uncertainty = None
        if uncertainty is not None:
            uncertainty = np.array(uncertainty, dtype=self._spectral_dist_dtype)
        if self.covariance_matrix is not None and (len_error := len(self.covariance_matrix)) != len_values:
            raise InconsistentUncertaintySizeError(len_error, len_values, name)
        if uncertainty is not None:
            if uncertainty.ndim == self.spectral_dist.ndim:
                self.covariance_matrix = np.diag(uncertainty**2)
            elif uncertainty.ndim == self.spectral_dist.ndim + 1:
                self.covariance_matrix = uncertainty
            else:
                raise InconsistentUncertaintyShapeError(uncertainty.ndim, self.spectral_dist.ndim, name)

    @classmethod
    def stub(cls, name=None) -> Self:
        """ Initializes an object in case of the data problems """
        stub_filter_set = FilterSet('Generic_Bessell.B', 'Generic_Bessell.V')
        return cls(stub_filter_set, np.zeros((2, 1, 1)[:cls.ndim]), name=name)

    @property
    def wavelength_nm(self) -> npt.NDArray:
        """ Returns the definition range of the filter system """
        return self.filter_set.wavelength_nm

    def convert_from_photon_spectral_density(self):
        """
        Returns a new PhotospectralObject converted from photon spectral density
        to energy spectral density, using the fact that E = h c / λ.
        """
        if len(self.filter_set) > 1:
            profiles = self.filter_set.normalize()
            scale_factors = (profiles / profiles.wavelength_nm).integrate()
            scale_factors = cast(npt.NDArray, scale_factors)
            return self * (scale_factors / scale_factors.mean())
        else:
            return deepcopy(self)

    def convert_from_energy_spectral_density_per_frequency(self):
        """
        Returns a new PhotospectralObject converted from frequency spectral density
        to energy spectral density, using the fact that f_λ = f_ν c / λ².
        """
        if len(self.filter_set) > 1:
            profiles = self.filter_set.normalize()
            # (squaring the nm array will overflow uint16)
            scale_factors = (profiles / profiles.wavelength_nm / profiles.wavelength_nm).integrate()
            scale_factors = cast(npt.NDArray, scale_factors)
            return self * (scale_factors / scale_factors.mean())
        else:
            return deepcopy(self)

    def _determine_at_trusted_wavelengths(self, requested_wavelengths: npt.NDArray):
        """
        Directly uses the provided wavelength grid to create a new object.
        See `determine_at_wavelengths()` for the general case.
        """
        from .spectral_reconstruction import spectral_reconstruction
        res = spectral_reconstruction(self, requested_wavelengths)
        return res

    # To check:
    # Spectrum * Photospectrum = Spectrum
    # Photospectrum * Spectrum = Spectrum
    def _apply_element_wise_operation(self, operand: BaseObject, value_handling: Callable, error_handling: Callable) -> Self:
        """
        Returns a new PhotospectralObject formed from element-wise operation with
        a SpectralObject or another PhotospectralObject. Operations between objects
        of the same dimensionality and all (photo)spectrum operations are supported.

        The filter system of the second object, if it does not match, is converted
        to the filter system of the first object!
        """
        filter_set = self.filter_set
        if isinstance(operand, 'SpectralObject') or (isinstance(operand, PhotospectralObject) and operand.filter_set != filter_set):
            # Converting to a PhotospectralObject of the same filter system
            from .convolution import observe
            operand = observe(operand, filter_set)
        value = value_handling(self.spectral_dist, operand.spectral_dist)
        error = error_handling(self.spectral_dist, self.covariance_matrix, operand.spectral_dist, operand.covariance_matrix)
        higher_dim = (self, operand)[self.ndim < operand.ndim]
        return higher_dim.__class__(filter_set, value, error, name=higher_dim.name)



class Photospectrum(PhotospectralObject, Item):
    """
    Class to work with set of filters measurements (1D PhotospectralObject).

    Attributes:
    - `filter_set` (FilterSet): instance of the class storing filter profiles
    - `wavelength_nm` (npt.NDArray): shortcut for filter_set.wavelength_nm, the definition range
    - `spectral_dist` (npt.NDArray): array of "brightness" in energy density units (not a photon counter)
    - `covariance_matrix`: (npt.NDArray): optional matrix that stores uncertainty and its correlations
    - `name` (Any): object identifier
    """
    pass


class PhotospectralSet(PhotospectralObject, Set):
    """
    Class to work with set of filters measurements (2D PhotospectralObject).

    Attributes:
    - `filter_set` (FilterSet): instance of the class storing filter profiles
    - `wavelength_nm` (npt.NDArray): shortcut for filter_set.wavelength_nm, the definition range
    - `spectral_dist` (npt.NDArray): array of "brightness" in energy density units (not a photon counter)
    - `covariance_matrix`: (npt.NDArray): optional matrix that stores uncertainty and its correlations
    - `name` (Any): object identifier
    - `size` (int): spatial axis length
    """
    pass


class PhotospectralCube(PhotospectralObject, Cube):
    """
    Class to work with set of filters measurements (3D PhotospectralObject).

    Attributes:
    - `filter_set` (FilterSet): instance of the class storing filter profiles
    - `wavelength_nm` (npt.NDArray): shortcut for filter_set.wavelength_nm, the definition range
    - `spectral_dist` (npt.NDArray): array of "brightness" in energy density units (not a photon counter)
    - `covariance_matrix`: (npt.NDArray): optional matrix that stores uncertainty and its correlations
    - `name` (Any): object identifier
    - `width` (int): horizontal spatial axis length
    - `height` (int): vertical spatial axis length
    - `size` (int): number of pixels
    """

    def flatten(self):
        """ Returns a PhotospectralSet with linearized spatial axis """
        value = self.spectral_dist.reshape(self.spectral_size, self.spatial_size)
        error = None if self.covariance_matrix is None else self.covariance_matrix.reshape(self.spectral_size, self.spatial_size)
        return PhotospectralSet(self.filter_set, value, error, self.name)
