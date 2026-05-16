import numpy as np
import numpy.typing as npt
from typing import Union, Self, cast
from uuid import uuid4

from .spectral_objects import Spectrum, SpectralSet
from .data_manager import script_folder
from .errors import FilterNotFoundError

class Filter(Spectrum):
    """
    Stores filter profile as a Spectrum object.
    Initialization by filter name in SVO Filter Profile Service ID
    or by wavelength in nanometers.
    The created object is cached (multiton pattern).
    """
    _cached_filters = {} # {filter id: filter instance}

    def __new__(cls, filter_id):
        if isinstance(filter_id, (int, float)):
            wavelength = float(filter_id)
            filter_id = f'Monochromatic {wavelength} nm'
        if not isinstance(filter_id, str):
            raise TypeError('Filter ID must be of type str')
        # Return cached object if it exists
        if filter_id in cls._cached_filters:
            return cls._cached_filters[filter_id]

        # Or, create new instance
        instance = super().__new__(cls)
        cls._cached_filters[filter_id] = instance
        return instance

    def __init__(self, filter_id: str | int | float):
        # Prevent re-initialization when retrieving from cache
        if getattr(self, '_initialized', False):
            return
        # Mark as initialized so subsequent calls for the same ID (via __new__) don't reload
        self._initialized = True
        # 1. Try monochromatic (if ID is a wavelength)
        try:
            wavelength = float(filter_id)
            self.name = f'Monochromatic {wavelength} nm'
            spectrum = self.monochromatic(wavelength)
            self.wavelength_nm = spectrum.wavelength_nm
            self.spectral_dist = spectrum.spectral_dist
        except ValueError:
            # 2. Try loading from file
            try:
                file_path = next((script_folder/'filters').glob(f'{filter_id}.*'))
                data = np.loadtxt(file_path).T
                if str(file_path)[-1] == 'A':
                    data[0] /= 10 # temporal workaround to convert angstrom to nm
                    # TODO: delete after SVO Filter Profile Service support is ready!
                spectrum = Spectrum(*data, name=filter_id).edges_zeroed().normalize()
                self.name = filter_id
                self.wavelength_nm = spectrum.wavelength_nm
                self.spectral_dist = spectrum.spectral_dist
            except (StopIteration, FileNotFoundError):
                raise FilterNotFoundError(filter_id)

    @classmethod
    def from_spectrum(cls, spectrum: Spectrum) -> Self:
        """
        Converts Spectrum to a Filter object.
        It allows to convolve a spectrum with another spectrum.
        It is needed, for example, to calculate Bond albedo using Solar spectrum as filter.
        """
        # Hash generator for filter_id is better spectrum.name because
        # there is no guarantee that spectra with the same names are identical
        profile = cls.__new__(cls, filter_id=uuid4())
        profile._initialized = True
        profile.__dict__.update(spectrum.__dict__)
        return profile.edges_zeroed().normalize()

    def _determine_at_trusted_wavelengths(self, requested_wavelengths: npt.NDArray):
        raise NotImplementedError('It is not possible to change the spectral axis for Filters.')

    def __repr__(self):
        return f'{self.__class__.__name__}({self.name!r})'

    # Convolution (moved to convolution.py)
    # def __rmatmul__(self, other: BaseObject) -> tuple[float, float | None]:
    #     operand1 = other.determine_at_wavelengths(self.wavelength_nm, strictly=True)
    #     operand2 = self
    #     br = integrate(self._mul_value(operand1.spectral_dist, operand2.spectral_dist), self.nm_step)
    #     std = self._mul_error(operand1.spectral_dist, operand1.std, operand2.spectral_dist, operand2.std)
    #     if std is not None:
    #         std = integrate(std, self.nm_step)
    #     return br, std


class FilterSet(SpectralSet):
    """
    Class to work with a set of filters profiles.
    It supports len() to get the number of filters and getitem() to get a profile.

    Example:
    `bvr = FilterSet('Generic_Bessell.B', 'Generic_Bessell.V', 'Generic_Bessell.R')`

    Attributes:
    - `wavelength_nm` (npt.NDArray): total wavelength range of the filter profiles
    - `spectral_dist` (npt.NDArray): matrix of the profiles with shape [len(nm), len(filters)]
    - `size` (int): spatial axis length
    """
    spectral_dist_cache = None
    wavelength_nm_cache = None

    def __init__(self, *filter_ids):
        self.filters: tuple = filter_ids

    @property
    def wavelength_nm(self):
        if self.spectral_dist_cache is None:
            nm_min = np.inf
            nm_max = 0
            for profile in self:
                nm_min = min(nm_min, profile.wavelength_nm[0])
                nm_max = max(nm_max, profile.wavelength_nm[-1])
            nm = self._grid(nm_min, nm_max)
            self.wavelength_nm_cache = nm
        else:
            nm = self.wavelength_nm_cache
        return nm

    @property
    def spectral_dist(self):
        if self.spectral_dist_cache is None:
            # Matrix packing
            nm = cast(npt.NDArray, self.wavelength_nm)
            br = np.zeros((len(nm), len(self)))
            for i, profile in enumerate(self):
                br[np.where((nm >= profile.wavelength_nm[0]) & (nm <= profile.wavelength_nm[-1])), i] = profile.spectral_dist
            self.spectral_dist_cache = br
        else:
            br = self.spectral_dist_cache
        return br

    @property
    def matrix(self):
        """
        Transforms filter profiles' transmission spectral distribution
        into a matrix that simplifies the accounting of the grid step:
        A = T^T * Δλ  ->  y = A x, where x is a spectrum and y is a photospectrum
        """
        return self.spectral_dist.T * self.nm_step

    def _determine_at_trusted_wavelengths(self, requested_wavelengths: npt.NDArray):
        raise NotImplementedError('It is not possible to change the spectral axis for FilterSets.')

    def __len__(self):
        return len(self.filters)

    def __iter__(self):
        """ Creates an iterator over the filters in the system """
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index: int | slice) -> Union[Filter, 'FilterSet']:
        """ Returns the Filter or a subset of filters """
        if isinstance(index, int):
            return Filter(self.filters[index])
        elif isinstance(index, slice):
            new_filters = self.filters[index]
            return FilterSet(*new_filters)
        else:
            raise TypeError(f"Index must be int or slice, not {type(index).__name__}")


    # Convolution (moved to convolution.py)

    # @singledispatchmethod
    # def __rmatmul__(self, other):
    #     raise NotImplementedError

    # @__rmatmul__.register
    # def _(self, other: Item) -> Photospectrum:
    #     operand1 = other.determine_at_wavelengths(self.wavelength_nm, strictly=True)
    #     operand2 = self
    #     br = integrate(self._mul_value(operand1.spectral_dist, operand2.spectral_dist), self.nm_step)
    #     std = self._mul_error(operand1.spectral_dist, operand1.std, operand2.spectral_dist, operand2.std)
    #     if std is not None:
    #         std = integrate(std, self.nm_step)
    #     return Photospectrum(operand2, br, std, name=operand1.name)

    # @__rmatmul__.register
    # def _(self, other: Set) -> PhotospectralSet:
    #     operand1 = other.determine_at_wavelengths(self.wavelength_nm, strictly=True)
    #     operand2 = self
    #     br = integrate(operand1.spectral_dist[:, :, np.newaxis] * operand2.spectral_dist[:, np.newaxis, :], self.nm_step).T
    #     # TODO: uncertainty processing
    #     return PhotospectralSet(operand2, br, name=operand1.name)

    # @__rmatmul__.register
    # def _(self, other: Cube) -> PhotospectralCube:
    #     operand1 = other.determine_at_wavelengths(self.wavelength_nm, strictly=True)
    #     operand2 = self
    #     # A loop-less implementation would require a 4D array,
    #     # which most computers do not have enough memory for.
    #     br = np.empty((len(operand2), *operand1.spatial_shape))
    #     #for i, profile in enumerate(operand2):
    #     for i in range(len(operand2)):
    #         profile = operand2.spectral_dist[:,i]
    #         br[i] = integrate((operand1.spectral_dist.T * profile).T, self.nm_step)
    #     # TODO: uncertainty processing
    #     return PhotospectralCube(operand2, br, name=operand1.name)
