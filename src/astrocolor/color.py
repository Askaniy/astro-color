import numpy as np
import numpy.typing as npt
from copy import deepcopy
from typing import Self

from .auxiliary import spatial_downscaling
from .core import RealObject
from .spectral_objects import Spectrum
from .filters import FilterSet
from .physical_models import sun_CALSPEC, vega_CALSPEC
from .convolution import observe


# CIE 1931 XYZ color matching functions, 2-deg
# https://cie.co.at/datatable/cie-1931-colour-matching-functions-2-degree-observer
# http://www.cvrl.org/cie.htm
xyz_cmf = FilterSet('CIE_1931_2deg.x', 'CIE_1931_2deg.y', 'CIE_1931_2deg.z')
visible_range = xyz_cmf.wavelength_nm # original CMF definition range is 360-830 nm

# There are CMFs transformed from the CIE (2006) LMS functions, 2-deg
# (https://cie.co.at/datatable/cie-2006-lms-cone-fundamentals-2-field-size-terms-energy)
# here: http://www.cvrl.org/database/text/cienewxyz/cie2012xyz2.htm, http://www.cvrl.org/ciexyzpr.htm
# However, the CIE 1931 XYZ standard is still widely used.

# Bradford chromatic adaptation matrices
# http://www.brucelindbloom.com/Eqn_ChromAdapt.html
matrix_B = np.array((
    ( 0.8951000,  0.2664000, -0.1614000),
    (-0.7502000,  1.7135000,  0.0367000),
    ( 0.0389000, -0.0685000,  1.0296000),
))
matrix_B_inv = np.array((
    ( 0.9869929, -0.1470543,  0.1599627),
    ( 0.4323053,  0.5183603,  0.0492912),
    (-0.0085287,  0.0400428,  0.9684867),
))


class ColorSystem:
    """
    This class builds and stores RGB to XYZ (and inverse) transformation matrices.
    The implementation is based on
    http://brucelindbloom.com/Eqn_RGB_XYZ_Matrix.html,
    http://www.brucelindbloom.com/Eqn_ChromAdapt.html,
    and https://scipython.com/blog/converting-a-spectrum-to-a-colour/.
    The formula derivation is simplified and is given in the (unpublished) paper.
    """

    def __init__(self, color_space: str, adaptation_white_point: str = ''):
        """
        Initialize the ColorSystem object.
        The color space and optional white point of the chromatic adaptation
        must be among the supported options.
        """
        # Save input names
        self.color_space_name = color_space
        self.white_point_name = adaptation_white_point
        # Reading chromaticity coordinates of the primary colors and the white point
        matrix_M, internal_white_point_name = self.supported_color_spaces[color_space]
        matrix_M = np.array(matrix_M).T
        vector_WPi = self.supported_white_points[internal_white_point_name]
        # Converting reduced chromaticity coordinates (x, y) to (x, y, z=1-x-y)
        matrix_M = np.vstack((matrix_M, 1 - matrix_M.sum(axis=0)))
        # Also, scaling the white point by brightness of the Y component
        vector_WPi = np.array((vector_WPi[0], vector_WPi[1], 1 - vector_WPi[0] - vector_WPi[1])) / vector_WPi[1]
        # Calculating the inverse chromaticity matrix
        matrix_M_inv = np.linalg.inv(matrix_M)
        # Calculating the scaling vector
        vector_S = matrix_M_inv.dot(vector_WPi)
        # RGB -> XYZ transformation matrix
        self.matrix = matrix_M * vector_S[np.newaxis, :]
        # XYZ -> RGB transformation matrix
        self.inv_matrix = matrix_M_inv / vector_S[:, np.newaxis]
        # Optional chromatic adaptation
        if adaptation_white_point != '' and adaptation_white_point != internal_white_point_name:
            # White point preprocessing
            vector_WPa = self.supported_white_points[adaptation_white_point]
            vector_WPa = np.array((vector_WPa[0], vector_WPa[1], 1 - vector_WPa[0] - vector_WPa[1])) / vector_WPa[1]
            # White scale in cone response domain
            vector_A = matrix_B.dot(vector_WPi) / matrix_B.dot(vector_WPa)
            # Applying the chromatic adaptation
            self.matrix = (matrix_B / vector_A[np.newaxis, :]) @ matrix_B_inv @ self.matrix
            self.inv_matrix = self.inv_matrix @ matrix_B_inv @ (matrix_B * vector_A[:, np.newaxis])
        # The matrices are used transposed
        self.matrix = self.matrix.T
        self.inv_matrix = self.inv_matrix.T

    def xyz_to_rgb(self,
            value0: npt.NDArray,
            error0: npt.NDArray | None = None
        ) -> tuple[npt.NDArray, npt.NDArray | None]:
        """ Converts XYZ color array into a RGB color space array """
        # 1D implementation: rgb = self.inv_matrix @ xyz
        value1 = np.einsum('ij, j... -> i...', self.inv_matrix, value0)
        if error0 is None:
            error1 = None
        else:
            # 1D implementation: cov_rgb = self.inv_matrix @ cov_xyz @ self.inv_matrix.T
            error1 = np.einsum('ij, jk..., kl -> il...', self.inv_matrix, error0, self.inv_matrix)
        return value1, error1

    def rgb_to_xyz(self,
            value0: npt.NDArray,
            error0: npt.NDArray | None = None
        ) -> tuple[npt.NDArray, npt.NDArray | None]:
        """ Converts RGB color array into the XYZ color space array """
        # 1D implementation: rgb = self.matrix @ xyz
        value1 = np.einsum('ij, j... -> i...', self.matrix, value0)
        if error0 is None:
            error1 = None
        else:
            # 1D implementation: cov_rgb = self.matrix @ cov_xyz @ self.matrix.T
            error1 = np.einsum('ij, jk..., kl -> il...', self.matrix, error0, self.matrix)
        return value1, error1

    @staticmethod
    def spectrum_to_white_point(spectrum: Spectrum) -> npt.NDArray:
        """ Returns (x, y) coordinates of the spectrum on the chromaticity diagram """
        xyz = observe(spectrum, xyz_cmf).spectral_dist
        return xyz[:2] / xyz.sum()

    # Values are color primaries (red, green, blue) and white points used.
    # See https://en.wikipedia.org/wiki/RGB_color_spaces
    # and http://brucelindbloom.com/index.html?WorkingSpaceInfo.html
    supported_color_spaces = {
        'CIE 1931 XYZ': (((1, 0), (0, 1), (0, 0)), 'Illuminant E'),
        'CIE 1931 RGB': (((0.73474284, 0.26525716), (0.27377903, 0.7174777), (0.16655563, 0.00891073)), 'Illuminant E'),
        'sRGB': (((0.64, 0.33), (0.30, 0.60), (0.15, 0.06)), 'Illuminant D65'),
        'Display P3': (((0.68, 0.32), (0.265, 0.69), (0.15, 0.06)), 'Illuminant D65'),
        'Adobe RGB': (((0.64, 0.33), (0.21, 0.71), (0.15, 0.06)), 'Illuminant D65'),
        'Wide Gamut RGB': (((0.7347, 0.2653), (0.1152, 0.8264), (0.1566, 0.0177)), 'Illuminant D50'),
        'ProPhoto RGB': (((0.734699, 0.265301), (0.159597, 0.840403), (0.036598, 0.000105)), 'Illuminant D50'),
        'HDTV': (((0.67, 0.33), (0.21, 0.71), (0.15, 0.06)), 'Illuminant D65'),
        'UHDTV': (((0.708, 0.292), (0.170, 0.797), (0.13, 0.046)), 'Illuminant D65'),
    }

    # Values are (x, y) coordinates.
    # https://en.wikipedia.org/wiki/Standard_illuminant#White_points_of_standard_illuminants
    supported_white_points = {
        'Illuminant A': (0.44758, 0.40745),
        'Illuminant B': (0.34842, 0.35161),
        'Illuminant C': (0.31006, 0.31616),
        'Illuminant D50': (0.34567, 0.35850),
        'Illuminant D55': (0.33242, 0.34743),
        'Illuminant D65': (0.31272, 0.32903),
        'Illuminant D75': (0.29902, 0.31485),
        'Illuminant D93': (0.28315, 0.29711),
        'Illuminant E': (1/3, 1/3),
        'Vega': spectrum_to_white_point(vega_CALSPEC),
        'Sun': spectrum_to_white_point(sun_CALSPEC),
    }


xyz_color_system = ColorSystem('CIE 1931 XYZ', 'Illuminant E')


class ColorObject:
    """
    This class stores a color brightness array (`self.br`) with values in the 0-1 range,
    postprocessing attributes, color system, and provides conversion methods.

    The brightness array is required to be of length 3 along the first axis.
    The indices correspond to the red, green and blue channel respectively.

    Attributes:
    - `color_system` is a frozen attribute, change by creating a new object with `to_color_system()`
    - `gamma_correction` makes the output to model the nonlinearity of the human eye’s perception of luminance
    - `maximize_brightness` normalize the output to the brightest RGB channel value
    - `scale_factor` multiplies the values of the output by a constant (implemented as property to check the input)
    """

    gamma_correction = False
    maximize_brightness = False
    _scale_factor = 1.

    def __init__(self,
            spectral_dist: npt.NDArray,
            covariance_matrix: npt.NDArray | None,
            color_system: ColorSystem
        ):
        """
        ColorObject requires a brightness array and corresponding color system.
        Default color space is CIE 1931 XYZ. Covariance matrix is optional.
        """
        self.spectral_dist = spectral_dist
        self.covariance_matrix = covariance_matrix
        self._color_system = color_system

    @classmethod
    def from_spectral_data(cls, data: RealObject) -> Self:
        """ Convolves (photo)spectrum with CIE 1931 XYZ color matching functions """
        photospectrum = observe(data, xyz_cmf)
        return cls(photospectrum.spectral_dist, photospectrum.covariance_matrix, xyz_color_system)

    def to_color_system(self, new_color_system: ColorSystem) -> Self:
        """
        Return a new ColorObject with changed color system.
        Attention! For saturated colors, color system conversion is not always reversible!
        """
        output = deepcopy(self)
        output.spectral_dist, output.covariance_matrix \
            = new_color_system.xyz_to_rgb(*self._color_system.rgb_to_xyz(self.spectral_dist))
        if np.any(output.spectral_dist < 0):
            # We're not in the color system gamut: approximate by desaturating
            # 1D implementation: rgb -= np.min(rgb)
            negative_mask = np.any(output.spectral_dist < 0, axis=0)
            output.spectral_dist[:, negative_mask] -= output.spectral_dist.min(axis=0)[negative_mask]
        output._color_system = new_color_system
        return output

    def to_array(self) -> npt.NDArray:
        """ Implies post-processing functions: gamma correction and brightness maximizing """
        arr = np.nan_to_num(self.spectral_dist, copy=True)
        if self.maximize_brightness and arr.max() != 0:
            arr /= arr.max()
        if self.scale_factor != 1:
            arr *= self.scale_factor
        if self.gamma_correction:
            arr = self.apply_gamma_correction(arr)
        return arr

    def grayscale(self) -> npt.NDArray | float:
        """ Converts color to grayscale using CIE 1931 luminance (Y in XYZ color space) """
        y = self.to_color_system(xyz_color_system).spectral_dist[1]
        if self.gamma_correction:
            y = self.apply_gamma_correction(y)
        return y

    @property
    def color_system(self) -> ColorSystem:
        """
        Color system cannot be set directly for safety reasons.
        Use to_color_system() to perform data conversion together with the color system.
        """
        return self._color_system

    @property
    def scale_factor(self) -> float:
        return self._scale_factor

    @scale_factor.setter
    def scale_factor(self, value):
        """ Checks the scale factor input """
        try:
            self._scale_factor = max(0, float(value))
        except ValueError:
            pass
            #print('Scale factor of ColorObject object must be a number.')

    @staticmethod
    def apply_gamma_correction(arr0: npt.ArrayLike) -> npt.NDArray:
        """ Applies sRGB gamma correction to the array """
        arr = np.asarray(arr0) # to allow float input use mask
        mask = arr < 0.0031308
        arr[mask] *= 12.92
        arr[~mask] = 1.055 * np.power(arr[~mask], 1./2.4) - 0.055
        return arr


class ColorPoint(ColorObject):
    """
    Class to work with an array of red, green and blue values.
    Stores brightness values in the range 0 to 1 in the `spectral_dist` attribute, numpy array of shape (3).
    """

    def to_bit(self, bit: int, clip: bool = False) -> npt.NDArray:
        """ Returns color array, scaled to the appropriate power of two (not rounded) """
        factor = 2**bit - 1
        arr = self.to_array()
        if clip:
            arr = np.clip(arr, 0, 1)
        return arr * factor

    def to_html(self) -> str:
        """ Converts fractional rgb values to HTML-styled hexadecimal string """
        return '#{:02x}{:02x}{:02x}'.format(*self.to_bit(8, clip=True).round().astype('int'))


class ColorLine(ColorObject):
    """
    Class to work with a line of red, green and blue channels.
    Stores brightness values in the range 0 to 1 in the `spectral_dist` attribute, numpy array of shape (3, X).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spectral_dist = np.atleast_2d(self.spectral_dist) # interprets color points as lines

    @property
    def size(self) -> int:
        """ Returns spatial axis length """
        return self.spectral_dist.shape[1]


class ColorImage(ColorObject):
    """
    Class to work with an image of red, green and blue channels.
    Stores brightness values in the range 0 to 1 in the `spectral_dist` attribute, numpy array of shape (3, X, Y).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spectral_dist = np.atleast_3d(self.spectral_dist) # interprets color points and lines as images

    def upscale(self, times: int) -> 'ColorImage':
        """ Creates a new ColorImage with increased size by an integer number of times """
        output = deepcopy(self)
        output.spectral_dist = np.repeat(np.repeat(output.spectral_dist, times, axis=0), times, axis=1)
        return output

    def downscale(self, pixels_limit: int):
        """ Brings the resolution of the image to approximately match the number of pixels """
        output = deepcopy(self)
        output.spectral_dist, output.covariance_matrix = \
            spatial_downscaling(output.spectral_dist, output.covariance_matrix, pixels_limit)
        return output

    @property
    def width(self) -> int:
        """ Returns horizontal spatial axis length """
        return self.spectral_dist.shape[1]

    @property
    def height(self) -> int:
        """ Returns vertical spatial axis length """
        return self.spectral_dist.shape[2]

    @property
    def size(self):
        """ Returns the number of pixels """
        return self.width * self.height
