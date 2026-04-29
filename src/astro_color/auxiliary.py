""" File containing constant and functions required in various places, but without dependencies """

import numpy as np
import numpy.typing as npt
from math import sqrt, ceil
from typing import Iterable, SupportsFloat, cast, Sequence

from .errors import UnsupportedDimensionError


# ------------ Core Section ------------

def integrate(array: npt.NDArray, step: int | float, precisely: bool = False): # -> float | npt.NDArray
    """
    Integration along the spectral axis.
    Uses the rectangle method by default and Riemann sum with midpoint in the "precise" mode.

    It is the inaccurate method that is most often used. Not because of speed, but because
    it is equivalent to matrix multiplication, which is used in spectrum reconstruction.
    In practice, the difference between the methods gives an accuracy gain of less than
    one hundredth of a factor.
    """
    if precisely:
        return step * 0.5 * np.sum(array[:-1] + array[1:], axis=0) # Riemann sum
    else:
        return step * np.sum(array, axis=0) # rectangle method

def is_smooth(array: npt.ArrayLike) -> bool:
    """ Boolean function, checks the second derivative for sign reversal, a simple criterion for smoothness """
    diff2 = np.diff(np.diff(array, axis=0), axis=0)
    return bool(np.all(diff2 <= 0) | np.all(diff2 >= 0))

def spectral_binning(
        nm0: npt.NDArray,
        br0: npt.NDArray,
        std0: npt.NDArray | None,
        nm1: npt.NDArray,
        step: int | float,
        nm0_diff: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray | None]:
    """
    Cumulative-integral binning method for a uniform grid. Fast (O(N) complexity), vectorized.
    Requires at least one measurement per bin; otherwise use `spectral_downscaling()`.
    """
    # Computing bin edges assuming that nm1 is a uniform grid
    half_step = 0.5 * step
    nm1_edges = np.append(nm1 - half_step, nm1[-1] + half_step)
    nm0_diff = stretch(nm0_diff, br0.shape[1:])
    # Cumulative distribution function
    br_cdf = np.zeros(br0.shape, dtype=np.float64)
    br_cdf[1:] = np.cumsum(0.5 * (br0[:-1] + br0[1:]) * nm0_diff, axis=0) # Riemann sum
    br1 = np.diff(linear_interp(nm0, br_cdf, nm1_edges), axis=0) / step
    std1 = None
    if std0 is not None:
        # Problem 2
        std_cdf = np.zeros(nm0.shape, dtype=np.float64)
        std_cdf[1:] = np.cumsum(0.5 * (std0[:-1] + std0[1:]) * nm0_diff, axis=0)
        std1 = np.diff(linear_interp(nm0, br_cdf, nm1_edges), axis=0) / step
    return br1, std1

fwhm_factor = np.sqrt(8 * np.log(2))

def gaussian_width(current_resolution, target_resolution):
    return np.sqrt(np.abs(target_resolution**2 - current_resolution**2)) / fwhm_factor

def gaussian_convolution(nm0: npt.NDArray, br0: npt.NDArray, nm1: npt.NDArray, step: int | float):
    """
    Applies Gaussian convolution to a non-uniform sparse mesh. Eliminates holes and noise from spectral axis.

    Args
    - nm0: original spectral axis
    - br0: original spectrum
    - nm1: required uniform grid
    - step: standard deviation of the Gaussian
    """
    factor = -0.5 / step**2 # Gaussian exponent multiplier
    br1 = np.empty_like(nm1, dtype=br0.dtype)
    for i in range(len(nm1)):
        br0_convolved = br0 * np.exp(factor*(nm0 - nm1[i])**2)
        br1[i] = np.average(br0, weights=br0_convolved)
    return br1

def spectral_downscaling(
        nm0: npt.NDArray,
        br0: npt.NDArray,
        std0: npt.NDArray | None,
        nm1: npt.NDArray,
        step: int | float
    ) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Returns spectrum brightness values with decreased resolution.
    Incoming graphs or point clouds may have gaps and areas of varying resolution.

    Args
    - nm0: original spectral axis
    - br0: original spectrum, spectral set or spectral cube
    - std0: ther standard deviations
    - nm1: required uniform grid
    - step: resolution of the required uniform grid

    It is known that the Gaussian standard deviation, without any assumptions, must correspond to the grid step.
    Knowing the required "blur" and the local "blur", the missing degree of "blur" can be calculated from
    the error propagation equation.

    The idea is inspired by https://gist.github.com/keflavich/37a2705fb4add9a2491caf2dfa195efd
    """
    cube_flag = br0.ndim == 3 # spectral cube processing
    if br0.min() < 0:
        br0 = np.clip(br0, np.nextafter(0, 1), None) # strange NumPy errors with weights without it
    notnan = ~np.isnan(br0)
    nm0 = nm0[notnan]
    br0 = br0[notnan]
    if std0 is not None:
        std0 = std0[notnan]
    # Obtaining a graph of standard deviations for a Gaussian
    nm_diff = np.diff(nm0)
    nm_mid = (nm0[1:] + nm0[:-1]) * 0.5
    # Calculates the continuous (smoothed by gaussian) density of the original spectral grid
    std_local = gaussian_width(gaussian_convolution(nm_mid, nm_diff, nm1, step*2), step) # missing "blur"
    # Gaussian exponent multipliers (0.001 is a small value to prevent zero division error like with 203 Pompeja)
    factors = -0.5 / np.clip(std_local, 0.001, None)**2
    # Convolution with Gaussian of variable standard deviation
    br1 = np.empty_like(nm1, dtype=br0.dtype)
    if cube_flag:
        br1 = stretch(br1, br0.shape[1:3])
    if std0 is None:
        std1 = None
        uncertainty_weights = np.ones_like(nm0)
    else:
        std1 = np.empty_like(br1)
        uncertainty_weights = std0**(-2)
    for i in range(len(nm1)):
        # Variable convolution kernel
        gaussian_weights = np.exp(factors[i]*(nm0 - nm1[i])**2)
        weights = uncertainty_weights * gaussian_weights
        if cube_flag:
            weights = stretch(weights, br0.shape[1:3])
        try:
            br1[i] = np.average(br0, weights=weights, axis=0)
            if std1 is not None:
                # Assumed formula, not proved (Problem 3)
                std1[i] = np.sum(weights, axis=0)**(-0.5)
                # If we had normal binning (on a limited interval), the formula would be
                # np.sum(uncertainty_weights[i])**(-0.5),
                # and at the limit, it would give σ1 = σ0 / sqrt(N)
                # Taking advantage of the fact that gaussian_weights take values from 1
                # at the center to 0 at infinity, I use them for summation of uncertainty_weights
        except ZeroDivisionError:
            br1[i] = np.average(br0, axis=0)
            if std1 is not None:
                std1[i] = 0
    return br1, std1

def spatial_downscaling(
        spectral_dist: npt.NDArray,
        covariance_matrix: npt.NDArray | None,
        pixels_limit: int
    ) -> tuple[npt.NDArray, npt.NDArray | None]:
    """ Brings the spatial resolution of the cube to approximately match the number of pixels """
    # TODO: averaging like in https://stackoverflow.com/questions/10685654/reduce-resolution-of-array-through-summation
    _, x, y = spectral_dist.shape
    factor = ceil(sqrt(x * y / pixels_limit))
    if covariance_matrix is None:
        return spectral_dist[:,::factor,::factor], None
    else:
        return spectral_dist[:,::factor,::factor], covariance_matrix[:,:,::factor,::factor]

def smoothness_matrix(n: int, order: int = 1):
    """
    Generates a smoothness operator matrix of the specified size n.
    Supported options:
    - identity matrix (n x n), for hight restriction
    - first-order difference operator (n-1 x n), for height change restriction
    - second-order difference operator (n-2 x n), for curvature restriction
    """
    if order == 0:
        # Hight restriction
        return np.eye(n, dtype=np.uint8)
    else:
        m = n - order
        L = np.zeros((m, n), dtype=np.int8)
        # Note: int8 makes it ~2 times faster
        match order:
            # Note: numpy doesn't make it faster
            # tested: diag = np.eye(m, dtype='int8'); L[:,:-1] = diag; L[:,1:] -= diag
            case 1:
                # Height change restriction
                for i in range(m):
                    L[i, i] = 1
                    L[i, i+1] = -1
            case 2:
                # Curvature restriction
                for i in range(m):
                    L[i, i] = 1
                    L[i, i+1] = -2
                    L[i, i+2] = 1
            case _:
                raise ValueError(f'Order {order} of smoothness matrix is not supported.')
    return L

def expand2x(array0: npt.NDArray):
    """ Expands the array along the first axis by half """
    new_length = 2 * array0.shape[0] - 1
    match array0.ndim:
        case 1:
            array1 = np.empty(new_length, dtype=array0.dtype)
        case 2: # spectral set processing
            array1 = np.empty((new_length, array0.shape[1]), dtype=array0.dtype)
        case 3: # spectral cube processing
            array1 = np.empty((new_length, array0.shape[1], array0.shape[2]), dtype=array0.dtype)
        case _:
            raise UnsupportedDimensionError(array0.ndim)
    array1[0::2] = array0
    array1[1::2] = (array0[:-1] + array0[1:]) * 0.5
    return array1

def linear_interp(x0: npt.NDArray, y0: npt.NDArray, x1: npt.NDArray):
    """
    Equivalent to the `np.interp(x1, x0, y0)`, but also works for sets and cubes.
    Allows extrapolation by using the first or last point.
    """
    idx = np.clip(np.searchsorted(x0, x1), 0, len(x0) - 1)
    x_left = x0[idx - 1]
    y_left = y0[idx - 1]
    delta_x = x0[idx] - x_left
    delta_y = y0[idx] - y_left
    return y_left + ((x1 - x_left) / delta_x * delta_y.T).T

def custom_interp(array0: npt.NDArray, k=16):
    """
    Returns curve or cube values with twice the resolution. Can be used in a loop.
    Optimal in terms of speed to quality ratio: around 2 times faster than splines in scipy.

    Args:
    - `array0` (npt.NDArray): values to be interpolated in shape (2, N)
    - `k` (int): lower -> more chaotic, higher -> more linear, best results around 10-20
    """
    array1 = expand2x(array0)
    match array0.ndim:
        case 1:
            zero = np.zeros((1,))
        case 2: # spectral set processing
            zero = np.zeros((1, array0.shape[1]))
        case 3: # spectral cube processing
            zero = np.zeros((1, array0.shape[1], array0.shape[2]))
        case _:
            raise UnsupportedDimensionError(array0.ndim)
    delta_left = np.concatenate((zero, array0[1:-1] - array0[:-2]))
    delta_right = np.concatenate((array0[2:] - array0[1:-1], zero))
    array1[1::2] += (delta_left - delta_right) / k
    return array1

def interpolate(x0: npt.NDArray, y0: npt.NDArray, x1: npt.NDArray, step: int | float) -> npt.NDArray:
    """
    Returns interpolated `y0` values on a uniform grid `x0`. Uses enhanced linear interpolation.
    Combination of `custom_interp` (which returns an uneven mesh) and linear interpolation after it.
    The chaotic-linearity parameter increases with each iteration to reduce the disadvantages of custom_interp.
    """
    for i in range(int(np.log2(np.diff(x0).max() / step))):
        x0 = expand2x(x0)
        y0 = custom_interp(y0, k=11+i)
    return linear_interp(x0, y0, x1)


def stretch(arr: npt.NDArray, times: int | tuple, copy=False):
    """
    Adds dimensions to the array at the end and repeats it there.
    Uses broadcast by default for memory-efficiency. Uses np.tile() for copying.
    """
    if isinstance(times, int):
        times = (times,)
    new_axes = len(times) * (np.newaxis,)
    if copy:
        return np.tile(arr[..., *new_axes], (*((1,) * len(arr.shape)), *times))
    else:
        return np.broadcast_to(arr[..., *new_axes], (*arr.shape, *times))


def custom_extrap(grid: npt.NDArray, derivative: float | npt.NDArray, corner_x: int | float, corner_y: npt.NDArray) -> npt.NDArray:
    """
    Returns an intuitive continuation of the function on the grid using information about the last point.
    Extrapolation bases on function f(x) = exp( (1-x²)/2 ): f' has extrema of ±1 in (-1, 1) and (1, 1).
    Therefore, it scales to complement the spectrum more easily than similar functions.
    """
    if np.all(derivative) == 0: # extrapolation by constant
        return stretch(corner_y, grid.size)
    else:
        grid = stretch(grid, corner_y.shape)
        sign = np.sign(derivative)
        return np.exp((1 - (np.abs(derivative) * (grid - corner_x) / corner_y - sign)**2) / 2) * corner_y

def extrap_std(corner_y: float | npt.NDArray, x_arr: npt.NDArray):
    """ The exponential growth of uncertainty is completely arbitrary and needs to be investigated """
    return corner_y * 0.05 * (1.01**x_arr - 1)


weights_center_of_mass = 1 - 1 / np.sqrt(2)

def extrapolating(
        x: npt.NDArray,
        y: npt.NDArray,
        std: npt.NDArray | None,
        x_arr: npt.NDArray,
        step: int,
        avg_steps=20
    ):
    """
    Defines a (multi-dimensional) curve an intuitive continuation on the x_arr, if needed.
    `avg_steps` is a number of corner curve points to be averaged if the curve is not smooth.
    Averaging weights on this range grow linearly closer to the edge (from 0 to 1).
    The exponential growth of uncertainty is completely arbitrary and needs to be investigated.
    """
    spatial_shape = y.shape[1:] # (,) for 1D; (n,) for 2D; (w, h) for 3D
    # std processing is currently broken for spectral cubes and sets
    if std is None:
        # Uncertainty is generated
        std = np.zeros_like(y)
        std_left = std_right = 0.
    else:
        std_left = std[0]
        std_right = std[-1]
    if len(x) == 1: # filling with equal-energy spectrum
        x1 = np.arange(min(x_arr[0], x[0]), max(x_arr[-1], x[0])+step, step)
        y1 = stretch(y[0], x1.size)
        std = extrap_std(y[0], np.abs(x1 - x[0]))
        x = x1
        y = y1
    else:
        if x[0] > x_arr[0]:
            # Extrapolation to blue
            x1 = np.arange(x_arr[0], x[0], step)
            if np.all(y[0] == 0):
                # Corner point is zero -> no extrapolation needed: most likely it's a filter profile
                y1 = std1 = np.zeros((x1.size, *spatial_shape))
            else:
                y_arr = y[:avg_steps]
                if is_smooth(y_arr):
                    diff = y[1]-y[0]
                    corner_y = y[0]
                else:
                    # Linear weights. Could be more complicated, but there is no need
                    avg_weights = stretch(np.arange(-avg_steps, 0)[avg_steps-y_arr.shape[0]:], spatial_shape)
                    diff = np.average(np.diff(y_arr, axis=0), weights=avg_weights[:-1], axis=0)
                    corner_y = np.average(y_arr, weights=avg_weights, axis=0) - diff * avg_steps * weights_center_of_mass
                y1 = custom_extrap(x1, diff/step, x[0], corner_y)
                std1 = std_left + stretch(extrap_std(corner_y, np.arange(int(x[0]-x_arr[0]), 0, -step) - step), spatial_shape)
                #                                                        ^^^ solves bug with uint16
            x = np.append(x1, x)
            y = np.append(y1, y, axis=0)
            std = np.append(std1, std, axis=0)
        if x[-1] < x_arr[-1]:
            # Extrapolation to red
            x1 = np.arange(x[-1], x_arr[-1], step) + step
            if np.all(y[0] == 0):
                # Corner point is zero -> no extrapolation needed: most likely it's a filter profile
                y1 = std1 = np.zeros((x1.size, *spatial_shape))
            else:
                y_arr = y[-avg_steps:]
                if is_smooth(y_arr):
                    diff = y[-1]-y[-2]
                    corner_y = y[-1]
                else:
                    avg_weights = stretch(np.arange(avg_steps)[:y_arr.shape[0]] + 1, spatial_shape)
                    diff = np.average(np.diff(y_arr, axis=0), weights=avg_weights[1:], axis=0)
                    corner_y = np.average(y_arr, weights=avg_weights, axis=0) + diff * avg_steps * weights_center_of_mass
                y1 = custom_extrap(x1, diff/step, x[-1], corner_y)
                std1 = std_right + stretch(extrap_std(corner_y, np.arange(0, x_arr[-1]-x[-1], step)), spatial_shape)
            x = np.append(x, x1)
            y = np.append(y, y1, axis=0)
            std = np.append(std, std1, axis=0)
    if std is not None and std.sum() == 0:
        std = None
    return x, y, std



# ------------ Database Processing Section ------------

def parse_value_std(data: float | Sequence[float]) -> tuple[float, float | None]:
    """
    Guarantees the output of the value and its standard deviation.

    Supported input types:
    - value
    - [value, std]
    - [value, +std1, -std2]
    - [value, -std1, +std2]
    """
    if isinstance(data, SupportsFloat):
        # no standard deviation
        return float(data), None
    elif isinstance(data, Sequence):
        match len(data):
            case 2:
                # regular standard deviation
                return cast(tuple[float, float], tuple(data))
            case 3:
                # asymmetric standard deviation
                value, std1, std2 = data
                std = 0.5 * (abs(std1) + abs(std2)) # reduced to regular
                return value, std
    raise ValueError(f'Invalid data input: {data}. Must be a numeric value or a [value, std] list.')

def parse_value_std_list(arr: Iterable) -> tuple[npt.NDArray, npt.NDArray | None]:
    """ Splits the values and standard deviations into two arrays """
    try:
        arr = np.array(arr, dtype=np.float64) # ValueError here means inhomogeneous shape
    except ValueError:
        # Inhomogeneous standard deviation input
        values = []
        for data in arr:
            value, _ = parse_value_std(data)
            values.append(value)
        return np.array(values, dtype=np.float64), None
    try:
        # No standard deviation
        if arr.ndim == 0:
            arr = np.atleast_1d(arr)
        elif arr.ndim > 1:
            raise ValueError # means std is there
        return arr, None
    except ValueError:
        # Standard deviations are present
        values = []
        stds = []
        for data in arr:
            value, std = parse_value_std(data)
            values.append(value)
            stds.append(std)
        return np.array(values, dtype=np.float64), np.array(stds, dtype=np.float64)

def repeat_if_value(data: int | float | npt.ArrayLike, arr_len: int) -> npt.NDArray[np.float64]:
    """ If the input consists of a single number, stretches to 1D array """
    arr = np.array(data, dtype=np.float64)
    if arr.ndim == 0:
        # A single number
        return np.repeat(arr, arr_len)
    else:
        return arr

def mag2irradiance(mag: int | float | npt.NDArray, zero_point: float = 1.):
    """ Converts magnitudes to irradiance (by default in Vega units) """
    return zero_point * 10**(-0.4 * mag)

def std_mag2std_irradiance(std_mag: int | float | npt.NDArray, irradiance: int | float | npt.NDArray):
    """
    Converts standard deviation of the magnitude to a irradiance standard deviation.

    The formula is derived from the error propagation equation:
    I(mag) = zero_point ∙ 10^(-0.4 mag)
    std_I² = (d I / d mag)² ∙ std_mag²
    I' = zero_point∙(10^(-0.4 mag))' = zero_point∙10^(-0.4 mag)∙ln(10^(-0.4)) = I∙(-0.4) ln(10)
    std_I = |I'| ∙ std_mag = 0.4 ln(10) ∙ I ∙ std_mag
    """
    return 0.4 * np.log(10) * irradiance * std_mag

def color_index_splitter(index: str) -> tuple[str, str]:
    """
    Dashes in filter names are allowed in the SVO Filter Profile Service.
    This function should fix all or most of the problems caused.
    """
    try:
        filter1, filter2 = index.split('-')
    except ValueError:
        dotpart1, dotpart2, dotpart3 = index.split('.') # one dot per full filter name
        dashpart1, dashpart2 = dotpart2.split('-', 1)
        filter1 = f'{dotpart1}.{dashpart1}'
        filter2 = f'{dashpart2}.{dotpart3}'
    return filter1, filter2

def color_indices_parser(indices: dict):
    """
    Converts color indices to linear brightness, assuming mag=0 in the first filter.
    Each new color index must refer to a previously specified one.
    Note: The output order may sometimes not be in ascending wavelength order.
    This can be corrected by knowing the filter profiles, which better to do outside the function.

    For standard deviations the error propagation equation is used:
    f(x, y) = x - y
    std_f² = (df/dx)² std_x² + (df/dy)² std_y² = std_x² + std_y²
    where x, y are magnitudes and f is a color index.

    Finding standard deviations of a photospectrum built from color indices is an ill-posed problem:
    it's just like with integrating, we loose constant after differentiation (color indices is
    a discrete differential form of a photospectrum).
    For the photospectrum itself, it's not a problem: the solutions dimension is just scaling
    the spectrum on a constant (it's pretty obvious that color indices always lost brightness
    information).
    But the solutions space for their standard deviations is more complex, I found its geometric
    interpretation: since the standard deviations subtracting rule is, in fact, the Pythagorean theorem,
    the solutions space is the same as if you try to build a line of right triangles, for each one
    the next cathetus is linked to a previous cathetus by their square.
    (To draw on the plane you have to use rhombuses of the same area and the same equal sides
    instead of squares.)
    N hypotenuses are standard deviations of color indices, and N+1 different cathetes are the sought
    standard deviations of the photospectrum.
    The whole triangle line possible positions can be described by just one parameter (1D parametric
    space of solutions).
    For simplicity, I will choose the first cathetus (the first sought standard deviation) as
    a variable of this space.
    Such triangles can "collapse" if the previous cathetus is greater than the next hypotenuse!
    I tried to find an analytical solution, but the requirement for the optimal solution I derived
    suggested that about a half of the triangles should be collapsed.

    In the numerical approach I use, some solutions are collapsed (the `try-except` code block),
    but there are a some range of possible solutions too, which one to choose?
    I decided to choose the solution with minimal standard deviation of standard deviations it gives.
    To tighten the solution selection criteria, it can be assumed that the size of the standard deviation
    is inversely proportional to the root of the irradiance (in the Poisson noise approximation).
    So it's better to minimize the differences not between the stds of magnitudes, but between
    the stds of scaled irradiances.
    """
    first_color_index = tuple(indices.keys())[0]
    filter0, _ = color_index_splitter(first_color_index)
    _, std0 = parse_value_std(indices[first_color_index])
    # Photospectrum calculation
    uncertainty_flag = True
    filters = {filter0: 0} # mag=0 for the first point (arbitrarily)
    for key, value in indices.items():
        bluer_filter, redder_filter = color_index_splitter(key)
        mag, std = parse_value_std(value)
        if std is None:
            uncertainty_flag = False
        if bluer_filter in filters:
            filters |= {redder_filter: filters[bluer_filter] - mag}
        else:
            filters |= {bluer_filter: filters[redder_filter] + mag}
    irradiance = mag2irradiance(np.array(tuple(filters.values())))
    filter_names = filters.keys() # name setting before using the variable for std processing
    std = None
    # Uncertainty calculation
    if uncertainty_flag:
        std0 = cast(float, std0)
        shot_noise_factor = np.sqrt(irradiance) # common Poisson noise factor
        std_of_std = np.inf
        for std_assumed in np.linspace(0, std0, 1001):
            impossible_assumption = False
            # Numerically select the best value of the standard deviation of the first point,
            # on which all other standard deviations clearly depend
            filters = {filter0: std_assumed}
            for key, value in indices.items():
                bluer_filter, redder_filter = color_index_splitter(key)
                index_std = cast(float, parse_value_std(value)[1])
                try:
                    if bluer_filter in filters:
                        filters |= {redder_filter: sqrt(index_std**2 - filters[bluer_filter]**2)}
                    else:
                        filters |= {bluer_filter: sqrt(index_std**2 - filters[redder_filter]**2)}
                except ValueError:
                    # This means that the difference under the root is negative
                    # and the initial standard deviation assumption is not possible
                    impossible_assumption = True
                    break
            if not impossible_assumption:
                new_std = std_mag2std_irradiance(np.array(tuple(filters.values())), irradiance)
                # Finding the minimum deviation between std as solution quality criterion
                # The standard deviations are scaled by the Poisson noise factor
                new_std_of_std = np.std(new_std * shot_noise_factor)
                if new_std_of_std < std_of_std:
                    std = new_std
                    std_of_std = new_std_of_std
                    continue
                else:
                    # Means that the best values of standard deviations were found
                    # in the last iteration and they started to diverge
                    break
    return filter_names, irradiance, std



c_kms = 299792.458 # Speed of light in km/s

def cosmological_redshift(wave, z):
    """ Applyes the redshift correction to the wavelength array (1+z = λ_obs/λ_emit) """
    return wave / (1 + z)

def calc_redshift_sqrt(vel):
    """ Calculates the redshift from the velocity in km/s """
    v = vel / c_kms
    return np.sqrt((1+v)/(1-v)) - 1

def calc_redshift_exp(vel):
    """ Calculates the redshift from the velocity in km/s """
    v = vel / c_kms
    return np.exp(v) - 1


def repr_generator(arr_1D: npt.NDArray, is_int: bool = False):
    format = '' if is_int else '.3f'
    match len(arr_1D):
        case 0:
            return ''
        case 1:
            return f'{arr_1D[0]:{format}}'
        case 2:
            return f'{arr_1D[0]:{format}}, {arr_1D[1]:{format}}'
        case 3:
            return f'{arr_1D[0]:{format}}, {arr_1D[1]:{format}}, {arr_1D[2]:{format}}'
        case _:
            return f'{arr_1D[0]:{format}}, {arr_1D[1]:{format}}, ..., {arr_1D[-1]:{format}}'
