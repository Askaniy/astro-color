from typing import Any
import warnings


# Errors

class AstroColorError(Exception):
    pass

class UnsupportedDimensionError(AstroColorError, ValueError):
    def __init__(self, ndim_recieved: int, name: Any = None):
        subject = f'data array of {name}' if name else 'data array'
        super().__init__(f'The {subject} must have a dimension of 1, 2, or 3, not {ndim_recieved}.')

class InconsistentDimensionError(AstroColorError, ValueError):
    def __init__(self, ndim_recieved: int, ndim_expected: int, name: Any):
        msg = f'The received array dimension ({ndim_recieved}) differs from the expected one ({ndim_expected})'
        if name is not None:
            msg += f' for {name}'
        super().__init__(msg + '.')

class InconsistentAxesError(AstroColorError, ValueError):
    def __init__(self, spectral_axis: int, spatial_axis: int, name: Any):
        msg = f'Arrays of spectral and spatial axes do not match ({spectral_axis} vs {spatial_axis})'
        if name is not None:
            msg += f' for {name}'
        super().__init__(msg + '.')

class InconsistentUncertaintySizeError(AstroColorError, ValueError):
    def __init__(self, len_error: int, len_values: int, name: Any):
        msg = f'Uncertainty array does not match the spectral axis ({len_error} vs {len_values})'
        if name is not None:
            msg += f' for {name}'
        super().__init__(msg + '.')

class InconsistentUncertaintyShapeError(AstroColorError, ValueError):
    def __init__(self, shape_error: int, shape_values: int, name: Any):
        msg = f'Uncertainty shape {shape_error} does not match the data shape {shape_values}'
        if name is not None:
            msg += f' for {name}'
        super().__init__(msg + '. It cannot be a standard deviation or a covariance matrix.')

class FilterNotFoundError(AstroColorError):
    def __init__(self, filter_id: Any):
        super().__init__(f'Filter "{filter_id}" not found in the "filters" folder.')


# Warnings

class ErasingCorrelationsWarning(UserWarning):
    pass

def erasing_correlations_warning(name: Any):
    msg = 'The full covariance matrix is not supported here. The diagonal is used to estimate errors'
    if name is not None:
        msg += f' for {name}'
    warnings.warn(msg + '.', ErasingCorrelationsWarning, stacklevel=2)

class NanValuesWarning(UserWarning):
    pass

def nan_values_warning(input: str, name: Any):
    msg = f'NaN values detected in the {input} input been replaced with zeros'
    if name is not None:
        msg += f' for {name}'
    warnings.warn(msg + '.', NanValuesWarning, stacklevel=2)

class ZeroBrightnessWarning(UserWarning):
    pass

def zero_brightness_warning(name: Any):
    msg = 'A division-by-zero error occurred in the calculations due to the zero brightness'
    if name is not None:
        msg += f' of object {name}'
    warnings.warn(msg + '.', ZeroBrightnessWarning, stacklevel=2)

class EmptySpectralIntersectionWarning(UserWarning):
    pass

def empty_spectral_intersection_warning(nm0: int, nm1: int, start: int, end: int, name: Any = None):
    msg = f'The requested wavelength range [{start} ... {end}] lies outside the range [{nm0} ... {nm1}]! An empty result returned'
    if name is not None:
        msg += f' for {name}'
    warnings.warn(msg + '.', EmptySpectralIntersectionWarning, stacklevel=2)

def empty_spectral_intersection_operator_warning(operation_name, name1, name2, start, end):
    warnings.warn(
        f'''
        There is no intersection between the spectra for the element-wise operation "{operation_name}":
        "{name1}" ends on {end} nm and "{name2}" starts on {start} nm. Stub object was created.
        ''',
        EmptySpectralIntersectionWarning,
        stacklevel=2
    )
