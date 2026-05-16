import numpy as np
import numpy.typing as npt

from .spectral_objects import Spectrum
from .data_manager import script_folder



sun_data = np.load(script_folder/'data/Sun_CALSPEC.npz')
sun_CALSPEC = Spectrum(sun_data['wavelength_nm'], sun_data['spectral_dist'])
del sun_data

vega_data = np.load(script_folder/'data/Vega_CALSPEC.npz')
vega_CALSPEC = Spectrum(vega_data['wavelength_nm'], vega_data['spectral_dist'])
del vega_data



# TODO: StellarModel, RayleighModel, SynchrotronModel, ...


h = 6.626e-34 # Planck constant
c = 299792458 # Speed of light
k = 1.381e-23 # Boltzmann constant
const1 = 2 * h * c * c # * np.pi to get exitance (W/m2) in the assumption of Lambertian surface
const2 = h * c / k


class BlackBodyModel():
    """ Creates a Spectrum object based on Planck's law and redshift formulas """

    def __init__(self, temperature: int | float, velocity=0., vII=0.) -> None:
        self.T = temperature
        self.v = velocity
        self.vII = vII

    def planck_radiance(self, nm: int | float | npt.NDArray) -> float | npt.NDArray:
        m = nm * 1e-9
        radiance = const1 / (m**5 * (np.exp(const2 / (m * self.T)) - 1))
        return radiance * 1e-9 # per m -> per nm

    def _determine_at_trusted_wavelengths(self, requested_wavelengths: npt.NDArray):
        """
        Directly uses the provided wavelength grid to create a new object.
        See `determine_at_wavelengths()` for the general case.
        """
        doppler = 1
        grav = 1
        if self.T == 0:
            physics = False
        else:
            physics = True
            if self.v != 0:
                if abs(self.v) != 1:
                    doppler = np.sqrt((1-self.v) / (1+self.v))
                else:
                    physics = False
            if self.vII != 0:
                if self.vII != 1:
                    grav = np.exp(-0.5 * self.vII**2)
                else:
                    physics = False
        if physics:
            br = self.planck_radiance(requested_wavelengths * doppler * grav)
        else:
            br = np.zeros(requested_wavelengths.size)
        return Spectrum(requested_wavelengths, br, name=f'BB with T={round(self.T)} K')
