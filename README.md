# AstroColor

A Python library for converting between photometric systems, true color calculation, and space image processing.

It performs synthetic photometry on low-resolution (5-nm wavelength grid step) spectra as well as spectral reconstruction from measurements in filters based on Tikhonov regularization.

**Pre-alpha version! Do not use!**


## Installation

Use AstroColor in a virtual environment:
```sh
python3 -m venv .venv
.venv/bin/pip install git+https://github.com/Askaniy/astrocolor.git
```

Or add AstroColor to your [uv](https://github.com/astral-sh/uv) project:
```sh
uv add git+https://github.com/Askaniy/astrocolor.git
```


## Key features

- Calculate synthetic photometry
```py
import astrocolor as ac

spectrum = ac.Spectrum(
    wavelength_nm=[400, 500, 600, 700],
    spectral_dist=[1, 2, 2, 1]
)
v_band = ac.Filter('Generic_Bessell.V')
flux_value, flux_error = ac.observe(spectrum, v_band)
```

- Create a filter system for optimizations
```py
johnson_system = ac.FilterSet(
    'Generic_Bessell.B',
    'Generic_Bessell.V',
    'Generic_Bessell.R'
)
photospectrum_BVR = ac.observe(spectrum, johnson_system)
```

- Reconstruct photometry measurements into a smooth spectrum
```py
reconstructed = ac.spectral_reconstruction(photospectrum_BVR, requested_wavelengths=[400, 700])
```

- Convert measurements directly between photometric systems
```py
sloan_system = ac.FilterSet('SLOAN_SDSS.g', 'SLOAN_SDSS.r')
photospectrum_gr = ac.observe(photospectrum_BVR, sloan_system)
```

- Calculate true colors
```py
color_xyz = ac.ColorPoint.from_spectral_data(ac.sun_CALSPEC)
color_system = ac.ColorSystem('sRGB', 'Illuminant E') # recommended
color_rgb = color_xyz.to_color_system(color_system)
color_rgb.maximize_brightness = True
color_html = color_rgb.to_html()
```

- Model spectra
```py
bb_3000K = ac.BlackBodyModel(3000).determine_at_wavelengths([400, 700])
```

- Process images via spectral cube reconstruction
```py
# coming soon after debugging
```

- Error propagation with covariance matrices
```py
# coming soon after debugging
```



## History

[TrueColorTools](https://github.com/Askaniy/TrueColorTools) were created in 2020 to resolve disputes regarding the color of celestial bodies.
It features a graphical user interface and a user-expandable spectral database.
Over time, the core of the program became self-contained enough to be spun off into a library.
The refactoring took place in 2026; it opens up a general astronomical application.

Not vibe coded!
