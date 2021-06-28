import numpy as np
import pandas as pd

def response(Values, RAOpd, Hs, Tp, gamma):
    # jonswapSz as defined by 4 parameters,
    # omega     - radial frequency	[ra d /s]
    # Hs		- Significant wave height	[m]
    # Tp		- Peak wave period		[s]
    # Sz		- Energy Density Spectrum	[ m ^ 2 *s] also [ m ^2( s /rad)]
    periods = np.linspace(Values['t_min'], Values['t_max'], Values['n_t'])
    omegas = [(2*np.pi/x) for x in periods]
    dofs = ['Surge', 'Sway', 'Heave', 'Roll', 'Pitch', 'Yaw']
    omega_p = 2 * np.pi / Tp
    # T1 = Tp * 0.834
    sigma_a = 0.07
    sigma_b = 0.09

    sigma = [sigma_a]*Values['n_t']
    A = [0] * Values['n_t']
    B = [0] * Values['n_t']
    Sz = [0] * Values['n_t']

    for a in range(len(periods)):
        if (omegas[a]) > omega_p:
            sigma[a] = sigma_b
        B[a] = np.exp(-(((omegas[a]/omega_p)-1)/(sigma[a]*np.sqrt(2)))**2)
        A[a] = gamma * B[a]
        Sz[a] = ((320*(Hs**2)*(omegas[a]**(-5)))/(Tp**4)) * np.exp(-1950*(omegas[a]**4)/(Tp**4)) * A[a]
        for c in range(0, 5):
            RAOpd[dofs[c]][a] = Sz[a] * RAOpd[dofs[c]][a] ** 2
    return Sz, RAOpd
