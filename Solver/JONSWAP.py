import numpy as np

def response(Values, RAOpd, Hs, Tp, gamma):
    # jonswapSz as defined by 4 parameters,
    # omega     - radial frequency	[ra d /s]
    # Hs		- Significant wave height	[m]
    # Tp		- Peak wave period		[s]
    # Sz		- Energy Density Spectrum	[ m ^ 2 *s] also [ m ^2( s /rad)]
    omega = np.linspace(2*np.pi/Values['t_max'], 2*np.pi/Values['t_min'], Values['n_t'])
    T1 = Tp * 0.834
    omega_p = 2*np.pi/Tp
    sigma_a = 0.07
    sigma_b = 0.09

    sigma = [sigma_a]*Values['n_t']
    A = [] * Values['n_t']
    B = [] * Values['n_t']
    Sz = [] * Values['n_t']
    # omega_p_loc = np.argwhere(omega > omega_p)[0][0]
    for a, b in enumerate(omega):
        if b > omega_p:
            sigma[a] = sigma_b
        B[a] = np.exp(-(((omega[a]/omega_p)-1)/(sigma[a]*np.sqrt(2)))**2)
        A[a] = gamma * B[a]
        Sz[a] = ((320*(Hs**2)*(omega[a]**(-5)))/(Tp**4)) * np.exp(-1950*(omega[a]**4)/(Tp**4)) * A[a]
    print(Sz)
