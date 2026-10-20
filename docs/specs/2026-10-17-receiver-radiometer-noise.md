# Loaded receiver noise and measured-product covariance

`receiver_noise.py` works at the loaded receiver-input plane with one-sided
voltage/current spectral covariances in V²/Hz and A²/Hz. The model is
`v_noise=e+Z_eff i`, where `Z_eff=ZL (ZA+ZL)^-1 ZA` uses luseepy's loading
matrix. A joint voltage/current PSD can carry cross-port and e/i
correlations. Positivity is checked after rescaling the two unit systems.
The diagonal-ASD helper requires both inputs explicitly; an impedance model
does not supply amplifier-noise measurements.

For Hermitian readout matrices `q_p=v^H A_p v`, proper complex Gaussian
spectral voltages give `Cov(q_p,q_q)=Tr(A_p S A_q S)/N`. Here `N=B tau`
for B in Hz and tau in seconds. Production pseudo-Stokes and all 16 real
science products are represented at the same voltage/kelvin reference
plane. Every inter-product covariance is retained.

For normalized fine-frequency power weights W, the channel covariance is
`sum_f W_bf W_cf Tr(A_p S_f A_q S_f)/(df tau)`. This includes frequency
variation in system power and the folded parent/zoom overlaps. It assumes
independent fine spectral estimates and long integration; finite temporal
filter correlations require a separate check. Nominal bin spacing is not
the effective bandwidth.

Independent checks include direct Kirchhoff voltage/current samples,
Gaussian quadrature that integrates all fourth moments exactly, luseepy's
individual variances, square-root integration scaling and channelized
complex-voltage Monte Carlo with frequency-dependent system power.

## Adopted receiver inputs

[Tamura et al. (2024)](https://doi.org/10.1029/2023RS007925), Section 3.1 and
Figure 8, report PA input-referred voltage noise near 2.2 nV/√Hz above
1 MHz with a 50 Ω termination. Section 3.3 and Figure 15 report integrated
EM noise near 3 nV/√Hz and a 5 nV/√Hz requirement. The gain has already been
removed in these input-referred values. They are distinct bench/design
cases, not a probability interval or the full flight noise covariance.
Extending them to the coupled antenna load is an assumption. A low-source-
impedance voltage test does not separate current noise or e/i correlation.

`receiver-noise.json` therefore uses three voltage cases and an explicitly
illustrative 0.03/0.1/0.3 pA/√Hz current bracket, with zero cross-port/e/i
correlations as an assumption. The current contribution under the 50 Ω
bench condition is small compared with the adopted voltage scale; it is
not fitted to that voltage measurement. Moon/antenna-loss temperatures
are separately declared and must be added once to sky system power.

The legacy `Throughput` class depends on absent Model54 noise/gain files
and a legacy scalar beam. It is not used. The new model imports only the
existing luseepy loading physics and does not modify either dependency.
