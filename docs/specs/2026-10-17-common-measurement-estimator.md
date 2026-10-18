# Common measured-spectrum estimator

Smooth templates are Legendre polynomials in physical fine frequency,
integrated through the same power-response matrix as the sky. This includes
the folded images of zoom bins. The default downstream comparison fits a
constant and slope separately to the three parents and 192 zoom bins.
Weights, flags, rank, conditioning and residual degrees of freedom are
exported. Saturated or rank-deficient fits raise an error.

Every component uses the same weighted linear residual map R. Real joint
covariance transforms on both frequency legs. When a structured coefficient
representation is available, apply R before forming its covariance to avoid
cancellation between large smooth terms. Q and U are subtracted in kelvin
before division by channelized I. A fixed external intensity reference and
an estimated intensity are separate modes: the latter retains I-Q/U
correlations with the full ratio Jacobian and rejects fractional intensity
noise above the declared linearization limit. Exact zero denominators give
undefined ratios, never infinite separation.

Six tests verify folded templates, weighted fits with flags, independent
finite-ensemble covariance, noisy ratios, operation order and undefined
cases. Each has a deliberately injected fault caught by its named test;
the hashes and outcomes are recorded in the backend validation record.

Observation exposure counts complete phase bins whose two edges and center
are all at solar altitude below the declared margin at the configured lunar
site. Electronics duty multiplies eligible integration. Zero exposure is an
unavailable measurement. A boxcar in sidereal time multiplies MEPA harmonic
kernels by sinc(m times bin duration / sidereal day), on every signal and
leakage calculation. The calendar start is a simulation coordinate, not a
mission launch prediction.
