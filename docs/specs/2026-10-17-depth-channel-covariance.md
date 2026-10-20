# Depth-mixed spherical channel covariance

The depth profile describes the same intrinsic spin-2 field multiplied by
`A_x(G)=integral w(f) exp[2 i x f (mu+sigma G)] df`, with x=lambda²,
unit-integral nonnegative w, and a shared Gaussian column G. Emission
depths remain correlated.

Using the probabilists' Hermite generating function in
[DLMF 18.12.16](https://dlmf.nist.gov/18.12.E16), the normalized coefficients
are

```
c_k(x) = integral w(f) exp(2 i mu x f - 2 sigma² x² f²)
         (2 i sigma x f)^k / sqrt(k!) df.
```

For correlated unit Gaussians, the
[Mehler identity, DLMF 18.18.28](https://dlmf.nist.gov/18.18.E28)
gives `E[h_k(G) h_j(G')]=delta_kj rho(theta)^k`. Therefore the raw phase
covariance and relation are sums of `c_k(x) conj(c_k(y)) rho^k` and
`c_k(x) c_k(y) rho^k`. Mode zero gives the smooth term; higher modes give
the centered term. This is the same double-depth average as the Phase 1
integral, evaluated as a separable series.

Apply the actual channel operator and intrinsic spectral amplitude to each
coefficient before forming moments. No fine-frequency-pair grid or
frequency stationarity assumption is required. `DepthChannelProjector`
integrates `rho^k` through the existing spin-2/readout angular weights,
retaining both complex moment types and all helicity/pair terms.

For one emitting depth, coefficient power is a Poisson mass with mean
`(2 sigma x f)^2`. Jensen's inequality bounds the mixed profile's omitted
power by the largest-depth Poisson tail. The solver chooses a degree with
tail at most 1e-12 and rejects a requested maximum degree that cannot meet
it. Minkowski and Cauchy–Schwarz propagate this into an absolute channel
covariance bound. This bounds mode truncation, not angular quadrature.

Depth order doubling is checked in coefficient L2 norm, beginning at the
phase-resolving order used in Phase 1. Logarithmic Poisson coefficients
recover surviving power even when the mean underflows. At high degree a
deviance/Stirling form avoids cancellation in `k log(k)` terms; the
correction uses [DLMF 5.11](https://dlmf.nist.gov/5.11). No physical amplitude
is normalized by a numerical sky pixel or coherence cell.

Tests use exact fractional screens, independent bivariate Gaussian
averages of the analytic slab sinc response, an adaptive truncated-Gaussian
depth integral and explicit Phase 1 double-depth covariance through the
same angular/readout estimator. A deliberately truncated series checks
the omitted-mode bound. Underflow and unresolved-depth controls must
reject a broken or uncertified result.

`mixed_depth_predictions.py compute` produces the 30 native-band and 20
channel cases in `mixed-depth.json`. Every smooth/centered/total component
passes its own comparison with refined depth order, doubled angular
resolution and lmax 64/96. The measured-bin cases also compare 12.2/24.4 Hz
fine grids and the f=1 cases compare the independent previous frequency-pair
solver. Mode-tail bounds are propagated to the final readout covariance.
`plot_mixed_depth.py` regenerates the six-panel real-instrument figure
without invalidating numerical caches when its presentation changes.
