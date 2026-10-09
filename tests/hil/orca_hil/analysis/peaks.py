"""Two-peak model fit of a compressed range profile, and mainlobe measurements.

The model is a1 * R(t - t1) + a2 * R(t - t2), where R is the ideal compressed
response (the reference chirp's autocorrelation, normalized so R(0) = 1) and
a1, a2 are complex. R is evaluated at fractional lags by band-limited (DFT)
interpolation, so the fit runs directly on the raw-rate compressed samples.
"""
from dataclasses import dataclass

import numpy as np

from .common import db10, upsample


class ChirpResponse:
    """Ideal compressed response R(lag) of a reference chirp."""

    def __init__(self, chirp, up=64):
        c = np.asarray(chirp, complex)
        self.chirp_len = len(c)
        self.N = int(2 ** np.ceil(np.log2(2 * len(c) + 64)))
        energy = np.sum(np.abs(c) ** 2)
        self.S = np.abs(np.fft.fft(c, self.N)) ** 2 / energy
        self.k = np.fft.fftfreq(self.N) * self.N
        # Table of R at lags j / up (circular), matching ``at`` exactly
        self.up = up
        N, h = self.N, self.N // 2
        pad = np.zeros(N * up, complex)
        pad[:h] = self.S[:h]
        pad[-h:] = self.S[h:]
        self._table = np.fft.ifft(pad) * up

    def at(self, lags):
        """R at arbitrary (fractional) lags."""
        lags = np.asarray(lags, float)
        ph = np.exp(2j * np.pi * np.multiply.outer(lags, self.k) / self.N)
        return ph @ self.S / self.N

    def table(self, lags):
        """R at lags that are multiples of 1/up (fast lookup)."""
        j = np.rint(np.asarray(lags) * self.up).astype(np.int64) % (self.N * self.up)
        return self._table[j]


@dataclass
class TwoPeakFit:
    t1: float
    t2: float
    a1: complex
    a2: complex
    region: slice
    model: np.ndarray            # model over ``region``
    residual_ratio_db: float     # residual energy / model energy over region


def _parabolic(y, i):
    if 0 < i < len(y) - 1:
        den = y[i - 1] - 2 * y[i] + y[i + 1]
        if den > 0:
            return i + 0.5 * (y[i - 1] - y[i + 1]) / den
    return float(i)


def fit_two_peaks(profile, resp: ChirpResponse, t1_0, t2_0, half_width=None,
                  span=1.5, step=1 / 16):
    """Fit the two-peak model near initial positions t1_0 < t2_0.

    The fit (and the residual ratio) uses the region from t1 - half_width to
    t2 + half_width, clipped to the profile; half_width defaults to the chirp length.
    """
    y = np.asarray(profile, complex)
    hw = resp.chirp_len if half_width is None else half_width
    region = slice(max(int(np.floor(t1_0 - hw)), 0), min(int(np.ceil(t2_0 + hw)) + 1, len(y)))
    m = np.arange(region.start, region.stop)
    yr = y[region]

    offs = np.arange(-span, span + step / 2, step)
    d1 = np.round((t1_0 + offs) * resp.up) / resp.up
    d2 = np.round((t2_0 + offs) * resp.up) / resp.up
    A1 = resp.table(m[None, :] - d1[:, None])
    A2 = resp.table(m[None, :] - d2[:, None])
    G11 = np.sum(np.abs(A1) ** 2, axis=1)
    G22 = np.sum(np.abs(A2) ** 2, axis=1)
    G12 = A1.conj() @ A2.T
    b1 = A1.conj() @ yr
    b2 = A2.conj() @ yr
    det = G11[:, None] * G22[None, :] - np.abs(G12) ** 2
    q = (G22[None, :] * np.abs(b1[:, None]) ** 2 + G11[:, None] * np.abs(b2[None, :]) ** 2
         - 2 * np.real(G12 * b1.conj()[:, None] * b2[None, :])) / det
    res = np.sum(np.abs(yr) ** 2) - q
    i, j = np.unravel_index(np.argmin(res), res.shape)
    t1 = d1[0] + step * _parabolic(res[:, j], i)
    t2 = d2[0] + step * _parabolic(res[i, :], j)

    A = np.stack([resp.at(m - t1), resp.at(m - t2)], axis=1)
    (a1, a2), *_ = np.linalg.lstsq(A, yr, rcond=None)
    model = A @ np.array([a1, a2])
    ratio = db10(np.sum(np.abs(yr - model) ** 2) / np.sum(np.abs(model) ** 2))
    return TwoPeakFit(float(t1), float(t2), complex(a1), complex(a2), region, model, float(ratio))


def leakage_removed(profile, resp, fit: TwoPeakFit):
    """Profile with the fitted internal-leakage term subtracted."""
    n = np.arange(len(profile))
    return np.asarray(profile, complex) - fit.a1 * resp.at(n - fit.t1)


@dataclass
class MainlobeMetrics:
    peak_pos: float
    peak_power: float
    width_3db: float         # samples
    pslr_db: float           # peak sidelobe / peak
    islr_db: float           # sidelobe energy / mainlobe energy
    extent: float | None     # farthest offset where |x|^2 > floor (samples), if floor given


def mainlobe_metrics(x, pos, sidelobe_half_width, up=8, floor_power=None, exclude=(), exclude_half_width=2.0):
    """Measure the response peak near ``pos`` in x (raw-rate complex samples).

    Sidelobes are taken within +/- sidelobe_half_width samples of the peak,
    outside the mainlobe (bounded by the first nulls on each side), and outside
    +/- exclude_half_width samples around each offset in ``exclude`` (e.g. echoes).
    """
    lo = max(int(np.floor(pos - sidelobe_half_width)) - 8, 0)
    hi = min(int(np.ceil(pos + sidelobe_half_width)) + 9, len(x))
    seg = np.asarray(x[lo:hi], complex)
    p = np.abs(upsample(seg, up)) ** 2
    t = lo + np.arange(len(p)) / up
    c = int(np.argmin(np.abs(t - pos)))
    w = 2 * up
    s0 = max(c - w, 0)
    pk = s0 + int(np.argmax(p[s0:c + w + 1]))
    peak = p[pk]

    def crossing(direction):
        i = pk
        while 0 < i < len(p) - 1 and p[i] >= peak / 2:
            i += direction
        a, b = p[i - direction], p[i]
        frac = (a - peak / 2) / (a - b) if a != b else 0.0
        return t[i - direction] + direction * frac / up

    width = crossing(+1) - crossing(-1)

    def null(direction):
        i = pk
        while 0 < i < len(p) - 1 and p[i + direction] < p[i]:
            i += direction
        return i

    nl, nr = null(-1), null(+1)
    side = np.abs(t - t[pk]) <= sidelobe_half_width
    side[nl:nr + 1] = False
    for off in exclude:
        side &= np.abs(t - t[pk] - off) > exclude_half_width
    main_e = np.sum(p[nl:nr + 1])
    pslr = db10(p[side].max() / peak) if side.any() else -np.inf
    islr = db10(p[side].sum() / main_e) if side.any() else -np.inf
    extent = None
    if floor_power is not None:
        above = np.nonzero(p > floor_power)[0]
        extent = float(np.max(np.abs(t[above] - t[pk]))) if above.size else 0.0
    return MainlobeMetrics(float(t[pk]), float(peak), float(width), float(pslr), float(islr), extent)
