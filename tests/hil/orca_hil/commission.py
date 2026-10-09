"""One-off fixture commissioning: pick base TX gain and the attenuator-check settings.

    python -m tests.hil.orca_hil.commission [--rx-gain 40] [--write]

Sweeps RF0.tx_gain at a fixed rx_gain, measuring raw max |x| and single-pulse
compressed SNR. The proposed base tx_gain is the one closest to raw max |x| in
0.1-0.3 FS and SNR in 20-30 dB; the attenuator check uses base - 20 dB. With
--write the values go into the fixture file (comments are preserved); a human
must review them before they are committed.
"""
import argparse
import shutil
from datetime import datetime

import numpy as np

from . import HIL_DIR, REPO_ROOT, configs, scratch, yamlio
from .analysis.profile import analyze_profile
from .lock import SdrLock
from .radar import RadarRunner, build_radar

RAW_TARGET = (0.1, 0.3)
SNR_TARGET = (20.0, 30.0)
ATTEN_CHECK_OFFSET_DB = 20


def _score(raw_max, snr):
    """Distance (in dB) outside the target windows; 0 when both are met."""
    def outside(v, lo, hi):
        return max(lo - v, 0, v - hi)
    raw_db = 20 * np.log10(max(raw_max, 1e-12))
    return outside(raw_db, *20 * np.log10(RAW_TARGET)) + outside(snr, *SNR_TARGET)


def sweep(fixture, rx_gain, tx_gains, n_pulses, workdir):
    runner = RadarRunner(fixture)
    clip = fixture["safety"]["rx_clip_fraction_max"]
    rows = []
    for g in tx_gains:
        if g > fixture["safety"]["tx_gain_max"]:
            break
        overrides = {"RF0": {"tx_gain": float(g), "rx_gain": float(rx_gain)},
                     "CHIRP": {"num_pulses": n_pulses}}
        cfg_rel = configs.build_config(fixture, overrides, workdir / f"tx{g:g}")
        res = runner.run(cfg_rel)
        _, ms = analyze_profile(res.zarr, fixture)
        m = {x.name: x.value for x in ms}
        row = {"tx_gain": float(g), "raw_max": m["raw_max_abs_fs"], "snr_db": m["single_pulse_snr_db"],
               "loopback_dbfs": m["loopback_peak_power_dbfs"], "noise_dbfs": m["noise_floor_dbfs"],
               "separation": m["peak_separation_samples"],
               "errors": res.summary["error_count"], "attempted": res.summary["pulses_attempted"]}
        rows.append(row)
        print(f"tx_gain {g:5.1f}  raw_max {row['raw_max']:.4f}  snr {row['snr_db']:5.1f} dB  "
              f"loopback {row['loopback_dbfs']:6.1f} dBFS  noise {row['noise_dbfs']:6.1f} dBFS  "
              f"sep {row['separation']:.2f}  errors {row['errors']}/{row['attempted']}", flush=True)
        shutil.rmtree(REPO_ROOT / cfg_rel.parent, ignore_errors=True)
        if row["raw_max"] > clip:
            print("raw max exceeds rx_clip_fraction_max; stopping sweep")
            break
    return rows


def propose(rows):
    best = min(rows, key=lambda r: (_score(r["raw_max"], r["snr_db"]), -r["tx_gain"]))
    att_gain = best["tx_gain"] - ATTEN_CHECK_OFFSET_DB
    gains = np.array([r["tx_gain"] for r in rows])
    levels = np.array([r["loopback_dbfs"] for r in rows])
    expected = float(np.interp(att_gain, gains, levels)) if gains.min() <= att_gain else None
    return best, att_gain, expected


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fixture", default=None)
    p.add_argument("--rx-gain", type=float, default=40.0)
    p.add_argument("--tx-start", type=float, default=10.0)
    p.add_argument("--tx-stop", type=float, default=75.0)
    p.add_argument("--tx-step", type=float, default=5.0)
    p.add_argument("--pulses", type=int, default=2000)
    p.add_argument("--write", action="store_true", help="write proposed values to the fixture file")
    args = p.parse_args(argv)

    files = sorted((HIL_DIR / "fixtures").glob("*.yaml"))
    fx_path = HIL_DIR / "fixtures" / f"{args.fixture}.yaml" if args.fixture else files[0]
    fixture = yamlio.load(fx_path)
    data_root = fixture["host"]["data_root"]
    scratch.ensure_scratch_link(data_root)
    workdir = REPO_ROOT / scratch.SCRATCH_LINK_NAME / f"commission_{datetime.now():%Y%m%dT%H%M%S}"

    with SdrLock(f"{data_root}/orca-hil.lock"):
        build = build_radar(fixture)
        print(f"radar built, libuhd: {build.libuhd}")
        gains = np.arange(args.tx_start, args.tx_stop + 1e-9, args.tx_step)
        try:
            rows = sweep(fixture, args.rx_gain, gains, args.pulses, workdir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    best, att_gain, expected = propose(rows)
    print(f"\nproposed rf_base: tx_gain {best['tx_gain']:g}, rx_gain {args.rx_gain:g} "
          f"(raw max {best['raw_max']:.3f} FS, single-pulse SNR {best['snr_db']:.1f} dB)")
    print(f"proposed attenuator check: tx_gain {att_gain:g}, expected peak "
          f"{'n/a' if expected is None else f'{expected:.1f} dBFS'}")
    if args.write:
        data = yamlio.load(fx_path, roundtrip=True)
        data["rf_base"]["tx_gain"] = best["tx_gain"]
        data["rf_base"]["rx_gain"] = args.rx_gain
        data["safety"]["attenuator_check"]["tx_gain"] = att_gain
        data["safety"]["attenuator_check"]["expected_peak_dbfs"] = None if expected is None else round(expected, 1)
        yamlio.dump(data, fx_path, roundtrip=True)
        print(f"wrote {fx_path.relative_to(REPO_ROOT)} (review before committing)")
    return rows


if __name__ == "__main__":
    main()
