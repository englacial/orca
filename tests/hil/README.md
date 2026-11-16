# Hardware-in-the-loop (HIL) tests

Tests that run the radar on a real SDR in RF loopback and check the results
against physics-derived bounds and per-fixture baselines. The current fixture
is a B206mini-i on a Raspberry Pi 5, TX/RX → 30 dB pad → 50 m coax → RX2
(`fixtures/b206mini_pi5_loopback_50m.yaml`).

## Running

From the repo root, with the ORCA conda env active
(`conda env update -n <env> -f environment-hil.yaml` adds the test dependencies):

```bash
make hil-unit                          # software-only tests (simulated captures), no hardware
make hil-fast                          # fast tier on hardware (<= 10 min)
make hil-full                          # full tier (a few hours)
make hil-fast HIL_ARGS="-m 't00 or t01'"   # a subset
```

Hardware tests only run with `--hil` (the make targets pass it); otherwise they
are skipped. T0 (preflight) always runs first, and if it fails every other
hardware test is skipped.

Options: `--tier fast|full`, `--fixture <id or path>`, `--keep-raw` (keep raw captures
even for passing tests), `--results-root <dir>`, `--setup-note "<text>"` (log an
undeclared hardware change; the run is marked non-standard).

## Fixtures and non-standard setups

A fixture's `hardware.chain` lists the RF parts from SDR TX to SDR RX in order
(types `cable`, `adapter`, `attenuator`, `limiter`, `filter`, `amplifier`; fields
are checked in `orca_hil/fixtures.py`). Numeric fields may be a frequency table
`{freq_mhz: [...], value: [...]}`, and `source: datasheet|measured` records where
a value came from.

To test added parts, write a fixture that extends a commissioned one and restates
the whole chain. It can live outside the repo and be passed by path:

```yaml
fixture_id: loopback_50m_rx_lna
extends: b206mini_pi5_loopback_50m
judge: info                       # status defaults to exploratory
hardware:
  chain:
    - {id: pad30, type: attenuator, loss_db: 30}
    - {id: cable50, type: cable, model: LMR-100, length_m: 50.0, velocity_factor: 0.66}
    - {id: lna, type: amplifier, role: lna, gain_db: 20, nf_db: 1.0, max_input_dbm: 15, powered: true}
```

Runs on a fixture without `status: commissioned`, or with `--setup-note`, get
`-nonstd` in the run id and `standard: false` in `run.yaml`, and cannot update
baselines.

## UHD

The fixture uses a UHD built from `~/uhd` by `scripts/build_uhd.sh`
(`make hil-build-uhd`), never conda's or apt's. T0 checks that the radar
binary links against it and that the build matches the source checkout.

## Outputs

Each session writes `<data_root>/runs/<UTC time>_<sha>[-dirty]_<tier>/` with
`run.yaml`, `configuration.yaml` (host, SDR and fixture snapshot), `junit.xml`,
and one directory per test holding `metrics.yaml`, plots, `reduced.zarr.zip`,
the generated radar configs, logs and `telemetry.csv`.

Raw captures live under `<repo>/hil_scratch` (a symlink to
`<data_root>/scratch`, needed because the radar resolves output paths relative
to the repo). They are deleted when a test passes and moved to
`<data_root>/kept/` when it fails, within `storage.kept_raw_quota_gb`.

## Publishing results

`--publish` commits the run to the results repo clone (`host.results_repo` in the
fixture) and pushes it. Its site builds on GitHub Pages. To publish existing runs,
or to build the site locally:

```bash
python -m tests.hil.orca_hil.publish [--no-push] <run_dir>...
make hil-site    # writes $(HIL_SITE), default /mnt/nvme/orca-hil/site
```

## Thresholds and baselines

Every metric is listed in `thresholds.yaml` as `abs` (valid on any healthy
fixture), `base` (relative to this fixture's baseline in `baselines/`) or
`info`. Without a baseline, `base` metrics are reported as inconclusive.

Baselines are kept per tier. To re-measure them (clean git tree required; commit
the result through a PR):

```bash
make hil-fast HIL_ARGS="--update-baseline --baseline-repeats 5"
make hil-full HIL_ARGS="--update-baseline --baseline-repeats 3"
```

## Commissioning a fixture

```bash
python -m tests.hil.orca_hil.commission            # TX gain sweep, prints proposed values
python -m tests.hil.orca_hil.commission --write    # also writes them to the fixture file
```

Review the proposed `rf_base` and `safety.attenuator_check` values before committing.
