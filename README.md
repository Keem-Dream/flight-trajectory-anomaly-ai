[README (1).md](https://github.com/user-attachments/files/28639377/README.1.md)
# Flight Trajectory Anomaly AI

Detecting spoofed and anomalous aircraft using machine learning on real ADS-B flight data.

## Overview

Aircraft broadcast their own position, altitude, and velocity over a system called **ADS-B**. The critical weakness: these broadcasts are **unauthenticated** — nothing signs or verifies them, so ground receivers trust whatever a transponder reports. An attacker with a transmitter can inject fake aircraft, lie about a real one's position, or replay old data, with no built-in way to tell a forged broadcast from a real one.

This project builds a software system that uses machine learning to decide whether ADS-B data is trustworthy, flagging flights that are likely spoofed, malfunctioning, or otherwise anomalous. It pairs two models with opposite strengths into an ensemble that stays robust even against attack patterns it was never trained on.

This is a **classical machine learning** project (tree-based models from scikit-learn), not deep learning. Classical ML was chosen because it fits tabular data well and trains in seconds — the right tool for per-flight feature data.

## Data

- **Source:** [OpenSky Network](https://opensky-network.org) — free, research-grade, real-world ADS-B data collected from receivers worldwide. No hardware required; data is pulled in pure software via the `pyopensky` library.
- **Collection method:** polling live aircraft states every ~20 seconds and stitching snapshots into per-aircraft trajectories using the unique `icao24` transponder ID.

## Pipeline

The project runs as a sequence of stages:

1. **Collect** — pull live aircraft states from OpenSky.
2. **Build trajectories** — group snapshots by `icao24` and order them over time, so motion (not just position) is visible.
3. **Engineer features** — convert each trajectory segment into the measurements that expose spoofing:
   - `dt_s` — time between readings
   - `dist_km` — distance traveled (great-circle / haversine)
   - `speed_ms` — implied ground speed
   - `climb_ms` — climb/descent rate
4. **Inject labeled fakes** — generate synthetic spoofed flights with known labels, so detector performance can be measured.
5. **Train and evaluate** — train detectors, score them against the known fakes.

## Models

Two models with deliberately opposite approaches:

**Isolation Forest (unsupervised — the generalist).**
Learns what normal flight looks like as a cluster and flags anything sitting outside it. It is never told what "fake" means; it only detects "doesn't fit the crowd." This makes it blunt but robust — it can catch novel attacks it has never seen.

**Random Forest (supervised — the specialist).**
Trained on labeled examples (fake vs real), it learns the boundary that separates them. This makes it precise on attack types it has seen, but blind to anything new. Evaluated with a train/test split (70/30) so it is scored only on flights it never trained on.

**Ensemble.**
The two are combined with the rule: flag a flight if *either* model catches it. Because the models fail in different ways — the Random Forest misses novel attacks, the Isolation Forest catches them — the combination covers each other's blind spots and is more robust than either alone.

## Results

The project followed a build → break → strengthen → engineer arc:

| Stage | Test | Result |
|---|---|---|
| Isolation Forest (unsupervised baseline) | Known attacks | ~75% recall |
| Random Forest (supervised) | Known attacks (held-out test set) | ~100% recall |
| Random Forest | **Novel** attack types it never trained on | **0%** |
| Random Forest (retrained on varied attacks) | Those attack types | ~99% |
| Random Forest alone | A truly novel 8th attack type | 0% |
| Isolation Forest alone | That same novel attack | 100% |
| **Ensemble (either model)** | That same novel attack | **100%** |

## Key Insights

**A supervised model only knows the attacks it was shown.** The Random Forest scored ~100% on known attacks but collapsed to 0% the instant it faced a pattern it hadn't trained on. Adding those patterns to training fixed that specific gap (0% → 99%) — but only for the attacks anticipated. There is always another unanticipated attack.

**A near-perfect score is a warning, not just a trophy.** The supervised model hit ~100% because the injected fakes followed a simple, learnable rule. That score reflects how well it learned *the injector's pattern*, not how it would perform against a real attacker.

**Robustness comes from combining models that fail differently.** The Isolation Forest, which hunts for generic anomalies rather than specific attack types, caught a completely unforeseen attack (100%) that the Random Forest missed entirely (0%). The ensemble inherits this robustness — engineering around the fundamental limit of supervised learning rather than pretending it away.

## Guardrails / Ethics

This is **defensive** security research. The project only receives and analyzes public data and works with synthetic, self-generated fakes in a lab setting. It never transmits, jams, or interferes with real aircraft or satellite signals — doing so is illegal. The boundary: build, simulate, and receive freely; never transmit against live systems.

## Tech Stack

- Python 3
- `pyopensky` — OpenSky Network data access
- `pandas` — data handling
- `scikit-learn` — Isolation Forest, Random Forest, train/test split, metrics

## Possible Next Steps

- **LLM reporting layer** — feed flagged flights to an LLM to produce plain-language threat reports, turning raw detections into analyst-ready output.
- **Sequence features** — add features that capture a whole trajectory's shape (e.g. heading consistency), to catch attacks that are plausible point-to-point but impossible overall.
- **LSTM (deep learning)** — rebuild detection as a sequence model in PyTorch to catch sequence-based attacks single-segment features can't, benchmarked against the classical baseline above.
- **Richer real data** — collect across different times and regions to build a more representative notion of "normal."
