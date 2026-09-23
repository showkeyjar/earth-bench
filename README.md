# EarthBench — Earth Decision Intelligence Benchmark

**地球决策智能基准测试框架 | Decision > Prediction**

---

## Quick Links

- **Daily Risk Dashboard**: [https://earth-ai.fun](https://earth-ai.fun)
- **RSS Feed**: [feed.xml](feed.xml)
- **JSON API**: [latest.json](latest.json)
- **History**: [history.json](history.json)

## What is EarthBench?

EarthBench is the world's first benchmark testing whether AI can make **real-world binary decisions** about environmental risks (fire, flood, drought, heatwave) using multi-source spatio-temporal observations.

We don't just predict numbers — we decide actions: *Should we activate Level 1 fire response today?*

### Key Features

- **Decision-first paradigm** — not prediction accuracy, but decision quality
- **Four hazard categories**: Fire, Flood, Drought, Heatwave
- **Five decision templates**: Alert, Deploy, Upgrade, Close, Recover
- **Multi-source data fusion**: Satellite (MODIS/FIRMS), meteorological stations (QWeather), hydrological sensors
- **Transparent reasoning**: Every decision includes full evidence chain and LLM inference trace
- **Ground truth verification**: Each decision verified against national/international thresholds
- **Probabilistic forecasting (CARS)** — 30-member ensemble heat probabilities from a frozen, 15-year-validated statistical model; agents can reason over forecast uncertainty, not just current observations
- **Economic value scoring (V)** — decisions scored by cost-loss value (money), not only accuracy/F1

### CARS Integration (probabilistic decision layer)

Integrated with the CRPS/CARS research project (`D:\code\ai\CRPS`):

| Component | What it does |
|-----------|--------------|
| `earthbench/cars_agent.py` | `CarsHeatAgent` / `CarsMultiAgent` — **impact-first rule**: alert iff P(harmful) ≥ C/L (expected-loss rule); harm basis = wet-bulb ≥ 27°C (EarthBench standard) with dry-bulb fallback; +2σ relative anomaly is severity context only, never a trigger |
| `earthbench/cars_serve.py` | Standalone frozen-model inference (deep NGR direct-load via `crps_cars`, no replica) + daily GEFS f048 fetch (t2m/r2/tmax/tmin) → harm probabilities (wet-bulb Stull, daily max 35/37°C national lines, daily min ≤0°C) + backfill mode (`--valid YYYY-MM-DD`) |
| `earthbench/cars_serve_impact.py` | **Impact channel (rain/wind)**: SeasonalCars fitted on frozen 2000–2016 archives at load (honest-selected modal hyperparams) + ops GEFS APCP f054–f072 6h-window sum (24h accumulation) & UGRD/VGRD10 5-snapshot mean → P(≥50mm), P(≥100mm), P(daily-mean wind ≥10.8/13.9 m/s). Tail policy per miss-depth criterion v2: tp = high-tail completion (deep rupture, miss 49%→23%), wind = raw (shallow, calibration wins) |
| `earthbench/cars_verify.py` | **Verification closed loop**: score past forecasts against next-day observations (QWeather now-temp preferred, GEFS f000 fallback), accumulate Brier / hit / miss / false-alarm / CSI per city |
| `earthbench/cars_verify_impact.py` | **Impact verification loop** (rain/wind): model-anchored obs (valid-day 00z f006–f024 APCP sum = 24h rain, f000–f024 5-snapshot wind mean) → Brier / hit-miss-false-alarm / CSI / reliability bins per variable; absolute calibration delegated to the CRPS-side ERA5 out-of-sample audit |
| `earthbench/data/cars_cities.json` | Ops-editable city config (12 cities, per-city cold-alert gating for warm-winter cities) |
| `earthbench/eval.py::ValueEvaluator` | Cost-loss economic value scoring: V = (E_clim − E_agent)/(E_clim − E_perfect) |
| `earthbench/data/cars_t2m_deep_ngr.npz` | Frozen deep-NGR model (2000–2015 train, 2016 early-stop + GPD tail), CRPS ~0.97 K, +7.9% CRPSS vs seasonal (rolling 15y) |
| `earthbench/data/cars_climo.npz` | Monthly climatology + grid (for the +2σ relative-anomaly threshold) |
| `scripts/run_agent_comparison.py` | rule vs cars comparison + P_TRIG threshold sensitivity (expected-loss trade-off) |
| Daily CI | verify step (score past forecasts) → update step (publish new probabilities); history round-trips via gh-pages (`CARS_DATA_DIR=published_reports`); grib cache stays in-package (never deployed) |

Run the comparison:

```bash
python scripts/run_agent_comparison.py            # accuracy + V score + sensitivity
python -m earthbench --benchmark --agent cars     # full benchmark with CARS agent
python -m earthbench --benchmark --adversarial    # adversarial hard-case suite (rule baseline < 100%)
python -m earthbench.cars_serve                   # regenerate daily probabilities
python -m earthbench.cars_serve --valid 2026-09-19  # backfill (GEFS retention ~10 days)
python -m earthbench.cars_verify                  # closed-loop verification of past forecasts
```

**Adversarial suite (discriminative power)**: the 20 core cases score the rule baseline at 100% — they cannot rank agents that beat the baseline. `scenarios.py::get_adversarial_suite()` adds 8 L4 cases built on decision boundaries where linear weighted scoring systematically diverges from the national-standard lookup tables, in **both directions**: 4 false-alarm traps (sub-threshold factor mixes, AND-conditions missing one input, negated confirmations off by a hair) and 4 miss traps (a decisive factor diluted by normal ones — cold-dry fire weather, saturation-AND at exact thresholds, SPI alone at the drought line, muggy-OR without duration). The rule agent fails all 8 (0%, both FP and FN), while ground truth is still derived by the same independent `_gt_fn` standards. An agent that reasons over AND/OR structure instead of averaging can now measurably separate itself.

Pipeline cadence: D 08:00 publish probabilities for D+2 → D+3 CI scores them against observations (Brier/CSI accumulate in `cars_verification_history.json`, surfaced in the daily panel and `latest.json`).

Disclosed caveats: members are daily-mean t2m perturbations with diurnal peak/valley offsets and humidity taken unperturbed from c00; wet bulb via Stull (2011), typical error <1°C; deep NGR frozen on 2000–2015 training (2016 early-stop + EVT tail); ops GEFS vs reforecast v12 version drift; coverage 100–140°E / 20–50°N (out-of-grid falls back to a 37°C rule with reduced confidence); verification v1 observes via QWeather now-temp (20:00 Beijing ≈ daily-max proxy, 1–3°C error) or GEFS f000 (model-anchored). Impact-first principle: only human-harmful extremes are reported (wet bulb ≥ 27°C / daily max ≥ 35/37°C / daily min ≤ 0°C for warm-winter cities); relative +2σ anomalies are severity context, not triggers.

### Architecture

```
Scenarios (20 test cases × 4 categories × 4 difficulty levels)
    ↓
Agents (Rule-based baseline + LLM Agent via CARM framework)
    ↓
Benchmark Engine (Accuracy per category/difficulty)
    ↓
Publish Pipeline (Markdown + JSON + RSS + Web Dashboard)
```

## Project Structure

```
earthbench/
├── models.py            # Core data structures (Observation, ScenarioContext, DecisionOutput)
├── scenarios.py          # 20 benchmark test cases + adversarial suite + ScenarioStore
├── agents.py             # Rule-based Agents (Fire/Flood/Drought/HeatWave + MultiAlertRouter)
├── benchmark.py          # AlertBenchEvaluator — full benchmark engine
├── eval.py               # BaseEvaluator + BatchEvaluator (accuracy/P/R/F1 + Brier/ECE + ValueEvaluator)
├── weather.py            # Shared meteorology (Stull wet-bulb + inverse, single source)
├── llm.py                # Shared LLM YES/NO parsing + domain prompts (single source)
├── verification.py       # Closed-loop delayed verification (FIRMS/QWeather historical)
├── calibration.py        # Threshold self-tuning from verification feedback (EWMA)
├── templates.py          # DecisionTemplate engine + context validation
├── data_collectors.py    # QWeather API + NASA FIRMS satellite fire detection
├── enhance_data.py       # Data enhancement & enrichment pipeline
├── publish_pipeline.py   # 4-stage publish: collect → decide → report → distribute
├── integrations.py       # CARM/Mustard LLM bridge (Ollama/Qwen3)
└── __main__.py           # CLI entry point
tests/
└── test_evaluation.py    # 65 test cases covering all modules
.github/workflows/
├── daily-report.yml      # CI: daily pipeline + deploy to gh-pages + Cloudflare
└── auto-post.yml         # CI: auto-post to social media
```

## Quick Start

### Installation

```bash
git clone https://github.com/showkeyjar/earth-bench.git
cd earth-bench
pip install -e .
```

### Run Demo

```bash
# Single-scenario demo (forest fire)
python -m earthbench --demo

# Full benchmark (all 4 categories)
python -m earthbench --benchmark

# Filter by category
python -m earthbench --benchmark --category fire

# Eval mode (from JSON input)
python -m earthbench --eval --eval-input scenario.json
```

### Publish Pipeline

```bash
# Full pipeline (collect → LLM decide → report → distribute)
python -m earthbench.publish_pipeline --run

# Dry-run (no actual publication)
python -m earthbench.publish_pipeline --run --dry-run

# Test single scenario with CARM/LLM
python -m earthbench.publish_pipeline --test-publish
```

### Run Tests

```bash
python -m pytest tests/ -v
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `QWEATHER_API_KEY` | QWeather API key for weather data | (empty — falls back to scenario data) |
| `FIRMS_MAP_KEY` | NASA FIRMS API key for satellite fire detection | (empty — satellite detection disabled) |
| `OLLAMA_MODEL` | Ollama model name for LLM decisions | `qwen3:14b` |

All API keys default to empty strings. The framework gracefully degrades to built-in scenario data when keys are not provided.

## Scenario Categories

| Category | Key Variable | Alert Threshold | Source |
|---|---|---|---|
| Fire | FWI (Fire Weather Index) | >= 40 | ECMWF + MODIS |
| Flood | rainfall_24h | >= 100mm | QWeather + hydrological |
| Drought | SPI (Standardized Precipitation Index) | <= -1.5 | ECMWF + NDVI |
| Heatwave | wet_bulb_temp | >= 27 degrees C | QWeather |

## Difficulty Levels

- **L1 (Easy)**: Clear-cut extreme values, straightforward decision
- **L2 (Medium)**: Borderline values with secondary factors
- **L3 (Hard)**: Conflicting signals, requires multi-variable reasoning
- **L4 (Beyond)**: Edge cases that expose agent limitations

## Development Guide

### Adding a New Scenario

1. Add test case to `scenarios.py` in `get_alert_benchmark_suite()`
2. Include `case_id`, `difficulty`, `category`, `region`, `ground_truth`, `observations`
3. Ensure the `region` is registered in `REGION_LOCATION_MAP` (in `data_collectors.py`)
4. Run tests: `python -m pytest tests/ -v -k "BenchmarkSuite"`

### Adding a New Agent

1. Implement the `decide(context: ScenarioContext) -> DecisionOutput` interface
2. Register in `MultiAlertAgent.decide()` category_map if introducing a new category
3. Add corresponding test cases in `tests/test_evaluation.py`

### CI/CD

The project uses GitHub Actions for:
- **Daily pipeline**: Runs at 08:00 UTC, collects data, generates reports, deploys to GitHub Pages + Cloudflare
- **Auto-post**: Publishes summaries to social media platforms

## Contact

- Email: zergskj@163.com
- GitHub: https://github.com/showkeyjar/earth-bench
- Website: https://earth-ai.fun

## License

CC BY 4.0
