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
- **Eight hazard categories**: Fire, Flood, Drought, Heatwave, Landslide, Typhoon, Cold Wave, Snow
- **Five decision templates**: Alert, Deploy, Upgrade, Close, Recover
- **Multi-source data fusion**: Satellite (MODIS/FIRMS), meteorological stations (QWeather), hydrological sensors
- **Transparent reasoning**: Every decision includes full evidence chain and LLM inference trace
- **Ground truth verification**: Each decision verified against national/international thresholds
- **Probabilistic forecasting (CARS)** — 30-member ensemble heat probabilities from a frozen, 15-year-validated statistical model; agents can reason over forecast uncertainty, not just current observations
- **Economic value scoring (V)** — decisions scored by cost-loss value (money), not only accuracy/F1
- **Exposure-aware action levels (Phase B)** — `ExposureProfile` (population density / land use / critical infrastructure / vulnerable groups / outdoor activity) modulates the action tier (`monitor`/`alert`/`dispatch`) via a disclosed physical-severity × exposure-class matrix, **without rewriting physical ground truth**; paired cases with identical observations but different exposure yield different action truths; exposure-weighted V scoring charges misses in high-exposure regions more heavily
- **Decision-template closure (Phase C)** — the full emergency chain, not just a single alert: `DISPATCH` (pre-position resources), `UPGRADE` (raise level), `CLOSE` (shut roads/venues), and `RECOVER` (lift warnings only after conditions persist N hours). A `_persist_for` duration primitive makes premature recovery — "stop the moment the rain stops" — a measurable human-cost failure

### CARS Integration (probabilistic decision layer)

Integrated with the CRPS/CARS research project ([github.com/showkeyjar/CRPS](https://github.com/showkeyjar/CRPS)，本地 `D:\code\ai\CRPS`):

| Component | What it does |
|-----------|--------------|
| `earthbench/cars_agent.py` | `CarsHeatAgent` / `CarsMultiAgent` — **impact-first rule**: alert iff P(harmful) ≥ C/L (expected-loss rule); harm basis = wet-bulb ≥ 27°C (EarthBench standard) with dry-bulb fallback; +2σ relative anomaly is severity context only, never a trigger |
| `earthbench/cars_serve.py` | Standalone frozen-model inference (deep NGR direct-load via `crps_cars`, no replica) + daily GEFS f048 fetch (t2m/r2/tmax/tmin) → harm probabilities (wet-bulb Stull, daily max 35/37°C national lines, daily min ≤0°C) + backfill mode (`--valid YYYY-MM-DD`) |
| `earthbench/cars_serve_impact.py` | **Impact channel (rain/wind/snow)**: SeasonalCars fitted on frozen 2000–2016 archives at load (honest-selected modal hyperparams) + ops GEFS APCP f054–f072 6h-window sum (24h accumulation) & UGRD/VGRD10 5-snapshot mean → P(≥50mm), P(≥100mm), P(daily-mean wind ≥10.8/13.9/**17.2/24.5** m/s — 8/10-grade GB/T 19201 tiers synced with the benchmark typhoon GT, Phase D) and P(snowfall ≥10/20mm water-equivalent via **APCP × freeze mask** t2m<0.5°C, GB/T 28592-2012 calibre, disclosed deterministic c00 mask). Tail policy per miss-depth criterion v2: tp = high-tail completion (deep rupture, miss 49%→23%), wind = raw (shallow, calibration wins) |
| `earthbench/cars_verify.py` | **Verification closed loop**: score past forecasts against next-day observations (QWeather now-temp preferred, GEFS f000 fallback), accumulate Brier / hit / miss / false-alarm / CSI per city |
| `earthbench/cars_verify_impact.py` | **Impact verification loop** (rain/wind/wind-typhoon/snow): model-anchored obs (valid-day 00z f006–f024 APCP sum = 24h rain, f000–f024 5-snapshot wind mean, same-file t2m mean → freeze mask) → Brier / hit-miss-false-alarm / CSI / reliability bins per variable (incl. the 17.2 m/s typhoon tier and snowfall ≥10mm events, Phase D); legacy records missing new fields degrade gracefully; absolute calibration delegated to the CRPS-side ERA5 out-of-sample audit |
| `earthbench/data/cars_cities.json` | Ops-editable city config (12 cities, per-city cold-alert gating for warm-winter cities) |
| `earthbench/eval.py::ValueEvaluator` | Cost-loss economic value scoring: V = (E_clim − E_agent)/(E_clim − E_perfect); optional exposure-weighted losses (E3 misses cost 1.5x, disclosed); `SectorValueEvaluator` adds per-sector cost ratios (health/transport/agriculture/energy, disclosed assumptions) |
| `earthbench/data/cars_t2m_deep_ngr.npz` | Frozen deep-NGR model (2000–2015 train, 2016 early-stop + GPD tail), CRPS ~0.97 K, +7.9% CRPSS vs seasonal (rolling 15y) |
| `earthbench/data/cars_climo.npz` | Monthly climatology + grid (for the +2σ relative-anomaly threshold) |
| `scripts/run_agent_comparison.py` | rule vs cars comparison + P_TRIG threshold sensitivity (expected-loss trade-off) |
| Daily CI | verify step (score past forecasts) → update step (publish new probabilities); history round-trips via gh-pages (`CARS_DATA_DIR=published_reports`); grib cache stays in-package (never deployed) |

Run the comparison:

```bash
python scripts/run_agent_comparison.py            # accuracy + V score + sensitivity
python -m earthbench --benchmark --agent cars     # full benchmark with CARS agent
python -m earthbench --benchmark --adversarial    # adversarial hard-case suite (rule baseline < 100%)
python scripts/evaluate_llm_agent.py --backend mock --suite all   # offline LLM-harness smoke
python scripts/evaluate_llm_agent.py --backend carm --suite adversarial --limit 8  # local Qwen3.6-35B via Mustard CARM
python -m earthbench.cars_serve                   # regenerate daily probabilities
python -m earthbench.cars_serve --valid 2026-09-19  # backfill (GEFS retention ~10 days)
python -m earthbench.cars_verify                  # closed-loop verification of past forecasts
python scripts/validate_historical.py             # historical backfill validation (CI gate)
```

**Adversarial suite (discriminative power)**: the 40 core cases score the rule baseline at 100% — they cannot rank agents that beat the baseline. `scenarios.py::get_adversarial_suite()` adds 16 L4 cases built on decision boundaries where linear weighted scoring systematically diverges from the national-standard lookup tables, in **both directions**: 8 false-alarm traps (sub-threshold factor mixes, AND-conditions missing one input, negated confirmations off by a hair, chronic-low temperatures without a cold-wave drop, rain misread as snow without the freeze mask, urban short-duration drought with every index red but no impact evidence) and 8 miss traps (a decisive factor diluted by normal ones — cold-dry fire weather, saturation-AND at exact thresholds, moderate drought carried by impact evidence the rule agent cannot even read, muggy-OR without duration, antecedent-saturation landslide with no rain today, gust-only typhoon over the crane-safety line, cold-wave AND both just-met, road-icing compound both just-met). The rule agent fails all 16 (0%, both FP and FN), while ground truth is still derived by the same independent `_gt_fn` standards. An agent that reasons over AND/OR structure instead of averaging can now measurably separate itself.

**Drought impact gating** (slow-onset hazards): meteorological drought ≠ actionable drought. Modern cities are buffered by water-supply infrastructure, so a short, localized SPI red flag has no decision value. `infer_drought_ground_truth` is therefore a two-channel AND: the index channel (GB/T 20481-2017 SPI/Palmer tiers, as before) sets the severity tier, and an **impact gate** must also pass — duration ≥ 60 days, or affected-area ratio ≥ 40%, or urban water-supply deficit ≥ 10% (SL 424-2008 moderate line, disclosed calibre). Index-red-but-impact-green yields no alert (0.30); missing impact observations count as unverified (宁缺毋滥). The adversarial pair `adv-drought-urban-short-no-impact` (FP trap: every index red, all impact channels below the line) vs `adv-drought-impact-gated` (miss trap: moderate SPI carried by 95-day duration + 55% extent, invisible to the index-only rule agent) encodes exactly this asymmetry.

**Exposure suite (Phase B)**: `scenarios.py::get_exposure_suite()` adds 10 action-level cases — paired cases with **identical physical observations but different `ExposureProfile`** (urban-hospital vs forest; school vs village) yield different action truths (`monitor`/`alert`/`dispatch`) via a disclosed physical-severity × exposure-class matrix, plus the invariant case (no hazard × high exposure → monitor; exposure never creates risk) and two action-tier traps where the linear-confidence severity proxy errs in both directions (rule baseline 8/10).

**Template suites (Phase C)**: `get_dispatch/upgrade/close/recover_suite()` add 17 cases closing the emergency decision chain beyond ALERT — DISPATCH (resource pre-positioning: hazard AND exposure compound), UPGRADE (level comparison + trend extrapolation), CLOSE (danger × critical-infrastructure matrix), RECOVER (conditions must **persist N hours** via `_persist_for`, making premature recovery — "lift the warning the moment rain stops" — a measurable trap). Each suite is balanced: the rule baseline (50–60%) fails one false-alarm and one miss trap per template, including the flagship `rec-flood-premature` (water was still above the line 6h ago — truth says do NOT lift, the baseline does).

**Historical backfill validation**: `scenarios.py::get_historical_validation_suite()` + `scripts/validate_historical.py` anchor the GT functions to reality — publicly documented observations from seven real catastrophes are fed into the independent lookup tables and the derived verdicts must match what actually happened: Zhengzhou 7·20 (24h rainfall 645.6 mm, station record since 1951 → GT fires at the 0.95 tier; reality: level-Ⅰ flood response, 398 dead/missing), Beijing 23·7 Haihe basin flood (hourly max 111.8 mm fed as a rigorous lower bound of any 6h window → the short-duration channel fires at 0.85 even without a clean station 24h figure; reality: red rainstorm warning + level-Ⅰ response), Typhoon Doksuri landfall at Jinjiang (50 m/s sustained, strongest Fujian landfall in 63 years → 12-grade tier 0.95; reality: 2.67 M affected, ¥14.76 B loss), the Tongliao 2021 cold front (station-archived daily minima 4.6 → −6.2 °C → derived 24h drop 10.8 °C ≥ 8 → cold-wave top tier 0.95; reality: cold-wave + record blizzard complex), the Tongliao blizzard main day (24h water-equivalent 44.6 mm summed from station hourly archive, all hours at −6.5~−3.3 °C → extreme-blizzard tier 0.95; reality: snow depth 27 → 59 cm at station), the Chongqing 2022 heat peak (station-archived max 43.1 °C → red tier 0.95; reality: Beibei 45.0 °C all-time record, first national red heat warning of 2022, Sichuan-Chongqing industrial power cuts) and the Chongqing 2022 wildfire day (FWI 70.5 computed by the repo's disclosed simplified estimator from the station-archived 14:00 snapshot — 40.9 °C / RH 29 % / 5.4 m/s / zero rain, robust to snapshot choice → extreme fire danger 0.95; reality: satellite-confirmed simultaneous fires in Fuling/Jiangjin/Banan/Bishan), plus a disclosed synthetic benign control for the False path. **8/8 verdicts match, 0 GT divergences** — every figure is source-cited in `docs/historical-validation.md`, which also records three calibre findings from the backfill itself: the cold-wave GT turned out to use the superseded 2006 three-tier structure and was **rewritten to the GB/T 20484-2017 four-level system** (top tier: multi-window drop OR — 24h ≥ 8 / 48h ≥ 10 / 72h ≥ 12 °C — AND daily min ≤ 4 °C; full regression showed zero decision flips), with the residual national-vs-local divergence disclosed; the landslide category cannot be backfilled from public sources (Zhouqu 2010 triggering-rain intensity is not publicly quantified); and the drought category is likewise blocked because SPI/Palmer indices are not publicly archived (percent-of-normal statements cannot be rigorously standardized without station climatology — the rank-bound path SPI ≤ Φ⁻¹(1/n) is documented if a sourced station record surfaces). Seven of eight hazard GT functions are now anchored to documented reality; the two blocked categories share one failure class — index/intensity inputs with no public archive.

**Benchmark hygiene**: `trivial_agents.py` provides always-alert / never-alert / deterministic-random floors so scores have a frame of reference — on the adversarial suite the rule baseline (0%, both-sided failure) sits *below* random (56%), while always-alert fails one-sided (8 FP, 0 FN): the complementary failure modes prove the suite is not class-balance artifacts. The daily report's `📐 基准体检` section (`publish_pipeline._build_benchmark_health_section`) renders the full discriminative matrix (adversarial + 4 template suites + exposure action layer + historical anchor status) alongside the trivial-baseline references, offline and network-free; CI (`tests.yml`) runs both pytest and the historical validation script with exit-code semantics.

**LLM agent evaluation harness** (`llm_agent.py` + `scripts/evaluate_llm_agent.py`): the benchmark's intended consumer is now wired in. `ScenarioPromptBuilder` turns any `ScenarioContext` into a structured prompt (per-observation table, decision protocol mandating lookup-table reasoning over averaging, strict one-line-JSON output) with two disclosed information regimes — *informed* (default: a threshold digest generated from the same `scenarios.py` constants the GT uses, so the adversarial suite becomes a pure reasoning test: same information class as the rule baseline, verdict never disclosed) and *blind* (no thresholds — tests internalized standards knowledge, a harder separate track). `parse_llm_decision` handles clean/fenced JSON and prose fallback with explicit parse-failure counting. The backend connects **directly to the local llama.cpp engine** (`127.0.0.1:8082/v1`, model `qwen3.6-35b` — the same engine behind Mustard CARM's bigmodel_proxy; CARM's chat endpoint is a *tool router* that misroutes decision prompts to search, an integration finding disclosed in the module docstring) and runs the model in non-thinking mode (`chat_template_kwargs.enable_thinking=false`: the thinking chain otherwise exhausts the token budget in `reasoning_content` before any answer appears — 15/16 truncations observed, vs clean 62-token answers with the switch). **First full-suite snapshot (83 cases, informed mode, temperature 0)**: adversarial **62% vs rule baseline 0%** (0 FP / 6 FN — the suite's design goal of separating reasoning agents from linear averaging is now empirically demonstrated with a real LLM), basic 78% (1/8 — same conservative skew), exposure physical 50% (0/5), template suites ~50% (known v1 limitation: the alert-oriented protocol doesn't explain dispatch/upgrade/close/recover semantics — future work). A deterministic offline `MockLLMBackend` keeps all 12 harness tests CI-safe; reports archive prompt + raw output per run in `docs/llm-eval-<backend>.md`.

Pipeline cadence: D 08:00 publish probabilities for D+2 → D+3 CI scores them against observations (Brier/CSI accumulate in `cars_verification_history.json`, surfaced in the daily panel and `latest.json`).

Disclosed caveats: members are daily-mean t2m perturbations with diurnal peak/valley offsets and humidity taken unperturbed from c00; wet bulb via Stull (2011), typical error <1°C; deep NGR frozen on 2000–2015 training (2016 early-stop + EVT tail); ops GEFS vs reforecast v12 version drift; coverage 100–140°E / 20–50°N (out-of-grid falls back to a 37°C rule with reduced confidence); verification v1 observes via QWeather now-temp (20:00 Beijing ≈ daily-max proxy, 1–3°C error) or GEFS f000 (model-anchored). Impact-first principle: only human-harmful extremes are reported (wet bulb ≥ 27°C / daily max ≥ 35/37°C / daily min ≤ 0°C for warm-winter cities); relative +2σ anomalies are severity context, not triggers.

### Architecture

```
Scenarios (83 test cases: 40 basic × 8 categories + 16 adversarial + 10 exposure + 17 template)
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
├── scenarios.py          # 83 cases: 40 basic + 16 adversarial + 10 exposure + 17 template suites + ScenarioStore
├── agents.py             # Rule-based Agents (8 hazard categories + MultiAlertRouter with template closure)
├── llm_agent.py          # LLM evaluation harness: prompt builder + robust parser + CARM/mock backends
├── trivial_agents.py     # Trivial baselines (always/never/random) — score floors for reference
├── benchmark.py          # AlertBenchEvaluator — full benchmark engine
├── eval.py               # BaseEvaluator + BatchEvaluator (accuracy/P/R/F1 + Brier/ECE + ValueEvaluator)
├── weather.py            # Shared meteorology (Stull wet-bulb + inverse, single source)
├── llm.py                # Shared LLM YES/NO parsing + domain prompts (single source)
├── verification.py       # Closed-loop delayed verification (FIRMS/QWeather historical)
├── calibration.py        # Threshold self-tuning from verification feedback (EWMA)
├── templates.py          # DecisionTemplate engine + context validation
├── data_collectors.py    # QWeather API + NASA FIRMS satellite fire detection (region-level memoized)
├── gefs_io.py            # GEFS grib download/cache/read single implementation + cache expiry pruning
├── enhance_data.py       # Data enhancement & enrichment pipeline
├── publish_pipeline.py   # 4-stage publish: collect → decide → report → distribute
├── integrations.py       # CARM/Mustard LLM bridge (Ollama/Qwen3)
└── __main__.py           # CLI entry point
tests/
├── conftest.py           # Hermetic env (autouse clears external API keys)
└── 25 test modules       # ~265 cases: per-hazard / per-suite / CI-safe offline
.github/workflows/
├── daily-report.yml      # CI: daily pipeline + deploy to gh-pages + Cloudflare
├── tests.yml             # CI: ruff lint + pytest (3.10/3.12 matrix, coverage) + historical validation
└── auto-post.yml         # CI: weekly summary + GitHub release (Monday)
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
| Drought | SPI (Standardized Precipitation Index) × impact gate | <= -1.5 AND (duration >= 60d or extent >= 40% or urban deficit >= 10%) | ECMWF + NDVI + drought bulletins |
| Heatwave | wet_bulb_temp | >= 27 degrees C | QWeather |
| Landslide | rainfall_1h + effective_rainfall_3d + soil_moisture + susceptibility | 1h >= 25mm (high susceptibility) or 3d >= 100mm + soil >= 0.80 | Station + terrain susceptibility layer (disclosed empirical thresholds, see `docs/expansion-plan.md`) |
| Typhoon | wind_speed + wind_gust + rainfall_24h | daily mean >= 17.2 m/s (Beaufort 8) or gust >= 24.5 m/s (Beaufort 10); wind >= 13.9 + rain >= 50mm coupled | Station + CMA (GB/T 19201-2006 tropical cyclone grades) |
| Cold Wave | temperature_min series + temperature_drop_24h | multi-window drop OR: 24h >= 8 or 48h >= 10 or 72h >= 12 degrees C, AND min <= 4 (2017 four-level top tier; drop derived from up to 4 consecutive daily minima) | CMA (GB/T 20484-2017, multi-window AND) |
| Snow | snowfall_24h + road_surface_temp | snow >= 10mm (blizzard tiers 20/30) or snow >= 2.5mm AND road temp <= 0 degrees C (icing compound); rainfall counts as snow only via freeze mask (t2m < 0.5 degrees C) | CMA + road sensors (GB/T 28592-2012 snow grades + disclosed freeze-mask proxy) |

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
