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
├── scenarios.py          # 20 benchmark test cases + ScenarioStore
├── agents.py             # Rule-based Agents (Fire/Flood/Drought/HeatWave + MultiAlertRouter)
├── benchmark.py          # AlertBenchEvaluator — full benchmark engine
├── eval.py               # BaseEvaluator + BatchEvaluator (accuracy/precision/recall/F1)
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
