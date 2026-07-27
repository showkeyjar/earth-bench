# 🌍 EarthBench — Earth Decision Intelligence Benchmark

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
- **Multi-source data fusion**: Satellite (MODIS), meteorological stations, hydrological sensors
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

## Contact

- Email: zergskj@163.com
- GitHub: https://github.com/showkeyjar/earth-bench
- Website: https://earth-ai.fun

## License

CC BY 4.0
