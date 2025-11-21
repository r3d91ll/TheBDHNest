# TheBDHNest 🐉🪺

Production monitoring tools for Baby Dragon Hatchling (BDH) neural architecture training.

**Status**: Phase 0 - Standalone BDH monitoring system

**Roadmap**: Will evolve into TheNest - comprehensive ML training platform

---

## Features

- 🎯 **Real-time Training Dashboard** - Monitor active BDH training runs
- 📊 **Live Metrics** - Loss, accuracy, Hebbian norms, GPU utilization
- 🏠 **Experiment Gallery** - Browse and compare all experiments
- 🔬 **Neural Microscope** - Real-time inference visualization
- 🧪 **Model Testing** - Interactive inference interface
- 📈 **3D Model Visualization** - Dynamic architecture display

---

## Quick Start

### Installation

```bash
git clone https://github.com/r3d91ll/TheBDHNest
cd TheBDHNest
pip install -e .
```

### Configuration

```bash
cp config.yaml.example config.yaml
# Edit config.yaml - set experiments_root to your BDH experiments directory
```

### Launch Dashboard

```bash
bdhnest
# Or: python -m bdhnest.dashboard
```

Open browser to **http://localhost:8050**

---

## Configuration

Edit `config.yaml`:

```yaml
paths:
  experiments_root: "/path/to/BDH/experiments/"  # Where your experiments live

monitoring:
  port: 8050              # Dashboard port
  update_interval: 10     # Refresh interval (seconds)

gpu:
  monitoring_device: "cuda:1"   # GPU for monitoring
  inference_device: "cuda:0"    # GPU for inference
```

---

## Usage

### Monitor Training

1. Launch dashboard: `bdhnest`
2. Browse experiment gallery (home page)
3. Click experiment → view real-time metrics
4. Metrics auto-refresh every 10 seconds

### Test Models

1. Navigate to "Inference" page
2. Load checkpoint from experiment
3. Enter test input
4. View model predictions + Neural Microscope visualization

### Analyze Experiments

- Compare training curves across experiments
- Track Hebbian synapse evolution
- Monitor GPU utilization
- View iteration-wise statistics

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (for model inference)
- BDH experiments following standard structure:
  ```
  experiments/
  └── {experiment_name}/
      └── runs/
          └── {timestamp}/
              ├── metrics.jsonl
              ├── config.json
              └── checkpoints/
  ```

---

## Roadmap

**Phase 0** (Current): BDH-specific monitoring
- ✅ Real-time dashboard
- ✅ Neural Microscope
- ✅ Inference interface

**Phase 1** (Future): TheNest Platform
- 🔄 Multi-architecture support
- 🔄 Dataset management integration
- 🔄 Training orchestration
- 🔄 Expanded CLI (`thenest train`, `thenest dataset`, etc.)

---

## Architecture

TheBDHNest is structured for evolution into TheNest platform:

```
bdhnest/
├── models/          # Neural architectures (Phase 1: multi-arch)
├── pages/           # Dashboard pages
├── components/      # Reusable UI components
├── validation/      # Model testing & visualization
├── neural_microscope/  # Neural Microscope subsystem
└── config.py        # Configuration management
```

**Design Principle**: BDH-specific now, architecture-agnostic later.

---

## License

MIT License - see [LICENSE](LICENSE)

---

## Citation

If you use TheBDHNest in research, please cite:

```
@software{thenest2025,
  title={TheBDHNest: Production Monitoring for BDH Neural Architecture},
  author={Bucy, Todd},
  year={2025},
  url={https://github.com/r3d91ll/TheBDHNest}
}
```

**BDH Paper**: [The Dragon Hatchling (arXiv:2509.26507)](https://doi.org/10.48550/arXiv.2509.26507)

---

## Support

- 🐛 Issues: [GitHub Issues](https://github.com/r3d91ll/TheBDHNest/issues)
- 📖 Documentation: [docs/](docs/)
- 💬 Discussions: [GitHub Discussions](https://github.com/r3d91ll/TheBDHNest/discussions)
