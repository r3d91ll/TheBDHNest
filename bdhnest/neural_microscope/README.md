# Neural Microscope Module

**Location**: `src/utils/dashboard_v2/neural_microscope/`
**Purpose**: Dual-manifold visualization system for BDH neural activations (Validation-Aware)
**Version**: 1.0.0

⚠️ **CRITICAL**: This is a **research platform**, not a consumer product. All geometric projections are validated before use to ensure they represent real neural structure, not artifacts.

📖 **See Also**: `VALIDATION_INTEGRATION.md` for complete validation framework documentation

---

## Overview

The Neural Microscope provides real-time visualization of BDH model activations in two distinct geometric spaces, both grounded in the manifold hypothesis:

**All projections undergo rigorous validation** to ensure geometric patterns represent real structure.

### The Two Manifolds

**1. Physical Space (Weight Space)**
- Neurons positioned by learned 256D encoder weights
- Static per checkpoint (computed once, cached)
- Shows: "What does the model know?" (functional architecture)
- Use UMAP to project 256D weight vectors → 2D

**2. Semantic Space (Activation Space)**
- Neurons positioned by activation patterns during inference
- Dynamic per input (recomputed each inference)
- Shows: "What is the model doing?" (information flow)
- Use UMAP to project activation patterns → 2D

See `../NEURAL_MICROSCOPE_MANIFOLD_HYPOTHESIS.md` for complete theoretical foundation.

---

## Module Structure

```
neural_microscope/
├── __init__.py                          # Public API exports
├── README.md                            # This file
├── VALIDATION_INTEGRATION.md            # ⭐ Validation framework documentation
├── inference_engine.py                  # Core inference with activation capture (validation-aware)
├── neuron_position_calculator.py        # Physical space positioning (validation-aware)
├── activation_position_calculator.py    # Semantic space positioning (validation-aware)
├── validation_framework.py              # ⭐ Validation suite and ValidatedProjection
├── test_validation.py                   # ⭐ Validation framework tests
├── neural_microscope_config.py          # GPU allocation and config
├── neuron_position_cache/               # Cached physical space positions
└── activation_position_cache/           # Cached semantic space positions (optional)
```

---

## Core Components

### BDHInferenceEngine (Validation-Aware)

**Purpose**: Load BDH checkpoints and run inference with VALIDATED activation capture

**Usage**:
```python
from dashboard_v2.neural_microscope import BDHInferenceEngine

engine = BDHInferenceEngine(
    checkpoint_path="path/to/checkpoint.pt",
    device="cuda:2",  # Assigned inference GPU
    activation_threshold=0.01
)
# Output:
# Loading and validating physical space positions...
# ✅ Physical space validation PASSED for 4 heads

result = engine.run_inference("apple")
# Returns: {
#   'input_text': 'apple',
#   'output_text': 'apple',
#   'activations': [...],  # With VALIDATED physical space positions
#   'summary_stats': {...},
#   'physical_validation_status': {...},  # Validation results per head
#   'position_space': 'physical'
# }

# Switch to semantic space (VALIDATED)
semantic_result = engine.compute_activation_space_positions(
    result['activations'],
    method='umap'
)

if semantic_result is None:
    print("❌ Semantic validation FAILED")
else:
    updated_activations, validation_results = semantic_result
    print("✅ Semantic validation PASSED")
```

**Key Features**:
- ✅ **Automatic validation** of physical space on initialization
- ✅ **Fail-fast behavior** for semantic space (returns None if invalid)
- ✅ **Validation status** included in all results
- Automatic position loading (physical space from cache)
- Sparse activation capture (only records significant activations)
- GPU-aware model loading
- Activation threshold filtering

**Validation**: See `VALIDATION_INTEGRATION.md` for details

### NeuronPositionCalculator

**Purpose**: Compute physical space positions (UMAP of weight vectors)

**Usage**:
```python
from dashboard_v2.neural_microscope import NeuronPositionCalculator

calc = NeuronPositionCalculator()
positions = calc.get_neuron_positions(
    checkpoint_path="path/to/checkpoint.pt",
    method='umap',
    device='cpu'
)

# Returns: {head_idx: np.ndarray (8192, 2)}
# Cached automatically for future use
```

**Performance**:
- First computation: ~25 seconds (4 heads × 8,192 neurons)
- Cached loads: <1 second
- Cache invalidation: By checkpoint path + mtime hash

### ActivationPositionCalculator

**Purpose**: Compute semantic space positions (UMAP of activation patterns)

**Usage**:
```python
from dashboard_v2.neural_microscope import ActivationPositionCalculator

calc = ActivationPositionCalculator()
activations_updated = calc.compute_activation_positions_for_viz(
    activations=result['activations'],
    method='umap'
)

# Returns: Updated activation list with new (x, y) coordinates
```

**Performance**:
- Computation time: ~5-10 seconds per inference
- Optional caching: By input text hash (not yet implemented)
- Dynamic: Recomputed for each unique input

### GPUAllocator

**Purpose**: Manage GPU assignments for different tasks

**Usage**:
```python
from dashboard_v2.neural_microscope import GPUAllocator

allocator = GPUAllocator(config_file="path/to/config.yaml")
gpu_id = allocator.get_gpu_for_task('inference')  # Returns 2 (from config)
device = f"cuda:{gpu_id}"
```

---

## Integration with Dashboard

### Dashboard Page: `pages/neural_microscope.py`

**URL**: `/neural-microscope`

**Features**:
1. **Model Selector**: Choose checkpoint from any experiment
2. **GPU Selector**: Choose which GPU to load model on (inference task)
3. **Text Input**: Enter text to analyze
4. **Position Mode**: Toggle between Physical and Semantic space
5. **3D Visualization**: Interactive Plotly scatter plot
6. **Filters**: Top-K%, iteration, character position, activation type

**Workflow**:
```
1. User selects checkpoint
2. User chooses GPU for inference
3. User enters text ("apple")
4. Click "Run Inference"
   → Loads model on selected GPU
   → Runs inference with activation capture
   → Positions neurons in Physical Space (cached)
5. User toggles to "Semantic Space"
   → Recomputes positions based on activation patterns
   → Updates visualization
```

---

## Position Caching

### Physical Space Cache

**Location**: `neuron_position_cache/`

**Cache Key**: Hash of (checkpoint_path + file_mtime)

**Cache Files**: `neuron_positions_{hash}.pkl`

**Contents**:
```python
{
    0: np.ndarray (8192, 2),  # Head 0 positions
    1: np.ndarray (8192, 2),  # Head 1 positions
    2: np.ndarray (8192, 2),  # Head 2 positions
    3: np.ndarray (8192, 2),  # Head 3 positions
}
```

**Invalidation**: Automatic when checkpoint file modified

### Semantic Space Cache (Optional)

**Location**: `activation_position_cache/`

**Cache Key**: Hash of input text (not yet implemented)

**Rationale**: Semantic positions change per input, caching less valuable

---

## GPU Allocation

### Task Types

1. **Inference**: Model loading and forward pass
2. **Visualization**: UMAP/t-SNE computation
3. **Playback**: Animation rendering (future)

### Recommended Allocation

```yaml
# neural_microscope_config.yaml
gpus:
  inference: 2     # GPU2 for model loading
  visualization: 0  # GPU0 for UMAP (can use GPU-accelerated cuML)
  playback: 0       # GPU0 shared with visualization
```

**Benefits**:
- Inference on GPU2: Fast model loading and forward pass
- Visualization on GPU0: GPU-accelerated UMAP (6x faster than CPU)
- Playback on GPU0: 60 FPS animation with shared position cache

---

## Performance Characteristics

### Physical Space (Weight Space)

| Operation | Time | Cache |
|-----------|------|-------|
| **First Load** | ~25s | Computes UMAP, saves to cache |
| **Cached Load** | <1s | Loads from pickle file |
| **Memory** | ~260 KB | Per checkpoint |

### Semantic Space (Activation Space)

| Operation | Time | Cache |
|-----------|------|-------|
| **Computation** | ~5-10s | Per inference |
| **Memory** | ~50 MB | Transient (not cached by default) |

### GPU Acceleration

| Task | CPU | GPU (cuML) | Speedup |
|------|-----|------------|---------|
| **UMAP (8,192 neurons)** | ~6s/head | ~1s/head | 6x |
| **Inference** | ~2s | ~0.5s | 4x |
| **Playback** | 10-15 FPS | 60 FPS | 4-6x |

---

## API Reference

### BDHInferenceEngine

```python
class BDHInferenceEngine:
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda:0',
        activation_threshold: float = 0.01
    )

    def run_inference(
        self,
        text_input: str,
        max_new_tokens: int = 0,
        hebbian_learning: bool = False
    ) -> Dict

    def compute_activation_space_positions(
        self,
        activations: List[Dict],
        method: str = 'umap'
    ) -> List[Dict]
```

### NeuronPositionCalculator

```python
class NeuronPositionCalculator:
    def __init__(self, cache_dir: Optional[Path] = None)

    def get_neuron_positions(
        self,
        checkpoint_path: str,
        device: str = 'cpu',
        method: str = 'umap',
        force_recompute: bool = False
    ) -> Dict[int, np.ndarray]

    def extract_neuron_weights(
        self,
        checkpoint_path: str,
        device: str = 'cpu'
    ) -> Dict[int, np.ndarray]

    def compute_semantic_positions(
        self,
        weight_matrix: np.ndarray,
        method: str = 'umap'
    ) -> np.ndarray
```

### ActivationPositionCalculator

```python
class ActivationPositionCalculator:
    def __init__(self, cache_dir: Optional[Path] = None)

    def compute_activation_positions(
        self,
        activations: List[Dict],
        method: str = 'umap',
        random_state: int = 42
    ) -> Dict[Tuple[int, int], Tuple[float, float]]

    def compute_activation_positions_for_viz(
        self,
        activations: List[Dict],
        method: str = 'umap'
    ) -> List[Dict]
```

---

## Testing

### Unit Tests (TODO)

```bash
cd /home/todd/olympus/models/BDH
source venv/bin/activate
python -m pytest src/utils/dashboard_v2/neural_microscope/tests/
```

### Manual Testing

```python
# Test physical space positioning
from dashboard_v2.neural_microscope import BDHInferenceEngine

engine = BDHInferenceEngine(
    checkpoint_path="experiments/BDH_Philosophy_Conversational_POC_v3/runs/20251110_145854/checkpoints/checkpoint_iter_40000.pt",
    device="cuda:2"
)

result = engine.run_inference("apple")
print(f"Activations captured: {len(result['activations'])}")
print(f"Summary: {result['summary_stats']}")

# Test semantic space recomputation
from dashboard_v2.neural_microscope import ActivationPositionCalculator

calc = ActivationPositionCalculator()
semantic_activations = calc.compute_activation_positions_for_viz(
    result['activations'],
    method='umap'
)

print(f"Semantic positions computed for {len(semantic_activations)} neurons")
```

---

## Future Enhancements

### Phase 1: GPU Acceleration
- [ ] Integrate cuML for GPU-accelerated UMAP
- [ ] Implement GPUManager for task-based allocation
- [ ] Add GPU selector to dashboard UI

### Phase 2: Playback Features
- [ ] Frame-by-frame playback through iterations
- [ ] Video export (MP4/WebM)
- [ ] Side-by-side comparison of multiple runs
- [ ] GPU-accelerated rendering (60 FPS target)

### Phase 3: Advanced Visualization
- [ ] 3D semantic space (3D UMAP + time as 4th dimension)
- [ ] Cluster analysis and automatic labeling
- [ ] Neuron trajectory tracking across training
- [ ] Multi-checkpoint comparison view

---

## Related Documentation

- **`../NEURAL_MICROSCOPE_MANIFOLD_HYPOTHESIS.md`**: Complete theoretical foundation
- **`../GPU_ALLOCATION_SYSTEM_DESIGN.md`**: GPU task allocation architecture
- **`../pages/NEURAL_MICROSCOPE_DUAL_MANIFOLD_VIEW.md`**: Dual-view implementation details
- **`../../CLAUDE.md`**: BDH project overview and development guidelines

---

## Changelog

### v1.0.0 (2025-11-11)
- Initial consolidated release
- Moved from `src/utils/` to `dashboard_v2/neural_microscope/`
- Integrated physical and semantic space positioning
- Created clean module API
- Established caching infrastructure
