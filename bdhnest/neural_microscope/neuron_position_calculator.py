"""
Physical Space Position Calculator (Validation-Aware)

Computes 2D positions for BDH neurons based on their learned weight vectors,
using dimensionality reduction (UMAP/t-SNE) to create semantically meaningful
spatial layouts where nearby neurons have similar functions.

CRITICAL: All projections must pass validation before positions can be used.
This ensures geometric patterns represent real neural structure, not artifacts.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import pickle
import hashlib
import logging

# Try cuML (GPU-accelerated) first, fall back to sklearn (CPU)
try:
    from cuml import UMAP as CUML_UMAP
    from cuml import TSNE as CUML_TSNE
    from cuml import PCA as CUML_PCA
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

try:
    from umap import UMAP as SKLEARN_UMAP
    SKLEARN_UMAP_AVAILABLE = True
except ImportError:
    SKLEARN_UMAP_AVAILABLE = False

from sklearn.manifold import TSNE as SKLEARN_TSNE
from sklearn.decomposition import PCA as SKLEARN_PCA

from .validation_framework import ValidationSuite, ValidatedProjection

# Import dashboard config for GPU allocation
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import get_dashboard_config

logger = logging.getLogger(__name__)


class NeuronPositionCalculator:
    """
    Calculates 2D semantic positions for BDH neurons based on weight vectors.
    
    Positions are cached per checkpoint to avoid recomputation.
    """
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        enable_validation: bool = True,
        validation_thresholds: Optional[Dict] = None
    ):
        self.cache_dir = cache_dir or Path(__file__).parent / "neuron_position_cache"
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Validation setup
        self.enable_validation = enable_validation
        self.validator = ValidationSuite(**(validation_thresholds or {})) if enable_validation else None
        self.last_validation_results = None

        logger.info(f"NeuronPositionCalculator initialized (validation={'enabled' if enable_validation else 'disabled'})")
        
    def _get_checkpoint_hash(self, checkpoint_path: str) -> str:
        """Get unique hash for checkpoint file"""
        # Use file path + modification time for cache key
        ckpt_path = Path(checkpoint_path)
        mtime = ckpt_path.stat().st_mtime if ckpt_path.exists() else 0
        key = f"{checkpoint_path}_{mtime}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, checkpoint_path: str) -> Path:
        """Get cache file path for checkpoint"""
        ckpt_hash = self._get_checkpoint_hash(checkpoint_path)
        return self.cache_dir / f"neuron_positions_{ckpt_hash}.pkl"
    
    def extract_neuron_weights(self, checkpoint_path: str, device: str = 'cpu') -> Dict[str, np.ndarray]:
        """
        Extract neuron weight vectors from BDH checkpoint.
        
        For each head, we extract the encoder weight matrix (D → N projection)
        where each of the N neurons has a D-dimensional weight vector.
        
        Returns:
            Dict mapping head_idx → weight matrix (N, D) where N=8192, D=256
        """
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state_dict']
        
        # BDH architecture: shared encoder across iterations
        # Encoder stored as single tensor: (nh, D, N) where nh=4, D=256, N=8192
        encoder_key = 'encoder'

        if encoder_key not in state_dict:
            raise ValueError(f"Could not find encoder. Available keys: {list(state_dict.keys())}")

        encoder_weights = state_dict[encoder_key]  # (nh, D, N)
        nh, D, N = encoder_weights.shape

        print(f"  ✓ Found encoder weights: {nh} heads × {N} neurons × {D} dims")

        # Extract and transpose weight matrix for each head
        # We need (N, D) not (D, N) for each neuron to have a D-dimensional weight vector
        head_weights = {}
        for head_idx in range(nh):
            # Transpose to get (N, D): each neuron has D-dimensional weight
            head_weights[head_idx] = encoder_weights[head_idx].T.cpu().numpy()

        print(f"  ✓ Extracted {len(head_weights)} heads, shape per head: {head_weights[0].shape}")
        
        return head_weights
    
    def compute_semantic_positions(
        self,
        weight_matrix: np.ndarray,
        method: str = 'umap',
        n_components: int = 2,
        random_state: int = 42
    ) -> np.ndarray:
        """
        Compute 2D semantic positions using dimensionality reduction.

        Uses GPU-accelerated cuML when available (config: visualization GPU),
        falls back to CPU sklearn otherwise.

        Args:
            weight_matrix: (N, D) array where each row is a neuron's weight vector
            method: 'umap', 'tsne', or 'pca'
            n_components: Number of dimensions (2 for x,y visualization)

        Returns:
            positions: (N, 2) array of (x, y) coordinates
        """
        N, D = weight_matrix.shape

        # Get dashboard config for backend selection
        config = get_dashboard_config()
        use_gpu = config.get_umap_backend() == 'cuml' and CUML_AVAILABLE

        if use_gpu:
            viz_device = config.get_umap_device()
            print(f"  🚀 GPU-accelerated {method.upper()} on {viz_device}: {N:,} neurons ({D}D → {n_components}D)")
        else:
            print(f"  💻 CPU {method.upper()}: {N:,} neurons ({D}D → {n_components}D)")

        try:
            if method == 'umap':
                if use_gpu:
                    # cuML UMAP (GPU-accelerated on visualization GPU)
                    umap_params = config.get_umap_params()
                    reducer = CUML_UMAP(
                        n_components=n_components,
                        n_neighbors=umap_params['n_neighbors'],
                        min_dist=umap_params['min_dist'],
                        metric=umap_params['metric'],
                        random_state=random_state,
                        verbose=umap_params['verbose']
                    )
                elif SKLEARN_UMAP_AVAILABLE:
                    # sklearn UMAP (CPU)
                    reducer = SKLEARN_UMAP(
                        n_components=n_components,
                        n_neighbors=15,
                        min_dist=0.1,
                        metric='cosine',
                        random_state=random_state,
                        n_jobs=-1,
                        verbose=True
                    )
                else:
                    raise ImportError("No UMAP implementation available")

            elif method == 'tsne':
                if use_gpu:
                    # cuML t-SNE (GPU)
                    reducer = CUML_TSNE(
                        n_components=n_components,
                        perplexity=30,
                        learning_rate=200,
                        random_state=random_state,
                        verbose=1
                    )
                else:
                    # sklearn t-SNE (CPU)
                    # First reduce to 50D with PCA for speed
                    if D > 50:
                        print(f"  ⚡ Pre-reducing with PCA: {D}D → 50D (for t-SNE speed)")
                        pca = SKLEARN_PCA(n_components=50, random_state=random_state)
                        weight_matrix = pca.fit_transform(weight_matrix)
                        D = 50

                    reducer = SKLEARN_TSNE(
                        n_components=n_components,
                        perplexity=30,
                        learning_rate='auto',
                        init='pca',
                        random_state=random_state,
                        n_jobs=-1,
                        verbose=1
                    )

            elif method == 'pca':
                if use_gpu:
                    # cuML PCA (GPU)
                    reducer = CUML_PCA(n_components=n_components, random_state=random_state)
                else:
                    # sklearn PCA (CPU)
                    reducer = SKLEARN_PCA(n_components=n_components, random_state=random_state)
            else:
                raise ValueError(f"Unknown method: {method}. Use 'umap', 'tsne', or 'pca'")

            positions = reducer.fit_transform(weight_matrix)

            # Convert cuDF/cuPy to numpy if needed
            if use_gpu and hasattr(positions, 'values'):
                positions = positions.values  # cuDF DataFrame → numpy
            elif use_gpu and hasattr(positions, 'get'):
                positions = positions.get()  # cuPy array → numpy

            print(f"  ✓ Embedding complete: {positions.shape}")
            return positions

        except Exception as e:
            if use_gpu and config.should_fallback_umap_to_cpu():
                print(f"  ⚠️  GPU {method.upper()} failed: {e}")
                print(f"  → Falling back to CPU {method.upper()}...")
                # Recursively call with GPU disabled
                config._config['umap']['backend'] = 'umap-learn'
                return self.compute_semantic_positions(weight_matrix, method, n_components, random_state)
            else:
                raise
    
    def normalize_positions(self, positions: np.ndarray, scale: float = 100.0) -> np.ndarray:
        """
        Normalize positions to a reasonable range for visualization.
        
        Args:
            positions: (N, 2) array
            scale: Target scale (default 100 = 0-100 range)
            
        Returns:
            normalized: (N, 2) array scaled to [0, scale] range
        """
        # Center at origin
        positions = positions - positions.mean(axis=0)
        
        # Scale to [-1, 1] based on max absolute value
        max_val = np.abs(positions).max()
        if max_val > 0:
            positions = positions / max_val
        
        # Scale to [0, scale] range
        positions = (positions + 1) * scale / 2
        
        return positions
    
    def get_neuron_positions(
        self,
        checkpoint_path: str,
        device: str = 'cpu',
        method: str = 'umap',
        force_recompute: bool = False
    ) -> Dict[int, np.ndarray]:
        """
        Get or compute semantic positions for all neurons in all heads.
        
        Uses caching to avoid recomputing for same checkpoint.
        
        Args:
            checkpoint_path: Path to BDH checkpoint
            device: Device to load checkpoint on
            method: 'umap', 'tsne', or 'pca'
            force_recompute: Ignore cache and recompute
            
        Returns:
            Dict mapping head_idx → positions array (N, 2)
        """
        cache_path = self._get_cache_path(checkpoint_path)
        
        # Try to load from cache
        if not force_recompute and cache_path.exists():
            print(f"📦 Loading cached positions: {cache_path.name}")
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
                if cached['method'] == method:
                    print(f"  ✓ Cache hit! ({cached['num_neurons']:,} neurons × {cached['num_heads']} heads)")
                    return cached['positions']
                else:
                    print(f"  ⚠️  Cache method mismatch ({cached['method']} != {method}), recomputing...")
        
        # Compute positions
        print(f"🧠 Computing semantic neuron positions...")
        
        # Extract weight vectors
        head_weights = self.extract_neuron_weights(checkpoint_path, device)
        
        # Compute positions for each head
        head_positions = {}
        for head_idx, weights in head_weights.items():
            print(f"\n  Head {head_idx}:")
            positions = self.compute_semantic_positions(weights, method=method)
            positions = self.normalize_positions(positions, scale=100.0)
            head_positions[head_idx] = positions
        
        # Cache results
        cache_data = {
            'positions': head_positions,
            'method': method,
            'num_heads': len(head_positions),
            'num_neurons': len(head_positions[0]),
            'checkpoint_path': checkpoint_path
        }
        
        print(f"\n💾 Caching positions: {cache_path.name}")
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"  ✓ Done! Positions ready for visualization")

        return head_positions

    def get_validated_positions(
        self,
        checkpoint_path: str,
        device: str = 'cpu',
        method: str = 'umap',
        force_recompute: bool = False
    ) -> Dict[int, ValidatedProjection]:
        """
        Get VALIDATED positions for all neurons in all heads.

        CRITICAL: This is the method to use for research/analysis.
        Returns ValidatedProjection objects that only allow measurements
        if validation passes.

        Args:
            checkpoint_path: Path to BDH checkpoint
            device: Device to load checkpoint on
            method: 'umap', 'tsne', or 'pca'
            force_recompute: Ignore cache and recompute

        Returns:
            Dict mapping head_idx → ValidatedProjection object
        """
        if not self.enable_validation:
            logger.warning("Validation disabled - returning unvalidated positions!")

        # Get raw positions
        head_positions = self.get_neuron_positions(
            checkpoint_path, device, method, force_recompute
        )

        # Extract weight vectors for validation
        head_weights = self.extract_neuron_weights(checkpoint_path, device)

        # Validate each head's projection
        validated_positions = {}

        for head_idx, positions in head_positions.items():
            logger.info(f"\n{'='*70}")
            logger.info(f"Validating Physical Space Projection - Head {head_idx}")
            logger.info(f"{'='*70}")

            weight_matrix = head_weights[head_idx]  # (N, D)

            if self.enable_validation:
                # Run validation suite
                validation_results = self.validator.validate_projection(
                    projection_2d=positions,
                    high_dim_data=weight_matrix,
                    neuron_ids=list(range(len(positions))),
                    projection_type="physical"
                )

                # Store validation results
                self.last_validation_results = validation_results

                # Log validation report
                logger.info(self.validator.generate_validation_report())

                # Create ValidatedProjection
                validated_positions[head_idx] = ValidatedProjection(
                    projection_2d=positions,
                    validation_results=validation_results,
                    projection_type="physical"
                )

                # Check if validation passed
                all_passed = all(r.passed for r in validation_results.values())
                if all_passed:
                    logger.info(f"✅ Head {head_idx}: ALL validation tests PASSED")
                else:
                    logger.warning(f"❌ Head {head_idx}: Validation FAILED - positions unreliable")

            else:
                # No validation - create wrapper anyway
                logger.warning(f"⚠️  Head {head_idx}: Validation disabled - positions NOT validated!")
                validated_positions[head_idx] = ValidatedProjection(
                    projection_2d=positions,
                    validation_results={},
                    projection_type="physical"
                )

        return validated_positions

    def get_validation_report(self) -> str:
        """
        Get detailed validation report from last validation run.

        Returns:
            Formatted validation report string
        """
        if self.last_validation_results is None:
            return "No validation has been run yet."

        if not self.enable_validation:
            return "Validation is disabled."

        return self.validator.generate_validation_report()


def get_neuron_position(
    neuron_id: int,
    head_idx: int,
    iteration: int,
    positions_cache: Dict[int, np.ndarray]
) -> Tuple[float, float, float]:
    """
    Get semantic (x, y, z) coordinates for a neuron.
    
    Args:
        neuron_id: Neuron index (0-8191)
        head_idx: Head index (0-3)
        iteration: BDH iteration (0-5)
        positions_cache: Pre-computed positions from get_neuron_positions()
        
    Returns:
        (x, y, z) coordinates where:
            x, y: Semantic position from dimensionality reduction
            z: iteration * 4 + head (temporal stacking)
    """
    if head_idx not in positions_cache:
        raise ValueError(f"Head {head_idx} not in position cache")
    
    head_positions = positions_cache[head_idx]
    
    if neuron_id >= len(head_positions):
        raise ValueError(f"Neuron {neuron_id} out of range (max: {len(head_positions)-1})")
    
    x, y = head_positions[neuron_id]
    z = iteration * 4 + head_idx
    
    return float(x), float(y), float(z)


if __name__ == "__main__":
    # Test with a checkpoint
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python neuron_position_calculator.py <checkpoint_path>")
        sys.exit(1)
    
    checkpoint_path = sys.argv[1]
    
    calculator = NeuronPositionCalculator()
    positions = calculator.get_neuron_positions(
        checkpoint_path,
        method='umap',  # Try 'tsne' or 'pca' as alternatives
        force_recompute=False
    )
    
    print(f"\n✅ Success! Positions computed for {len(positions)} heads")
    for head_idx, pos in positions.items():
        print(f"  Head {head_idx}: {pos.shape[0]:,} neurons positioned in 2D")
        print(f"    X range: [{pos[:, 0].min():.2f}, {pos[:, 0].max():.2f}]")
        print(f"    Y range: [{pos[:, 1].min():.2f}, {pos[:, 1].max():.2f}]")
