"""
Semantic Space Position Calculator (Validation-Aware)

Computes 2D positions for neurons based on their ACTUAL activation patterns
during inference, creating a dynamic manifold that shows where the current
input lives in the model's representational space.

CRITICAL: All projections must pass validation before positions can be used.
Semantic space validation is especially important as it changes per input.

Key Difference:
- neuron_position_calculator (physical): Static (one UMAP per checkpoint)
- activation_position_calculator (semantic): Dynamic (one UMAP per inference run)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle
import hashlib
import logging

try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️  UMAP not available, falling back to t-SNE")

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from .validation_framework import ValidationSuite, ValidatedProjection

logger = logging.getLogger(__name__)


class ActivationPositionCalculator:
    """
    Calculates 2D semantic positions for neurons based on their activation patterns.

    Unlike weight-space positioning (static per checkpoint), activation-space
    positioning is dynamic - it changes based on what the model is actually doing
    for a given input.
    """

    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        enable_validation: bool = True,
        validation_thresholds: Optional[Dict] = None
    ):
        self.cache_dir = cache_dir or Path(__file__).parent / "activation_position_cache"
        self.cache_dir.mkdir(exist_ok=True, parents=True)

        # Validation setup
        self.enable_validation = enable_validation
        self.validator = ValidationSuite(**(validation_thresholds or {})) if enable_validation else None
        self.last_validation_results = None

        logger.info(f"ActivationPositionCalculator initialized (validation={'enabled' if enable_validation else 'disabled'})")

    def compute_activation_positions(
        self,
        activations: List[Dict],
        method: str = 'umap',
        random_state: int = 42
    ) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """
        Compute 2D positions based on activation patterns.

        Args:
            activations: List of activation records from inference
            method: 'umap', 'tsne', or 'pca'
            random_state: Random seed for reproducibility

        Returns:
            Dict mapping (head_idx, neuron_id) -> (x, y) position
        """
        # Build activation matrix: each neuron gets a vector of its activations
        # across all iterations and character positions

        # First, collect all unique neurons and their activation patterns
        neuron_data = {}  # (head, neuron_id) -> list of (iter, char, value)

        for act in activations:
            key = (act['head'], act['neuron_id'])
            if key not in neuron_data:
                neuron_data[key] = []
            neuron_data[key].append({
                'iteration': act['iteration'],
                'char_pos': act['char_position'],
                'value': act['activation_value']
            })

        # Build feature matrix
        # Strategy: For each neuron, create feature vector of activations
        # across (iteration, char_position) grid

        # Determine grid dimensions
        max_iter = max(act['iteration'] for act in activations)
        max_char = max(act['char_position'] for act in activations)
        n_features = (max_iter + 1) * (max_char + 1)

        print(f"  Building activation matrix: {len(neuron_data)} neurons × {n_features} features")
        print(f"  Grid: {max_iter + 1} iterations × {max_char + 1} characters")

        # Build matrix
        neuron_keys = list(neuron_data.keys())
        activation_matrix = np.zeros((len(neuron_keys), n_features))

        for i, key in enumerate(neuron_keys):
            for act in neuron_data[key]:
                feature_idx = act['iteration'] * (max_char + 1) + act['char_pos']
                activation_matrix[i, feature_idx] = act['value']

        # Filter out neurons with zero activations (shouldn't happen, but safety)
        active_mask = activation_matrix.sum(axis=1) > 0
        if not active_mask.all():
            print(f"  ⚠️  Filtering {(~active_mask).sum()} neurons with zero activations")
            activation_matrix = activation_matrix[active_mask]
            neuron_keys = [k for i, k in enumerate(neuron_keys) if active_mask[i]]

        # Compute UMAP/t-SNE/PCA
        print(f"  🧮 Computing {method.upper()} for activation space...")
        positions_2d = self._reduce_dimensions(activation_matrix, method, random_state)

        # Normalize to [0, 100] range
        positions_2d = self._normalize_positions(positions_2d, scale=100.0)

        # Map back to (head, neuron_id) -> (x, y)
        position_map = {}
        for i, key in enumerate(neuron_keys):
            position_map[key] = (float(positions_2d[i, 0]), float(positions_2d[i, 1]))

        print(f"  ✓ Computed activation-space positions for {len(position_map)} neurons")

        return position_map

    def _reduce_dimensions(
        self,
        matrix: np.ndarray,
        method: str,
        random_state: int
    ) -> np.ndarray:
        """Apply dimensionality reduction"""
        N, D = matrix.shape

        if method == 'umap' and UMAP_AVAILABLE:
            # UMAP: Preserves local and global structure
            reducer = UMAP(
                n_components=2,
                n_neighbors=min(15, N - 1),  # Adjust for small datasets
                min_dist=0.1,
                metric='cosine',
                random_state=random_state,
                verbose=False  # Disable verbose for cleaner output
            )
        elif method == 'tsne':
            # t-SNE: Good for local structure
            # Pre-reduce with PCA if high-dimensional
            if D > 50:
                print(f"  ⚡ Pre-reducing with PCA: {D}D → 50D")
                pca = PCA(n_components=50, random_state=random_state)
                matrix = pca.fit_transform(matrix)
                D = 50

            reducer = TSNE(
                n_components=2,
                perplexity=min(30, N // 4),  # Adjust for small datasets
                learning_rate='auto',
                init='pca',
                random_state=random_state,
                verbose=0
            )
        elif method == 'pca':
            # PCA: Fast, linear
            reducer = PCA(n_components=2, random_state=random_state)
        else:
            raise ValueError(f"Unknown method: {method}")

        return reducer.fit_transform(matrix)

    def _normalize_positions(self, positions: np.ndarray, scale: float = 100.0) -> np.ndarray:
        """Normalize positions to [0, scale] range"""
        # Center at origin
        positions = positions - positions.mean(axis=0)

        # Scale to [-1, 1] based on max absolute value
        max_val = np.abs(positions).max()
        if max_val > 0:
            positions = positions / max_val

        # Scale to [0, scale] range
        positions = (positions + 1) * scale / 2

        return positions

    def get_activation_positions_for_inference(
        self,
        activations: List[Dict],
        method: str = 'umap',
        cache_key: Optional[str] = None
    ) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """
        Get activation-space positions, with optional caching.

        Args:
            activations: List of activation records
            method: Dimensionality reduction method
            cache_key: Optional cache key (e.g., hash of input text)

        Returns:
            Position map: (head, neuron_id) -> (x, y)
        """
        # Try cache if key provided
        if cache_key:
            cache_path = self.cache_dir / f"activation_pos_{cache_key}_{method}.pkl"
            if cache_path.exists():
                print(f"📦 Loading cached activation positions: {cache_path.name}")
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)

        # Compute positions
        position_map = self.compute_activation_positions(activations, method)

        # Cache if key provided
        if cache_key:
            cache_path = self.cache_dir / f"activation_pos_{cache_key}_{method}.pkl"
            print(f"💾 Caching activation positions: {cache_path.name}")
            with open(cache_path, 'wb') as f:
                pickle.dump(position_map, f)

        return position_map

    def compute_activation_positions_for_viz(
        self,
        activations: List[Dict],
        method: str = 'umap'
    ) -> List[Dict]:
        """
        Convenience method: recompute (x,y) positions for visualization.

        Args:
            activations: Activation list with existing positions
            method: Dimensionality reduction method

        Returns:
            Updated activation list with new (x, y) coordinates
        """
        position_map = self.compute_activation_positions(activations, method)

        # Update positions in activation records
        updated = []
        for act in activations:
            key = (act['head'], act['neuron_id'])
            if key in position_map:
                act_copy = act.copy()
                x, y = position_map[key]
                act_copy['x'] = x
                act_copy['y'] = y
                act_copy['z'] = act['iteration'] * 4 + act['head']
                updated.append(act_copy)
            else:
                updated.append(act)

        return updated

    def compute_validated_positions_for_viz(
        self,
        activations: List[Dict],
        method: str = 'umap'
    ) -> Optional[Tuple[List[Dict], ValidatedProjection]]:
        """
        Compute VALIDATED activation-space positions for visualization.

        CRITICAL: This is the method to use for research/analysis.
        Returns None if validation fails.

        Args:
            activations: Activation list from inference
            method: Dimensionality reduction method

        Returns:
            Tuple of (updated_activations, ValidatedProjection) if validation passes
            None if validation fails
        """
        if not self.enable_validation:
            logger.warning("Validation disabled - returning unvalidated positions!")
            updated = self.compute_activation_positions_for_viz(activations, method)
            return (updated, None)

        logger.info(f"\n{'='*70}")
        logger.info(f"Computing and Validating Semantic Space Projection")
        logger.info(f"{'='*70}")

        # Build activation matrix for validation
        position_map = self.compute_activation_positions(activations, method)

        # Extract unique neurons and their positions
        neuron_keys = list(position_map.keys())
        positions_2d = np.array([position_map[key] for key in neuron_keys])

        # Build activation feature matrix for validation
        # (Same matrix used for UMAP, needed for validation)
        neuron_data = {}
        for act in activations:
            key = (act['head'], act['neuron_id'])
            if key not in neuron_data:
                neuron_data[key] = []
            neuron_data[key].append({
                'iteration': act['iteration'],
                'char_pos': act['char_position'],
                'value': act['activation_value']
            })

        max_iter = max(act['iteration'] for act in activations)
        max_char = max(act['char_position'] for act in activations)
        n_features = (max_iter + 1) * (max_char + 1)

        activation_matrix = np.zeros((len(neuron_keys), n_features))
        for i, key in enumerate(neuron_keys):
            for act in neuron_data[key]:
                feature_idx = act['iteration'] * (max_char + 1) + act['char_pos']
                activation_matrix[i, feature_idx] = act['value']

        # Run validation
        logger.info("Running validation suite for semantic space...")
        validation_results = self.validator.validate_projection(
            projection_2d=positions_2d,
            high_dim_data=activation_matrix,
            neuron_ids=[f"{h}_{n}" for h, n in neuron_keys],
            projection_type="semantic"
        )

        # Store validation results
        self.last_validation_results = validation_results

        # Log validation report
        logger.info(self.validator.generate_validation_report())

        # Check if validation passed
        all_passed = all(r.passed for r in validation_results.values())

        if not all_passed:
            logger.error("❌ Semantic space validation FAILED")
            logger.error("Returning None - do NOT use these positions for analysis!")
            return None

        logger.info("✅ Semantic space validation PASSED")

        # Create ValidatedProjection
        validated_projection = ValidatedProjection(
            projection_2d=positions_2d,
            validation_results=validation_results,
            projection_type="semantic"
        )

        # Update activation positions
        updated = []
        for act in activations:
            key = (act['head'], act['neuron_id'])
            if key in position_map:
                act_copy = act.copy()
                x, y = position_map[key]
                act_copy['x'] = x
                act_copy['y'] = y
                act_copy['z'] = act['iteration'] * 4 + act['head']
                updated.append(act_copy)
            else:
                updated.append(act)

        return (updated, validated_projection)

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


def get_activation_position(
    neuron_id: int,
    head_idx: int,
    iteration: int,
    position_map: Dict[Tuple[int, int], Tuple[float, float]]
) -> Tuple[float, float, float]:
    """
    Get activation-space (x, y, z) coordinates for a neuron.

    Args:
        neuron_id: Neuron index
        head_idx: Head index
        iteration: BDH iteration
        position_map: Pre-computed activation positions

    Returns:
        (x, y, z) where x,y from activation manifold, z is temporal
    """
    key = (head_idx, neuron_id)

    if key not in position_map:
        raise ValueError(f"Neuron ({head_idx}, {neuron_id}) not in position map")

    x, y = position_map[key]
    z = iteration * 4 + head_idx  # Temporal stacking

    return float(x), float(y), float(z)


if __name__ == "__main__":
    # Test with sample activations
    print("=" * 80)
    print("Activation-Space Position Calculator Test")
    print("=" * 80)

    # Create sample activation data
    sample_activations = [
        {'iteration': 0, 'head': 0, 'neuron_id': 100, 'char_position': 0, 'activation_value': 1.5},
        {'iteration': 0, 'head': 0, 'neuron_id': 101, 'char_position': 0, 'activation_value': 1.2},
        {'iteration': 0, 'head': 0, 'neuron_id': 100, 'char_position': 1, 'activation_value': 0.8},
        {'iteration': 1, 'head': 0, 'neuron_id': 100, 'char_position': 0, 'activation_value': 2.1},
        {'iteration': 1, 'head': 1, 'neuron_id': 200, 'char_position': 0, 'activation_value': 0.9},
    ]

    calculator = ActivationPositionCalculator()
    positions = calculator.compute_activation_positions(sample_activations, method='pca')

    print(f"\n✅ Computed positions for {len(positions)} neurons")
    for key, (x, y) in positions.items():
        print(f"  Head {key[0]}, Neuron {key[1]}: ({x:.2f}, {y:.2f})")
