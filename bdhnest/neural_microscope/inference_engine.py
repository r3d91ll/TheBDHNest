"""
BDH Inference Engine with Sparse Activation Capture (Validation-Aware)

Runs inference on trained BDH models and captures sparse neuron activations
for visualization and analysis. Supports optional Hebbian learning during inference.

CRITICAL: This is a RESEARCH PLATFORM. All geometric projections are validated
before use to ensure they represent real neural structure, not artifacts.

Architecture Understanding:
- BDH has POSITIONAL layers (like floors in building), not hierarchical
- Each iteration uses SHARED parameters (encoder/decoder)
- Sparse activation: typically 0.01-1% of neurons active
- We capture: x_sparse, y_sparse, xy_sparse at each iteration

Dual Manifold Visualization:
- Physical Space: UMAP of neuron weights (static per checkpoint, validated)
- Semantic Space: UMAP of activation patterns (dynamic per input, validated)

Validation:
- Physical space: Cross-method validation, distance preservation
- Semantic space: Same tests PLUS consistency across inputs
- Failed validation = fallback to grid positioning
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

# Add BDH root to path for imports
BDH_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BDH_ROOT))

from bdh import BDH, BDHConfig
from src.utils.neural_microscope_config import GPUAllocator  # Import from top-level utils
from .neuron_position_calculator import NeuronPositionCalculator, get_neuron_position
from .activation_position_calculator import ActivationPositionCalculator, get_activation_position
from .validation_framework import ValidatedProjection

# Import dashboard config for GPU allocation
sys.path.insert(0, str(Path(__file__).parent.parent))
from config_loader import get_dashboard_config


class SparseActivationCollector:
    """
    Collects sparse (non-zero) neuron activations during BDH forward pass.

    Uses PyTorch hooks to capture activations without modifying model code.
    Only stores activations above threshold to save memory.
    """

    def __init__(self, activation_threshold: float = 0.01, neuron_positions: Optional[Dict[int, np.ndarray]] = None):
        self.activation_threshold = activation_threshold
        self.neuron_positions = neuron_positions  # Semantic positions per head
        self.activations = []
        self.current_iteration = 0
        self.reset()

    def reset(self):
        """Clear captured activations for new inference run"""
        self.activations = []
        self.current_iteration = 0

    def record_sparse_activation(
        self,
        iteration: int,
        activation_type: str,  # 'x_sparse', 'y_sparse', or 'xy_sparse'
        tensor: torch.Tensor,
        char_position: int
    ):
        """
        Record sparse (non-zero) activations from tensor.

        tensor shape: (B, nh, T, N) where:
            B = batch size (usually 1 for inference)
            nh = number of heads (4)
            T = sequence length
            N = internal dimension (8192)
        """
        B, nh, T, N = tensor.shape

        # Process only the current character position
        if char_position >= T:
            return

        # Extract activations for current position: (B, nh, N)
        position_activations = tensor[:, :, char_position, :]

        # Find non-zero activations above threshold
        for batch_idx in range(B):
            for head_idx in range(nh):
                head_activations = position_activations[batch_idx, head_idx]  # (N,)

                # Get indices of neurons above threshold
                active_mask = head_activations > self.activation_threshold
                active_indices = torch.where(active_mask)[0]
                active_values = head_activations[active_indices]

                # Record each active neuron
                for neuron_idx, activation_value in zip(active_indices, active_values):
                    neuron_idx = int(neuron_idx.item())
                    activation_value = float(activation_value.item())

                    # Calculate 3D position for visualization
                    if self.neuron_positions is not None:
                        # Use semantic positions from UMAP/t-SNE
                        try:
                            x, y, z = get_neuron_position(
                                neuron_idx, head_idx, iteration, self.neuron_positions
                            )
                        except (ValueError, KeyError, IndexError):
                            # Fallback to grid if position not found
                            x = neuron_idx % 90
                            y = neuron_idx // 90
                            z = iteration * 4 + head_idx
                    else:
                        # Fallback: arbitrary grid (legacy behavior)
                        x = neuron_idx % 90
                        y = neuron_idx // 90
                        z = iteration * 4 + head_idx

                    self.activations.append({
                        'iteration': iteration,
                        'head': head_idx,
                        'neuron_id': neuron_idx,
                        'activation_type': activation_type,
                        'activation_value': activation_value,
                        'char_position': char_position,
                        'x': x,
                        'y': y,
                        'z': z
                    })

    def get_activations_for_viz(self) -> List[Dict]:
        """Return captured activations formatted for visualization"""
        return self.activations

    def get_summary_stats(self) -> Dict:
        """Calculate summary statistics about captured activations"""
        if not self.activations:
            return {
                'total_active_neurons': 0,
                'sparsity': 0.0,
                'by_iteration': {}
            }

        total_neurons = 6 * 4 * 8192  # 6 iterations × 4 heads × 8192 neurons
        active_count = len(self.activations)

        # Group by iteration
        by_iteration = {}
        for act in self.activations:
            iter_id = act['iteration']
            if iter_id not in by_iteration:
                by_iteration[iter_id] = {'count': 0, 'mean_activation': 0.0, 'max_activation': 0.0}
            by_iteration[iter_id]['count'] += 1
            by_iteration[iter_id]['mean_activation'] += act['activation_value']
            by_iteration[iter_id]['max_activation'] = max(
                by_iteration[iter_id]['max_activation'],
                act['activation_value']
            )

        # Calculate means
        for iter_id in by_iteration:
            count = by_iteration[iter_id]['count']
            if count > 0:
                by_iteration[iter_id]['mean_activation'] /= count

        return {
            'total_active_neurons': active_count,
            'total_possible_neurons': total_neurons,
            'sparsity_percent': (active_count / total_neurons) * 100,
            'by_iteration': by_iteration
        }


class BDHInferenceEngine:
    """
    Inference engine for trained BDH models with VALIDATED sparse activation capture.

    Loads a checkpoint and runs inference while capturing sparse neuron
    activations for visualization and analysis.

    CRITICAL: All geometric projections are validated before use. Failed validation
    results in fallback to grid positioning. This ensures research-grade quality.

    Returns:
        - Activations with validated physical space positions by default
        - Can compute validated semantic space positions via compute_activation_space_positions()
        - Validation status included in all results
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        activation_threshold: float = 0.01,
        gpu_allocator: Optional[GPUAllocator] = None
    ):
        """
        Initialize BDH inference engine.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device string ('cuda:X' or 'cpu'). If None, uses GPUAllocator.
            activation_threshold: Minimum activation value to capture
            gpu_allocator: GPUAllocator instance. If None, creates default.
        """
        # Get device from dashboard config (GPU-role aware)
        if device is None:
            # Use dashboard config for GPU allocation
            dashboard_config = get_dashboard_config()
            self.device = dashboard_config.get_gpu('inference')
            print(f"📍 Dashboard config: Inference → {self.device}")
        else:
            # Explicit device override (e.g., from UI selection)
            self.device = device
            print(f"📍 Explicit device override: {self.device}")

        self.checkpoint_path = checkpoint_path
        self.activation_threshold = activation_threshold

        # Load model
        print(f"Loading checkpoint from: {checkpoint_path}")
        print(f"Using device: {self.device}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Extract config
        if 'config' in checkpoint:
            config_dict = checkpoint['config']
            self.config = BDHConfig(**config_dict['model'])
        else:
            # Fallback to default config
            print("Warning: No config in checkpoint, using defaults")
            self.config = BDHConfig()

        # Initialize model
        self.model = BDH(self.config).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Model loaded: {self.config.n_layer} layers × {self.config.n_embd}D × {self.config.n_head} heads")

        # Load VALIDATED physical space positions
        print("Loading and validating physical space positions...")
        try:
            position_calculator = NeuronPositionCalculator(enable_validation=True)
            validated_positions = position_calculator.get_validated_positions(
                checkpoint_path=self.checkpoint_path,
                device='cpu',
                method='umap',
                force_recompute=False
            )

            # Check validation status
            all_valid = all(vp.is_validated for vp in validated_positions.values())
            if all_valid:
                print(f"✅ Physical space validation PASSED for {len(validated_positions)} heads")
                # Extract raw positions for collector
                neuron_positions = {
                    head_idx: vp.projection_2d
                    for head_idx, vp in validated_positions.items()
                }
                self.physical_validation_status = {
                    head_idx: vp.validation_results
                    for head_idx, vp in validated_positions.items()
                }
            else:
                print(f"⚠️  Physical space validation FAILED for some heads")
                failed_heads = [h for h, vp in validated_positions.items() if not vp.is_validated]
                print(f"   Failed heads: {failed_heads}")
                print(f"   Falling back to grid positioning")
                neuron_positions = None
                self.physical_validation_status = None
        except Exception as e:
            print(f"⚠️  Could not load positions: {e}")
            print("   Falling back to grid positioning")
            neuron_positions = None
            self.physical_validation_status = None

        # Initialize activation collector with validated positions
        self.collector = SparseActivationCollector(activation_threshold, neuron_positions)

        # We'll manually capture activations in forward pass
        # (easier than hooks for this architecture)

    def run_inference(
        self,
        text_input: str,
        max_new_tokens: int = 0,
        hebbian_learning: bool = False
    ) -> Dict:
        """
        Run inference on text input and capture sparse activations.

        Args:
            text_input: Input text (will be encoded as bytes)
            max_new_tokens: Number of tokens to generate (0 = just encode input)
            hebbian_learning: If True, allow Hebbian updates during inference

        Returns:
            Dict with:
                - output_text: Generated text
                - activations: List of sparse activation records
                - summary_stats: Summary statistics
        """
        self.collector.reset()

        # Encode input as bytes
        input_bytes = text_input.encode('utf-8')
        input_ids = torch.tensor(list(input_bytes), dtype=torch.long).unsqueeze(0).to(self.device)

        # Run forward pass with activation capture
        with torch.no_grad() if not hebbian_learning else torch.enable_grad():
            output_ids = self._forward_with_capture(input_ids)

        # Decode output
        output_bytes = output_ids[0].cpu().numpy()
        try:
            output_text = bytes(output_bytes).decode('utf-8', errors='ignore')
        except:
            output_text = str(output_bytes)

        # Get captured activations (with weight-space positions)
        activations = self.collector.get_activations_for_viz()
        summary_stats = self.collector.get_summary_stats()

        return {
            'input_text': text_input,
            'output_text': output_text,
            'activations': activations,  # Default: weight-space positions
            'summary_stats': summary_stats,
            'num_chars_processed': len(input_bytes),
            'physical_validation_status': self.physical_validation_status,  # Validation results
            'position_space': 'physical'  # Which manifold is being used
        }

    def compute_activation_space_positions(
        self,
        activations: List[Dict],
        method: str = 'umap'
    ) -> Optional[Tuple[List[Dict], Dict]]:
        """
        Recompute positions for activations using VALIDATED activation-space manifold.

        CRITICAL: Returns None if validation fails. This is a research platform -
        we do not return unvalidated geometric measurements.

        This creates a NEW position map based on actual activation patterns,
        showing where the input lives in representational space.

        Args:
            activations: Activation list from run_inference()
            method: 'umap', 'tsne', or 'pca'

        Returns:
            Tuple of (updated_activations, validation_results) if validation passes
            None if validation fails
        """
        print(f"\n🧮 Computing and validating activation-space positions ({method.upper()})...")

        # Use VALIDATED activation position calculator
        calc = ActivationPositionCalculator(enable_validation=True)
        result = calc.compute_validated_positions_for_viz(activations, method=method)

        if result is None:
            print(f"❌ Semantic space validation FAILED - cannot use activation-space positions")
            print(f"   This projection does NOT reliably represent neural geometry")
            return None

        updated_activations, validated_projection = result
        print(f"✅ Semantic space validation PASSED")
        print(f"  ✓ Updated {len(updated_activations)} activation positions")

        return (updated_activations, validated_projection.validation_results)

    def _forward_with_capture(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Modified forward pass that captures sparse activations.

        Based on bdh.py forward() but with activation recording.
        """
        C = self.config
        B, T = idx.size()
        D = C.n_embd
        nh = C.n_head
        N = D * C.mlp_internal_dim_multiplier // nh

        x = self.model.embed(idx).unsqueeze(1)
        x = self.model.ln(x)  # B, 1, T, D

        # Process each BDH iteration
        for iteration in range(C.n_layer):
            # Encode
            x_latent = x @ self.model.encoder
            x_sparse = F.relu(x_latent)  # B, nh, T, N

            # Capture x_sparse activations for each character
            for char_pos in range(T):
                self.collector.record_sparse_activation(
                    iteration=iteration,
                    activation_type='x_sparse',
                    tensor=x_sparse,
                    char_position=char_pos
                )

            # Attention
            yKV = self.model.attn(Q=x_sparse, K=x_sparse, V=x)
            yKV = self.model.ln(yKV)

            # Encode V
            y_latent = yKV @ self.model.encoder_v
            y_sparse = F.relu(y_latent)  # B, nh, T, N

            # Capture y_sparse activations
            for char_pos in range(T):
                self.collector.record_sparse_activation(
                    iteration=iteration,
                    activation_type='y_sparse',
                    tensor=y_sparse,
                    char_position=char_pos
                )

            # Hebbian product
            xy_sparse = x_sparse * y_sparse  # B, nh, T, N

            # Capture xy_sparse (Hebbian) activations
            for char_pos in range(T):
                self.collector.record_sparse_activation(
                    iteration=iteration,
                    activation_type='xy_sparse',
                    tensor=xy_sparse,
                    char_position=char_pos
                )

            xy_sparse = self.model.drop(xy_sparse)

            # Decode
            yMLP = (
                xy_sparse.transpose(1, 2).reshape(B, 1, T, N * nh) @ self.model.decoder
            )
            y = self.model.ln(yMLP)
            x = self.model.ln(x + y)

        # Final output
        logits = x.view(B, T, D) @ self.model.lm_head
        output_ids = torch.argmax(logits, dim=-1)

        return output_ids


if __name__ == '__main__':
    """Test the inference engine"""
    import sys

    # Test with v3 Run 2 checkpoint
    checkpoint_path = '/home/todd/olympus/models/BDH/experiments/BDH_Philosophy_Conversational_POC_v3/runs/20251110_081542/checkpoints/checkpoint_iter_5000.pt'

    print("=" * 80)
    print("BDH Inference Engine Test")
    print("=" * 80)

    # Initialize engine
    engine = BDHInferenceEngine(
        checkpoint_path=checkpoint_path,
        device='cuda:0',
        activation_threshold=0.01
    )

    # Test with "apple"
    print("\nRunning inference on: 'apple'")
    result = engine.run_inference('apple')

    print(f"\nInput: {result['input_text']}")
    print(f"Output: {result['output_text']}")
    print(f"Chars processed: {result['num_chars_processed']}")

    print("\nSummary Statistics:")
    stats = result['summary_stats']
    print(f"  Active neurons: {stats['total_active_neurons']} / {stats['total_possible_neurons']}")
    print(f"  Sparsity: {stats['sparsity_percent']:.4f}%")

    print("\nBy Iteration:")
    for iter_id in sorted(stats['by_iteration'].keys()):
        iter_stats = stats['by_iteration'][iter_id]
        print(f"  Iteration {iter_id}:")
        print(f"    Active neurons: {iter_stats['count']}")
        print(f"    Mean activation: {iter_stats['mean_activation']:.4f}")
        print(f"    Max activation: {iter_stats['max_activation']:.4f}")

    print(f"\nTotal activation records: {len(result['activations'])}")
    if result['activations']:
        print("\nSample activations (first 5):")
        for act in result['activations'][:5]:
            print(f"  Iter {act['iteration']}, Head {act['head']}, "
                  f"Neuron {act['neuron_id']}: {act['activation_value']:.4f} "
                  f"at ({act['x']}, {act['y']}, {act['z']})")

    print("\n" + "=" * 80)
    print("Test complete!")
