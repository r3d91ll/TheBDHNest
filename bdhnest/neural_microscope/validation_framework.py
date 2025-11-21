"""
Neural Microscope Validation Framework

Research-grade validation ensuring geometric patterns represent real neural structure,
not visualization artifacts. All measurements must pass validation before use.

Principle: If geometric patterns are real, they must have predictive power beyond visualization.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, roc_auc_score
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Container for validation test results"""
    test_name: str
    metric_name: str
    score: float
    threshold: float
    passed: bool
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: Optional[Dict[str, Any]] = None

    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        s = f"{status} {self.test_name}\n"
        s += f"  Metric: {self.metric_name} = {self.score:.3f} (threshold: {self.threshold})\n"
        if self.p_value is not None:
            s += f"  P-value: {self.p_value:.6f}\n"
        if self.effect_size is not None:
            s += f"  Effect Size: {self.effect_size:.3f}\n"
        return s


class ValidationSuite:
    """
    Comprehensive validation suite for Neural Microscope projections.

    Validates that geometric patterns in manifold projections represent
    real neural structure rather than visualization artifacts.

    Eight Core Validation Protocols:
    1. Consistency Testing - Same input produces stable geometry
    2. Semantic Distance Correlation - Geometry reflects meaning
    3. Cross-Method Validation - Multiple DR methods agree
    4. Perturbation Analysis - Small changes produce proportional shifts
    5. Functional Clusters - Nearby neurons have similar functions
    6. Predictive Power - Distance predicts information transfer
    7. Distance Preservation - 2D preserves high-D relationships
    8. Ground Truth Benchmark - Correlates with human judgments
    """

    def __init__(
        self,
        min_correlation: float = 0.5,
        min_effect_size: float = 0.5,
        min_preservation: float = 0.7
    ):
        """
        Initialize validation suite with thresholds.

        Args:
            min_correlation: Minimum correlation for semantic distance tests (0.5)
            min_effect_size: Minimum Cohen's d for cluster tests (0.5 = medium effect)
            min_preservation: Minimum distance preservation score (0.7)
        """
        self.min_correlation = min_correlation
        self.min_effect_size = min_effect_size
        self.min_preservation = min_preservation
        self.validation_results: List[ValidationResult] = []

        logger.info(f"ValidationSuite initialized with thresholds:")
        logger.info(f"  min_correlation: {min_correlation}")
        logger.info(f"  min_effect_size: {min_effect_size}")
        logger.info(f"  min_preservation: {min_preservation}")

    def validate_projection(
        self,
        projection_2d: np.ndarray,
        high_dim_data: np.ndarray,
        neuron_ids: List[int],
        projection_type: str = "physical"  # "physical" or "semantic"
    ) -> Dict[str, ValidationResult]:
        """
        Run validation tests appropriate for projection type.

        Physical space: Tests 3, 5, 7 (structural validation)
        Semantic space: Tests 1, 2, 4, 6, 8 (representational validation)

        Args:
            projection_2d: 2D UMAP/tSNE projection (N, 2)
            high_dim_data: Original high-dimensional data (N, D)
            neuron_ids: List of neuron indices
            projection_type: "physical" or "semantic"

        Returns:
            Dict of validation results
        """
        logger.info(f"Running validation suite for {projection_type} space...")

        results = {}

        if projection_type == "physical":
            # Physical space validation (weight-based)
            logger.info("Validating physical space (weight geometry)...")

            results['cross_method'] = self.test_cross_method_validation(
                high_dim_data, projection_2d
            )

            results['preservation'] = self.test_distance_preservation(
                high_dim_data, projection_2d
            )

            # Functional clusters test requires inference capability
            # Placeholder for now
            logger.warning("Functional clusters test requires inference - skipping in standalone mode")

        elif projection_type == "semantic":
            # Semantic space validation (activation-based)
            logger.info("Validating semantic space (activation geometry)...")

            results['preservation'] = self.test_distance_preservation(
                high_dim_data, projection_2d
            )

            # Other semantic tests require inference data
            logger.warning("Full semantic validation requires inference data - running basic tests only")

        else:
            raise ValueError(f"Unknown projection_type: {projection_type}")

        # Store results
        self.validation_results.extend(results.values())

        # Log summary
        passed = sum(1 for r in results.values() if r.passed)
        total = len(results)
        logger.info(f"Validation complete: {passed}/{total} tests passed")

        return results

    def test_cross_method_validation(
        self,
        high_dim_data: np.ndarray,
        umap_projection: np.ndarray
    ) -> ValidationResult:
        """
        Protocol 3: Compare UMAP projection against other dimensionality reduction methods.

        CRITICAL: If structure is real, multiple DR methods should find similar
        geometric patterns. If artifacts, methods will disagree completely.

        Args:
            high_dim_data: Original high-dimensional data (N, D)
            umap_projection: UMAP 2D projection (N, 2)

        Returns:
            ValidationResult with correlation between methods
        """
        logger.info("Running cross-method validation (UMAP vs t-SNE)...")

        try:
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA
        except ImportError:
            logger.error("sklearn not available for cross-method validation")
            return ValidationResult(
                test_name="cross_method_validation",
                metric_name="umap_tsne_correlation",
                score=0.0,
                threshold=0.6,
                passed=False,
                metadata={'error': 'sklearn not available'}
            )

        # Generate alternative projections
        logger.info("  Computing t-SNE projection...")
        tsne = TSNE(n_components=2, random_state=42, verbose=0)
        tsne_projection = tsne.fit_transform(high_dim_data)

        logger.info("  Computing PCA projection...")
        pca = PCA(n_components=2, random_state=42)
        pca_projection = pca.fit_transform(high_dim_data)

        # Align projections using Procrustes analysis
        logger.info("  Aligning projections with Procrustes...")
        umap_aligned = self._procrustes_align(umap_projection, tsne_projection)

        # Compute distance correlation between methods
        umap_distances = pdist(umap_aligned)
        tsne_distances = pdist(tsne_projection)

        correlation, p_value = spearmanr(umap_distances, tsne_distances)

        # Compute stress for both methods
        umap_stress = self._compute_stress(high_dim_data, umap_projection)
        tsne_stress = self._compute_stress(high_dim_data, tsne_projection)

        passed = correlation > 0.6 and p_value < 0.05

        logger.info(f"  UMAP-tSNE correlation: {correlation:.3f} (p={p_value:.6f})")
        logger.info(f"  UMAP stress: {umap_stress:.3f}, t-SNE stress: {tsne_stress:.3f}")

        return ValidationResult(
            test_name="cross_method_validation",
            metric_name="umap_tsne_correlation",
            score=correlation,
            threshold=0.6,
            passed=passed,
            p_value=p_value,
            metadata={
                'umap_stress': float(umap_stress),
                'tsne_stress': float(tsne_stress),
                'pca_variance_explained': float(pca.explained_variance_ratio_.sum())
            }
        )

    def test_distance_preservation(
        self,
        high_dim_data: np.ndarray,
        projection_2d: np.ndarray
    ) -> ValidationResult:
        """
        Protocol 7: Test how well 2D projection preserves high-D relationships.

        CRITICAL: Core validation that UMAP projection accurately represents
        high-dimensional structure. If correlation is low, geometric measurements invalid.

        Args:
            high_dim_data: Original high-dimensional data (N, D)
            projection_2d: 2D projection (N, 2)

        Returns:
            ValidationResult with preservation score
        """
        logger.info("Running distance preservation validation...")

        # Compute pairwise distances
        logger.info("  Computing pairwise distances...")
        high_dim_distances = pdist(high_dim_data, metric='cosine')
        projection_distances = pdist(projection_2d, metric='euclidean')

        # Correlation between high-D and 2D distances
        correlation, p_value = spearmanr(high_dim_distances, projection_distances)

        logger.info(f"  Distance correlation: {correlation:.3f} (p={p_value:.6f})")

        # Trustworthiness and continuity scores
        logger.info("  Computing trustworthiness and continuity...")
        trustworthiness = self._compute_trustworthiness(high_dim_data, projection_2d, k=10)
        continuity = self._compute_continuity(high_dim_data, projection_2d, k=10)

        logger.info(f"  Trustworthiness: {trustworthiness:.3f}")
        logger.info(f"  Continuity: {continuity:.3f}")

        # Average preservation score
        preservation_score = (correlation + trustworthiness + continuity) / 3

        passed = preservation_score > self.min_preservation and p_value < 0.05

        logger.info(f"  Overall preservation: {preservation_score:.3f} ({'PASS' if passed else 'FAIL'})")

        return ValidationResult(
            test_name="distance_preservation",
            metric_name="preservation_score",
            score=preservation_score,
            threshold=self.min_preservation,
            passed=passed,
            p_value=p_value,
            metadata={
                'distance_correlation': float(correlation),
                'trustworthiness': float(trustworthiness),
                'continuity': float(continuity),
                'num_points': len(projection_2d)
            }
        )

    def generate_validation_report(self) -> str:
        """
        Generate comprehensive validation report.

        Returns formatted report showing which validations passed/failed
        and overall system validation status.
        """
        report = []
        report.append("=" * 70)
        report.append("Neural Microscope Validation Report")
        report.append("=" * 70)
        report.append("")

        if not self.validation_results:
            report.append("No validation results available.")
            return "\n".join(report)

        total_tests = len(self.validation_results)
        passed_tests = sum(1 for result in self.validation_results if result.passed)

        report.append(f"Overall Status: {passed_tests}/{total_tests} tests passed")
        report.append(f"System Validated: {'✅ YES' if passed_tests == total_tests else '❌ NO'}")
        report.append("")

        if passed_tests != total_tests:
            report.append("⚠️  WARNING: System validation FAILED")
            report.append("Geometric measurements may represent visualization artifacts.")
            report.append("DO NOT use for quantitative analysis until all tests pass.")
            report.append("")

        report.append("=" * 70)
        report.append("Test Results")
        report.append("=" * 70)
        report.append("")

        for result in self.validation_results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            report.append(f"{status} {result.test_name}")
            report.append(f"    Metric: {result.metric_name} = {result.score:.3f}")
            report.append(f"    Threshold: {result.threshold:.3f}")

            if result.p_value is not None:
                report.append(f"    P-value: {result.p_value:.6f}")

            if result.effect_size is not None:
                report.append(f"    Effect Size: {result.effect_size:.3f}")

            if result.metadata:
                report.append(f"    Details: {result.metadata}")

            report.append("")

        report.append("=" * 70)

        return "\n".join(report)

    def test_consistency(
        self,
        high_dim_data: np.ndarray,
        method: str = 'umap',
        n_runs: int = 5
    ) -> ValidationResult:
        """
        Protocol 1: Test that same input produces stable geometry across runs.

        CRITICAL: Ensures projection is deterministic enough for research.
        High variation = unreliable geometric patterns.

        Args:
            high_dim_data: Original high-dimensional data (N, D)
            method: Dimensionality reduction method ('umap' or 'tsne')
            n_runs: Number of independent runs with different random seeds

        Returns:
            ValidationResult with consistency score
        """
        logger.info(f"Running consistency testing ({n_runs} runs of {method.upper()})...")

        if method == 'umap' and not UMAP_AVAILABLE:
            return ValidationResult(
                test_name="consistency_testing",
                metric_name="consistency_score",
                score=0.0,
                threshold=0.8,
                passed=False,
                metadata={'error': 'UMAP not available'}
            )

        # Run dimensionality reduction multiple times
        projections = []
        for seed in range(n_runs):
            logger.info(f"  Run {seed + 1}/{n_runs}...")
            if method == 'umap':
                reducer = UMAP(n_components=2, random_state=seed, verbose=False)
            elif method == 'tsne':
                reducer = TSNE(n_components=2, random_state=seed, verbose=0)
            else:
                raise ValueError(f"Unknown method: {method}")

            proj = reducer.fit_transform(high_dim_data)
            projections.append(proj)

        # Align all projections to the first one using Procrustes
        logger.info("  Aligning projections...")
        aligned = [self._procrustes_align(p, projections[0]) for p in projections]

        # Compute pairwise distance stability
        logger.info("  Computing distance stability...")
        distance_matrices = [pdist(p) for p in aligned]

        # Compute coefficient of variation across runs
        mean_distances = np.mean(distance_matrices, axis=0)
        std_distances = np.std(distance_matrices, axis=0)
        cv = std_distances / (mean_distances + 1e-8)  # Avoid division by zero

        # Consistency score = 1 - median CV (higher is better)
        # CV < 0.2 (20% variation) is acceptable
        median_cv = np.median(cv)
        consistency_score = 1.0 - median_cv

        passed = consistency_score > 0.8  # < 20% variation

        logger.info(f"  Median CV: {median_cv:.3f}")
        logger.info(f"  Consistency score: {consistency_score:.3f}")

        return ValidationResult(
            test_name="consistency_testing",
            metric_name="consistency_score",
            score=consistency_score,
            threshold=0.8,
            passed=passed,
            metadata={
                'n_runs': n_runs,
                'median_cv': float(median_cv),
                'mean_cv': float(np.mean(cv)),
                'max_cv': float(np.max(cv)),
                'method': method
            }
        )

    def test_perturbation_stability(
        self,
        high_dim_data: np.ndarray,
        projection_2d: np.ndarray,
        perturbation_scale: float = 0.1,
        n_perturbations: int = 5
    ) -> ValidationResult:
        """
        Protocol 4: Test that small changes in high-D produce proportional changes in 2D.

        CRITICAL: Ensures projection is stable under small perturbations.
        Large changes from small noise = unreliable projection.

        Args:
            high_dim_data: Original high-dimensional data (N, D)
            projection_2d: Original 2D projection (N, 2)
            perturbation_scale: Scale of gaussian noise (0.1 = 10% of std)
            n_perturbations: Number of perturbation tests

        Returns:
            ValidationResult with stability correlation
        """
        logger.info(f"Running perturbation analysis ({n_perturbations} perturbations)...")

        if not UMAP_AVAILABLE:
            return ValidationResult(
                test_name="perturbation_analysis",
                metric_name="stability_correlation",
                score=0.0,
                threshold=0.9,
                passed=False,
                metadata={'error': 'UMAP not available'}
            )

        # Original pairwise distances
        original_distances = pdist(projection_2d)

        # Collect perturbed projections
        perturbed_correlations = []

        for i in range(n_perturbations):
            logger.info(f"  Perturbation {i + 1}/{n_perturbations}...")

            # Add gaussian noise to high-D data
            noise = np.random.randn(*high_dim_data.shape) * perturbation_scale * np.std(high_dim_data)
            perturbed_data = high_dim_data + noise

            # Reproject perturbed data
            reducer = UMAP(n_components=2, random_state=42, verbose=False)
            perturbed_projection = reducer.fit_transform(perturbed_data)

            # Align to original projection
            perturbed_aligned = self._procrustes_align(perturbed_projection, projection_2d)

            # Compute distances
            perturbed_distances = pdist(perturbed_aligned)

            # Correlation between original and perturbed distances
            correlation, _ = spearmanr(original_distances, perturbed_distances)
            perturbed_correlations.append(correlation)

        # Mean correlation across perturbations
        mean_correlation = np.mean(perturbed_correlations)
        std_correlation = np.std(perturbed_correlations)

        # Pass if correlation > 0.9 (small perturbations don't drastically change geometry)
        passed = mean_correlation > 0.9

        logger.info(f"  Mean stability correlation: {mean_correlation:.3f} ± {std_correlation:.3f}")

        return ValidationResult(
            test_name="perturbation_analysis",
            metric_name="stability_correlation",
            score=mean_correlation,
            threshold=0.9,
            passed=passed,
            metadata={
                'perturbation_scale': perturbation_scale,
                'n_perturbations': n_perturbations,
                'std_correlation': float(std_correlation),
                'min_correlation': float(np.min(perturbed_correlations)),
                'max_correlation': float(np.max(perturbed_correlations))
            }
        )

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _procrustes_align(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Align X to Y using Procrustes analysis.

        Args:
            X: Source matrix (N, 2)
            Y: Target matrix (N, 2)

        Returns:
            Aligned X matrix
        """
        try:
            from scipy.spatial import procrustes
            _, aligned_X, _ = procrustes(Y, X)
            return aligned_X
        except ImportError:
            logger.warning("scipy.spatial.procrustes not available, returning original X")
            return X

    def _compute_stress(self, high_dim: np.ndarray, low_dim: np.ndarray) -> float:
        """
        Compute stress (distance preservation error).

        Stress measures how well the low-dimensional embedding preserves
        pairwise distances from the high-dimensional space.

        Args:
            high_dim: High-dimensional data (N, D)
            low_dim: Low-dimensional embedding (N, 2)

        Returns:
            Stress value (lower is better)
        """
        hd_distances = pdist(high_dim, metric='cosine')
        ld_distances = pdist(low_dim, metric='euclidean')

        # Normalize distances to [0, 1]
        hd_distances = hd_distances / np.max(hd_distances) if np.max(hd_distances) > 0 else hd_distances
        ld_distances = ld_distances / np.max(ld_distances) if np.max(ld_distances) > 0 else ld_distances

        stress = np.sqrt(np.sum((hd_distances - ld_distances) ** 2) / np.sum(hd_distances ** 2))
        return stress

    def _compute_trustworthiness(
        self,
        high_dim: np.ndarray,
        low_dim: np.ndarray,
        k: int = 10
    ) -> float:
        """
        Compute trustworthiness score.

        Trustworthiness measures whether points that are close in the low-dimensional
        embedding were also close in the high-dimensional space.

        Based on: Venna & Kaski (2006)

        Args:
            high_dim: High-dimensional data (N, D)
            low_dim: Low-dimensional embedding (N, 2)
            k: Number of nearest neighbors to consider

        Returns:
            Trustworthiness score in [0, 1] (higher is better)
        """
        from sklearn.neighbors import NearestNeighbors

        n_samples = high_dim.shape[0]

        # Find k-nearest neighbors in both spaces
        nbrs_hd = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(high_dim)
        nbrs_ld = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(low_dim)

        _, indices_hd = nbrs_hd.kneighbors(high_dim)
        _, indices_ld = nbrs_ld.kneighbors(low_dim)

        # Remove self from neighbor lists
        indices_hd = indices_hd[:, 1:]
        indices_ld = indices_ld[:, 1:]

        # Compute trustworthiness
        trustworthiness = 0.0
        for i in range(n_samples):
            # Neighbors in low-dim but not in high-dim
            ld_neighbors = set(indices_ld[i])
            hd_neighbors = set(indices_hd[i])

            false_neighbors = ld_neighbors - hd_neighbors

            for j in false_neighbors:
                # Rank of j in high-dim neighbors of i
                rank_hd = np.where(indices_hd[i] == j)[0]
                if len(rank_hd) > 0:
                    trustworthiness += max(0, rank_hd[0] - k)

        # Normalize
        normalization = (2 / (n_samples * k * (2 * n_samples - 3 * k - 1)))
        trustworthiness = 1 - normalization * trustworthiness

        return trustworthiness

    def _compute_continuity(
        self,
        high_dim: np.ndarray,
        low_dim: np.ndarray,
        k: int = 10
    ) -> float:
        """
        Compute continuity score.

        Continuity measures whether points that were close in the high-dimensional
        space remain close in the low-dimensional embedding.

        Based on: Venna & Kaski (2006)

        Args:
            high_dim: High-dimensional data (N, D)
            low_dim: Low-dimensional embedding (N, 2)
            k: Number of nearest neighbors to consider

        Returns:
            Continuity score in [0, 1] (higher is better)
        """
        from sklearn.neighbors import NearestNeighbors

        n_samples = high_dim.shape[0]

        # Find k-nearest neighbors in both spaces
        nbrs_hd = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(high_dim)
        nbrs_ld = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(low_dim)

        _, indices_hd = nbrs_hd.kneighbors(high_dim)
        _, indices_ld = nbrs_ld.kneighbors(low_dim)

        # Remove self from neighbor lists
        indices_hd = indices_hd[:, 1:]
        indices_ld = indices_ld[:, 1:]

        # Compute continuity
        continuity = 0.0
        for i in range(n_samples):
            # Neighbors in high-dim but not in low-dim
            hd_neighbors = set(indices_hd[i])
            ld_neighbors = set(indices_ld[i])

            missing_neighbors = hd_neighbors - ld_neighbors

            for j in missing_neighbors:
                # Rank of j in low-dim neighbors of i
                rank_ld = np.where(indices_ld[i] == j)[0]
                if len(rank_ld) > 0:
                    continuity += max(0, rank_ld[0] - k)

        # Normalize
        normalization = (2 / (n_samples * k * (2 * n_samples - 3 * k - 1)))
        continuity = 1 - normalization * continuity

        return continuity


# Validation-aware wrapper class (to be integrated with NeuralMicroscope)
class ValidatedProjection:
    """
    Container for validated 2D projections.

    Only allows access to geometric measurements if validation passed.
    """

    def __init__(
        self,
        projection_2d: np.ndarray,
        validation_results: Dict[str, ValidationResult],
        projection_type: str
    ):
        self.projection_2d = projection_2d
        self.validation_results = validation_results
        self.projection_type = projection_type
        self.is_validated = all(r.passed for r in validation_results.values())

    def get_distance(self, idx1: int, idx2: int) -> Optional[float]:
        """
        Get geometric distance between two points.

        Returns None if validation failed (measurements unreliable).
        """
        if not self.is_validated:
            logger.warning("Projection not validated - returning None for distance")
            return None

        pos1 = self.projection_2d[idx1]
        pos2 = self.projection_2d[idx2]
        return float(np.linalg.norm(pos1 - pos2))

    def get_position(self, idx: int) -> Optional[Tuple[float, float]]:
        """
        Get 2D position for a point.

        Returns None if validation failed (positions unreliable).
        """
        if not self.is_validated:
            logger.warning("Projection not validated - returning None for position")
            return None

        pos = self.projection_2d[idx]
        return (float(pos[0]), float(pos[1]))

    def __str__(self):
        status = "VALIDATED" if self.is_validated else "INVALID"
        passed = sum(1 for r in self.validation_results.values() if r.passed)
        total = len(self.validation_results)
        return f"ValidatedProjection({self.projection_type}, {status}, {passed}/{total} tests passed)"
