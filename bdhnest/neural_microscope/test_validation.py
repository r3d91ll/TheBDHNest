#!/usr/bin/env python3
"""
Test Script for Neural Microscope Validation Framework

Tests that validation framework correctly validates geometric projections
and rejects artifacts.

Usage:
    python test_validation.py --checkpoint path/to/checkpoint.pt
"""

import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add BDH root to path
BDH_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(BDH_ROOT))
sys.path.insert(0, str(BDH_ROOT / "src/utils"))

from dashboard_v2.neural_microscope import (
    NeuronPositionCalculator,
    ValidationSuite,
    ValidationResult
)


def test_physical_space_validation(checkpoint_path: str):
    """
    Test validation of physical space (weight-based) projections.

    Args:
        checkpoint_path: Path to BDH checkpoint file
    """
    logger.info("="*70)
    logger.info("TEST: Physical Space Validation")
    logger.info("="*70)

    # Create calculator with validation enabled
    calculator = NeuronPositionCalculator(enable_validation=True)

    logger.info(f"\nLoading checkpoint: {checkpoint_path}")

    try:
        # Get validated positions
        validated_positions = calculator.get_validated_positions(
            checkpoint_path=checkpoint_path,
            device='cpu',
            method='umap'
        )

        logger.info(f"\n✓ Got validated positions for {len(validated_positions)} heads")

        # Check validation status for each head
        for head_idx, validated_proj in validated_positions.items():
            logger.info(f"\nHead {head_idx}: {validated_proj}")

            if validated_proj.is_validated:
                logger.info(f"  ✅ Validation PASSED - positions are reliable")

                # Test that we can get measurements
                test_distance = validated_proj.get_distance(0, 1)
                if test_distance is not None:
                    logger.info(f"  ✓ Distance measurement works: {test_distance:.3f}")
                else:
                    logger.error(f"  ✗ Distance measurement returned None!")

            else:
                logger.warning(f"  ❌ Validation FAILED - positions are unreliable")

                # Test that measurements return None
                test_distance = validated_proj.get_distance(0, 1)
                if test_distance is None:
                    logger.info(f"  ✓ Correctly returned None for invalid projection")
                else:
                    logger.error(f"  ✗ Should have returned None but got: {test_distance}")

        # Print full validation report
        logger.info("\n" + "="*70)
        logger.info("FULL VALIDATION REPORT")
        logger.info("="*70)
        print(calculator.get_validation_report())

        return True

    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_validation_suite_directly():
    """
    Test ValidationSuite directly with synthetic data.

    This tests that the validation framework itself works correctly.
    """
    logger.info("="*70)
    logger.info("TEST: ValidationSuite with Synthetic Data")
    logger.info("="*70)

    import numpy as np

    # Create synthetic high-dim data with known structure
    n_points = 500
    n_dims = 128

    logger.info(f"\nGenerating synthetic data: {n_points} points × {n_dims} dimensions")

    # Create clustered data
    np.random.seed(42)
    cluster1 = np.random.randn(n_points // 2, n_dims) + np.array([5, 0] + [0]*(n_dims-2))
    cluster2 = np.random.randn(n_points // 2, n_dims) + np.array([-5, 0] + [0]*(n_dims-2))
    high_dim_data = np.vstack([cluster1, cluster2])

    # Create UMAP projection
    try:
        from umap import UMAP
        logger.info("Computing UMAP projection...")
        reducer = UMAP(n_components=2, random_state=42, verbose=False)
        projection_2d = reducer.fit_transform(high_dim_data)
        logger.info(f"  ✓ Projection complete: {projection_2d.shape}")
    except ImportError:
        logger.error("UMAP not available - cannot run test")
        return False

    # Run validation
    logger.info("\nRunning validation suite...")
    validator = ValidationSuite(
        min_correlation=0.5,
        min_preservation=0.7
    )

    validation_results = validator.validate_projection(
        projection_2d=projection_2d,
        high_dim_data=high_dim_data,
        neuron_ids=list(range(n_points)),
        projection_type="physical"
    )

    # Print results
    logger.info("\n" + "="*70)
    logger.info("VALIDATION RESULTS")
    logger.info("="*70)

    for test_name, result in validation_results.items():
        logger.info(f"\n{result}")

    # Summary
    passed = sum(1 for r in validation_results.values() if r.passed)
    total = len(validation_results)

    logger.info("\n" + "="*70)
    logger.info(f"Summary: {passed}/{total} tests passed")
    logger.info("="*70)

    if passed == total:
        logger.info("✅ ALL validation tests PASSED")
        return True
    else:
        logger.warning(f"❌ {total - passed} validation tests FAILED")
        return False


def main():
    parser = argparse.ArgumentParser(description='Test Neural Microscope validation framework')
    parser.add_argument('--checkpoint', type=str, help='Path to BDH checkpoint file')
    parser.add_argument('--synthetic', action='store_true', help='Test with synthetic data only')

    args = parser.parse_args()

    success = True

    # Test 1: ValidationSuite with synthetic data
    logger.info("\n\n")
    if not test_validation_suite_directly():
        success = False

    # Test 2: Physical space validation with real checkpoint
    if args.checkpoint and not args.synthetic:
        logger.info("\n\n")
        if not test_physical_space_validation(args.checkpoint):
            success = False
    elif not args.checkpoint and not args.synthetic:
        logger.warning("\nSkipping checkpoint validation (no --checkpoint provided)")
        logger.info("Run with --checkpoint path/to/checkpoint.pt to test with real data")

    # Final summary
    logger.info("\n\n")
    logger.info("="*70)
    if success:
        logger.info("✅ ALL TESTS PASSED")
        logger.info("="*70)
        logger.info("\nValidation framework is working correctly!")
        logger.info("Geometric measurements will be validated before use.")
        return 0
    else:
        logger.error("❌ SOME TESTS FAILED")
        logger.error("="*70)
        logger.error("\nValidation framework has issues - do not use for research!")
        return 1


if __name__ == '__main__':
    sys.exit(main())
