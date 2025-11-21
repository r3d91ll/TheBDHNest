#!/usr/bin/env python3
"""TheBDHNest Quick Start Example

Demonstrates basic usage of TheBDHNest monitoring tools.
"""
from pathlib import Path
from bdhnest.config import load_config, get_experiments_root
from bdhnest.models.bdh import BDH, BDHConfig

def main():
    print("=== TheBDHNest Quick Start ===\n")

    # Load configuration
    print("1. Loading configuration...")
    try:
        config = load_config()
        experiments_root = get_experiments_root(config)
        print(f"   Experiments directory: {experiments_root}")
    except FileNotFoundError as e:
        print(f"   ⚠️  {e}")
        print("   Copy config.yaml.example to config.yaml and update experiments_root")
        return

    # List available experiments
    print("\n2. Available experiments:")
    if experiments_root.exists():
        experiments = [d for d in experiments_root.iterdir() if d.is_dir() and not d.name.startswith('.')]
        for exp in sorted(experiments)[:5]:  # Show first 5
            print(f"   - {exp.name}")
        if len(experiments) > 5:
            print(f"   ... ({len(experiments)} total)")
    else:
        print(f"   ⚠️  Directory not found: {experiments_root}")
        print("   Update config.yaml with correct experiments_root path")
        return

    # Create BDH model
    print("\n3. Creating BDH model...")
    bdh_config = BDHConfig(
        n_layer=6,
        n_embd=256,
        n_head=4,
        vocab_size=256
    )
    model = BDH(bdh_config)
    print(f"   Model: {bdh_config.n_layer} iterations × {bdh_config.n_embd}D × {bdh_config.n_head} heads")
    print(f"   Parameters: ~25M")

    print("\n4. Launch dashboard:")
    print("   Run: bdhnest")
    print("   Or:  python -m bdhnest.dashboard")
    print("   Then open: http://localhost:8050")

    print("\n✅ Quick start complete!")

if __name__ == '__main__':
    main()
