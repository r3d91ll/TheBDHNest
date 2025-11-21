#!/usr/bin/env python3
"""
TheBDHNest Dashboard - BDH Training Monitoring

Phase 0: BDH-specific monitoring
Phase 1: Will become TheNest platform dashboard

Built from scratch with correct BDH terminology and concepts.
No Transformer assumptions.

Key principles:
- "Iterations" not "Layers" (shared parameters applied repeatedly)
- "Block size" = training batch size, NOT context window
- Hebbian learning = synaptic strengthening, not just gradient descent
- Unlimited sequence length via recurrent state

Usage:
    python -m bdhnest.dashboard [--port PORT]
    Or: bdhnest

Default port: 8050
"""

import argparse
from pathlib import Path

import dash
from dash import dcc, html
import dash_bootstrap_components as dbc

from .config import load_config, get_monitoring_settings

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP],
    suppress_callback_exceptions=True,
    use_pages=True  # Enable multi-page support
)

app.title = "TheBDHNest - BDH Training Dashboard"

# Navigation bar
navbar = dbc.Navbar(
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.A(
                    dbc.Row([
                        dbc.Col(html.I(className="bi bi-brain me-2", style={'fontSize': '1.5rem'})),
                        dbc.Col(dbc.NavbarBrand("TheBDHNest", className="ms-2")),
                    ], align="center", className="g-0"),
                    href="/",
                    style={"textDecoration": "none"},
                )
            ], width="auto"),
            dbc.Col([
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink([
                        html.I(className="bi bi-grid-3x3-gap me-1"),
                        "Experiments"
                    ], href="/", active="exact")),
                    dbc.NavItem(dbc.NavLink([
                        html.I(className="bi bi-gear me-1"),
                        "Settings"
                    ], href="/settings", active="exact")),
                ], navbar=True, className="me-auto"),
            ], className="flex-grow-1"),
            dbc.Col([
                html.Div([
                    html.Span("Baby Dragon Hatchling", className="text-muted small me-3"),
                    html.Span("v0.1.0 - Phase 0", className="badge bg-success")
                ])
            ], className="ms-auto text-end", width="auto"),
        ], className="w-100 align-items-center"),
    ], fluid=True),
    color="dark",
    dark=True,
    className="mb-3",
)

# Layout
app.layout = html.Div([
    navbar,
    dbc.Container([
        dash.page_container  # Pages will be rendered here
    ], fluid=True)
])


def main():
    """Entry point for dashboard"""
    parser = argparse.ArgumentParser(description="TheBDHNest - BDH Training Dashboard")
    parser.add_argument('--port', type=int, help='Port to run dashboard on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    args = parser.parse_args()

    # Load config for port
    try:
        config = load_config()
        monitoring = get_monitoring_settings(config)
        port = args.port if args.port else monitoring.get('port', 8050)
        debug = monitoring.get('debug', False)
    except FileNotFoundError as e:
        print(f"⚠️  {e}")
        print("Using defaults: port=8050, debug=False")
        port = args.port if args.port else 8050
        debug = False

    print("=" * 70)
    print("TheBDHNest - BDH Training Dashboard v0.1.0")
    print("=" * 70)
    print(f"\nStarting dashboard on http://localhost:{port}")
    print("\nKey Features:")
    print("  ✓ Correct BDH terminology (iterations, not layers)")
    print("  ✓ Hebbian learning metrics")
    print("  ✓ Real-time experiment monitoring")
    print("  ✓ Neural Microscope visualization")
    print("\nPress Ctrl+C to stop")
    print("=" * 70)
    print()

    app.run(
        host=args.host,
        port=port,
        debug=debug
    )


if __name__ == '__main__':
    main()
