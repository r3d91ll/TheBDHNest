"""
Home Page - Experiment Gallery

Discover and browse all BDH experiments.
"""

import dash
from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
from pathlib import Path
import json
from datetime import datetime
import subprocess
import re

from ..config import get_experiments_root

dash.register_page(__name__, path='/', name='Experiment Gallery', order=1)

# Find BDH experiments directory from config
EXPERIMENTS_DIR = get_experiments_root()


def get_running_experiments():
    """
    Detect which experiments are currently running and on which GPU.

    Uses two methods:
    1. Check for recently updated metrics.jsonl files (< 2 minutes old)
    2. Read GPU from the metrics file itself (more reliable than config.json)

    Returns:
        dict: {experiment_name: {'running': True, 'gpu': 'cuda:0', 'pid': None}}
    """
    running_exps = {}

    try:
        # Scan all experiments for recent metrics updates
        for exp_dir in EXPERIMENTS_DIR.iterdir():
            if not exp_dir.is_dir():
                continue
            if exp_dir.name.startswith('.') or exp_dir.name.startswith('_'):
                continue

            runs_dir = exp_dir / "runs"
            if not runs_dir.exists():
                continue

            # Check all runs for recent activity
            for run_dir in runs_dir.iterdir():
                if not run_dir.is_dir() or run_dir.name.startswith('.'):
                    continue

                # Check for any metrics file (single or multi-phase)
                metrics_files = [
                    run_dir / "metrics.jsonl",
                    run_dir / "phase1_metrics.jsonl",
                    run_dir / "phase2_metrics.jsonl",
                ]

                # Find the most recently updated metrics file
                import time
                most_recent_file = None
                most_recent_mtime = 0

                for mf in metrics_files:
                    if mf.exists():
                        mtime = mf.stat().st_mtime
                        if mtime > most_recent_mtime:
                            most_recent_mtime = mtime
                            most_recent_file = mf

                if not most_recent_file:
                    continue

                # Check if metrics file was updated in last 2 minutes
                age_seconds = time.time() - most_recent_mtime

                if age_seconds < 120:  # 2 minutes
                    # This experiment is running!
                    # Read GPU from latest metrics entry
                    gpu = 'Unknown'
                    try:
                        with open(most_recent_file, 'r') as f:
                            lines = f.readlines()
                            if lines:
                                # Parse last line to get GPU info
                                last_entry = json.loads(lines[-1])

                                # Auto-detect GPU from metrics (more reliable than config)
                                for gpu_idx in [0, 1, 2]:
                                    util_key = f'gpu{gpu_idx}_util'
                                    mem_key = f'gpu{gpu_idx}_mem_used'
                                    if util_key in last_entry and last_entry[util_key] and last_entry[util_key] > 50:
                                        gpu = f'cuda:{gpu_idx}'
                                        break
                                    elif mem_key in last_entry and last_entry[mem_key] and last_entry[mem_key] > 10000:
                                        gpu = f'cuda:{gpu_idx}'
                                        break
                    except:
                        pass

                    running_exps[exp_dir.name] = {
                        'running': True,
                        'gpu': gpu,
                        'pid': None,  # Not tracking PID with this method
                        'run': run_dir.name
                    }
                    break  # Only mark experiment once

    except Exception as e:
        # If detection fails, just return empty dict
        pass

    return running_exps


def discover_experiments():
    """
    Discover all experiments with runs.

    Returns:
        List of dicts with experiment metadata
    """
    experiments = []

    if not EXPERIMENTS_DIR.exists():
        return experiments

    # Get running experiment status
    running_exps = get_running_experiments()

    for exp_dir in EXPERIMENTS_DIR.iterdir():
        if not exp_dir.is_dir():
            continue
        if exp_dir.name.startswith('.') or exp_dir.name.startswith('_'):
            continue

        # Try to load README first (experiments with README but no runs are still valid)
        readme_file = exp_dir / "README.md"
        description = ""
        if readme_file.exists():
            try:
                lines = readme_file.read_text().split('\n')
                # Get first non-empty, non-header line
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        description = line[:200]  # First 200 chars
                        break
            except:
                pass

        # Check for runs directory
        runs_dir = exp_dir / "runs"
        if not runs_dir.exists():
            # Skip experiments with no runs directory AND no README
            if not readme_file.exists():
                continue
            # Has README but no runs directory - show it anyway
            runs = []
            latest_run = None
            config = {}
        else:
            # Count runs
            runs = [r for r in runs_dir.iterdir() if r.is_dir() and not r.name.startswith('.')]

            # Get latest run (if any)
            if runs:
                latest_run = max(runs, key=lambda r: r.stat().st_mtime)

                # Try to load config
                config_file = latest_run / "config.json"
                config = {}
                if config_file.exists():
                    try:
                        with open(config_file) as f:
                            config = json.load(f)
                    except:
                        pass
            else:
                # Has runs directory but no runs yet - show if README exists
                if not readme_file.exists():
                    continue
                latest_run = None
                config = {}

        # Check if this experiment is running
        running_info = running_exps.get(exp_dir.name, {})

        experiments.append({
            'name': exp_dir.name,
            'path': str(exp_dir),
            'runs_count': len(runs),
            'latest_run': latest_run.name if latest_run else 'No runs yet',
            'last_modified': datetime.fromtimestamp(latest_run.stat().st_mtime) if latest_run else datetime.fromtimestamp(exp_dir.stat().st_mtime),
            'config': config,
            'description': description,
            'running': running_info.get('running', False),
            'gpu': running_info.get('gpu'),
            'pid': running_info.get('pid')
        })

    # Sort by most recent
    experiments.sort(key=lambda e: e['last_modified'], reverse=True)

    return experiments


def create_experiment_card(exp):
    """Create a card for an experiment"""

    # Get model info from config
    model_info = "Unknown architecture"
    if 'model' in exp['config']:
        m = exp['config']['model']
        n_layer = m.get('n_layer', '?')
        n_embd = m.get('n_embd', '?')
        n_head = m.get('n_head', '?')
        model_info = f"{n_layer} iterations × {n_embd}D × {n_head} heads"

    # Status badges
    status_badges = []
    if exp.get('running', False):
        # Running badge
        status_badges.append(
            dbc.Badge([
                html.I(className="bi bi-play-fill me-1"),
                "Running"
            ], color="success", className="me-2")
        )

        # GPU badge
        gpu = exp.get('gpu', 'Unknown')
        gpu_display = gpu.replace('cuda:', 'GPU').upper() if 'cuda' in gpu else gpu
        status_badges.append(
            dbc.Badge([
                html.I(className="bi bi-gpu-card me-1"),
                gpu_display
            ], color="primary", className="me-2")
        )
    else:
        status_badges.append(
            dbc.Badge([
                html.I(className="bi bi-stop-fill me-1"),
                "Stopped"
            ], color="secondary", className="me-2")
        )

    return dbc.Card([
        dbc.CardHeader([
            html.Div([
                html.H5(exp['name'], className="mb-0 d-inline-block me-3"),
                html.Div(status_badges, className="d-inline-block")
            ])
        ]),
        dbc.CardBody([
            html.P(exp['description'] or "No description available", className="text-muted small mb-2"),
            html.Hr(),
            html.Div([
                html.I(className="bi bi-cpu me-2"),
                html.Span(model_info, className="small"),
            ], className="mb-2"),
            html.Div([
                html.I(className="bi bi-folder me-2"),
                html.Span(f"{exp['runs_count']} run(s)", className="small me-3"),
                html.I(className="bi bi-clock me-2"),
                html.Span(exp['last_modified'].strftime("%Y-%m-%d %H:%M"), className="small"),
            ]),
        ]),
        dbc.CardFooter([
            dbc.Button(
                [html.I(className="bi bi-eye me-2"), "View Experiment"],
                href=f"/experiment?path={exp['path']}",
                color="primary",
                size="sm"
            )
        ])
    ], className="mb-3 shadow-sm")


# Layout
layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.H1([
                html.I(className="bi bi-grid-3x3-gap me-3"),
                "Experiment Gallery"
            ]),
            html.P("Browse all BDH training experiments", className="lead text-muted"),
            html.Hr(),
        ])
    ]),

    dbc.Row([
        dbc.Col([
            html.Div(id='experiment-gallery')
        ])
    ]),

    # Manual refresh only (removed auto-refresh per user request)
    # Graphs in experiment detail page auto-update every 10 seconds
    # User can manually refresh home page if needed
    dcc.Interval(id='gallery-refresh', interval=999999999, n_intervals=0)  # Effectively disabled
])


@callback(
    Output('experiment-gallery', 'children'),
    Input('gallery-refresh', 'n_intervals')
)
def update_gallery(n):
    """Update experiment gallery"""
    experiments = discover_experiments()

    if not experiments:
        return dbc.Alert([
            html.H4("No experiments found", className="alert-heading"),
            html.P(f"No experiment runs found in: {EXPERIMENTS_DIR}"),
            html.Hr(),
            html.P("Experiments should be in: experiments/EXPERIMENT_NAME/runs/TIMESTAMP/", className="mb-0")
        ], color="info")

    # Create cards in rows of 3
    cards = []
    for i in range(0, len(experiments), 3):
        row_exps = experiments[i:i+3]
        cards.append(
            dbc.Row([
                dbc.Col(create_experiment_card(exp), md=4)
                for exp in row_exps
            ], className="mb-3")
        )

    return html.Div([
        dbc.Alert([
            html.I(className="bi bi-info-circle me-2"),
            f"Found {len(experiments)} experiment(s)"
        ], color="light", className="mb-3"),
        html.Div(cards)
    ])
