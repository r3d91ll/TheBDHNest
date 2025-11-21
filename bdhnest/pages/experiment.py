"""
Experiment Detail Page

View training metrics and model details for a specific experiment.

BDH-Specific Metrics:
- Loss & Accuracy (standard)
- Encoder/Decoder Norms (Hebbian synaptic strength)
- Gradient Norm (training dynamics)
- GPU Utilization (system resources)
"""

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from pathlib import Path
import json
import sys
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add BDH root to path for model imports
BDH_ROOT = Path(__file__).parent.parent.parent.parent.parent
EXPERIMENTS_DIR = BDH_ROOT / "experiments"
sys.path.insert(0, str(BDH_ROOT))

# Import components
sys.path.insert(0, str(Path(__file__).parent.parent))
from components.inference import create_inference_interface
from components.neural_microscope import create_neural_microscope_interface

# Import Fisher visualization utilities
sys.path.insert(0, str(BDH_ROOT / "src"))
from utils.fisher_visualization import generate_fisher_visualization, load_fisher_matrix

try:
    from bdh import BDH, BDHConfig
    BDH_AVAILABLE = True
except ImportError:
    BDH_AVAILABLE = False

dash.register_page(__name__, path='/experiment')


def load_metrics(run_path):
    """Load metrics from JSONL file(s) - handles both single and multi-phase experiments"""
    run_path = Path(run_path)

    # Try different metrics file patterns
    metrics_files = [
        run_path / "metrics.jsonl",           # Single-phase experiments
        run_path / "phase1_metrics.jsonl",    # Multi-phase: Phase 1
        run_path / "phase2_metrics.jsonl",    # Multi-phase: Phase 2
    ]

    metrics = []
    for metrics_file in metrics_files:
        if metrics_file.exists():
            with open(metrics_file) as f:
                for line in f:
                    try:
                        metrics.append(json.loads(line))
                    except:
                        continue

    # Sort by iteration to ensure proper ordering when combining phases
    if metrics:
        metrics.sort(key=lambda x: x.get('iteration', 0))

    return metrics


def load_config(run_path):
    """Load experiment config"""
    config_file = Path(run_path) / "config.json"
    if not config_file.exists():
        return {}

    try:
        with open(config_file) as f:
            return json.load(f)
    except:
        return {}


def load_readme(exp_path):
    """Load experiment README"""
    readme_file = Path(exp_path) / "README.md"
    if not readme_file.exists():
        return "No README.md found"

    try:
        return readme_file.read_text()
    except:
        return "Error loading README.md"


def create_loss_chart(metrics):
    """Training loss curves - separate lines per dataset"""
    if not metrics:
        return go.Figure()

    # Split metrics by dataset (for curriculum learning)
    tinystories_metrics = [m for m in metrics if 'dataset' in m and 'TinyStories' in m['dataset']]
    instruction_metrics = [m for m in metrics if 'dataset' in m and 'Instruction' in m['dataset']]
    other_metrics = [m for m in metrics if 'dataset' not in m or ('TinyStories' not in m['dataset'] and 'Instruction' not in m['dataset'])]

    fig = go.Figure()

    # If we have dataset-specific metrics, show separate lines
    if tinystories_metrics or instruction_metrics:
        if tinystories_metrics:
            fig.add_trace(go.Scatter(
                x=[m['iteration'] for m in tinystories_metrics],
                y=[m.get('loss', m.get('train_loss', 0)) for m in tinystories_metrics],
                mode='lines+markers',
                name='TinyStories',
                line=dict(color='#3498db', width=2),
                marker=dict(size=4)
            ))

        if instruction_metrics:
            fig.add_trace(go.Scatter(
                x=[m['iteration'] for m in instruction_metrics],
                y=[m.get('loss', m.get('train_loss', 0)) for m in instruction_metrics],
                mode='lines+markers',
                name='Instructions',
                line=dict(color='#e74c3c', width=2),
                marker=dict(size=4)
            ))
    else:
        # Fallback: single line for experiments without dataset field
        iterations = [m['iteration'] for m in metrics]
        train_loss = [m.get('loss', m.get('train_loss', m.get('loss_task', m.get('loss_total', 0)))) for m in metrics]
        fig.add_trace(go.Scatter(
            x=iterations,
            y=train_loss,
            mode='lines',
            name='Training Loss',
            line=dict(color='#3498db', width=2)
        ))

    fig.update_layout(
        title="Training Loss (per dataset)",
        xaxis_title="Iteration",
        yaxis_title="Loss",
        template="plotly_white",
        hovermode='x unified',
        legend=dict(x=0.8, y=0.95)
    )

    return fig


def create_accuracy_chart(metrics):
    """Training accuracy over time - separate lines per dataset"""
    if not metrics:
        return go.Figure()

    # Split metrics by dataset (for curriculum learning)
    tinystories_metrics = [m for m in metrics if 'dataset' in m and 'TinyStories' in m['dataset']]
    instruction_metrics = [m for m in metrics if 'dataset' in m and 'Instruction' in m['dataset']]

    fig = go.Figure()

    # If we have dataset-specific metrics, show separate lines
    if tinystories_metrics or instruction_metrics:
        if tinystories_metrics:
            fig.add_trace(go.Scatter(
                x=[m['iteration'] for m in tinystories_metrics],
                y=[m.get('accuracy', m.get('train_accuracy', m.get('train_acc', 0))) for m in tinystories_metrics],
                mode='lines+markers',
                name='TinyStories',
                line=dict(color='#3498db', width=2),
                marker=dict(size=4)
            ))

        if instruction_metrics:
            fig.add_trace(go.Scatter(
                x=[m['iteration'] for m in instruction_metrics],
                y=[m.get('accuracy', m.get('train_accuracy', m.get('train_acc', 0))) for m in instruction_metrics],
                mode='lines+markers',
                name='Instructions',
                line=dict(color='#e74c3c', width=2),
                marker=dict(size=4)
            ))
    else:
        # Fallback: single line for experiments without dataset field
        iterations = [m['iteration'] for m in metrics]
        accuracy = [m.get('accuracy', m.get('train_accuracy', m.get('train_acc', 0))) for m in metrics]
        fig.add_trace(go.Scatter(
            x=iterations,
            y=accuracy,
            mode='lines',
            name='Training Accuracy',
            line=dict(color='#2ecc71', width=2),
            fill='tozeroy',
            fillcolor='rgba(46, 204, 113, 0.1)'
        ))

    fig.update_layout(
        title="Training Accuracy (per dataset)",
        xaxis_title="Iteration",
        yaxis_title="Accuracy (%)",
        template="plotly_white",
        hovermode='x unified',
        legend=dict(x=0.8, y=0.95)
    )

    return fig


def create_hebbian_chart(metrics):
    """
    Real Hebbian Learning Dynamics

    BDH has ONE shared data structure, iterated through 16 times (like paint layers).

    Tracks activation-based Hebbian learning:
    - Activation Rate: % of neurons active in the shared structure (HIGHER is better, target 95%)
    - Gate Strength: Magnitude of Hebbian gating (x_sparse * y_sparse)

    These metrics show how the model actually learns, not just parameter sizes.
    """
    if not metrics:
        return go.Figure()

    iterations = [m['iteration'] for m in metrics]

    # Activation rate metrics (CORRECTED v14+: use activation_rate, fallback to old deprecated names)
    x_activation = [m.get('activation_rate') or m.get('hebbian_x_sparsity') or m.get('hebbian_x_sparse') for m in metrics
                    if m.get('activation_rate') is not None or m.get('hebbian_x_sparsity') is not None or m.get('hebbian_x_sparse') is not None]
    y_activation = [m.get('y_activation_rate') or m.get('hebbian_y_sparsity') or m.get('hebbian_y_sparse') for m in metrics
                    if m.get('y_activation_rate') is not None or m.get('hebbian_y_sparsity') is not None or m.get('hebbian_y_sparse') is not None]
    gate_strength = [m.get('hebbian_gate_strength') or m.get('hebbian_xy_magnitude') for m in metrics
                     if m.get('hebbian_gate_strength') is not None or m.get('hebbian_xy_magnitude') is not None]

    # Filter iterations to match available data
    iterations_with_data = [m['iteration'] for m in metrics
                           if m.get('activation_rate') is not None or m.get('hebbian_x_sparsity') is not None or m.get('hebbian_x_sparse') is not None]

    fig = go.Figure()

    # Activation rate metrics (fraction of active neurons, 0-1 range, HIGHER is better)
    if x_activation:
        fig.add_trace(go.Scatter(
            x=iterations_with_data,
            y=[s * 100 for s in x_activation],  # Convert to percentage
            mode='lines',
            name='X Activation Rate (%) - Target: 95%',
            line=dict(color='#e74c3c', width=2),
            yaxis='y'
        ))

    if y_activation:
        fig.add_trace(go.Scatter(
            x=iterations_with_data,
            y=[s * 100 for s in y_activation],  # Convert to percentage
            mode='lines',
            name='Y Activation Rate (%) - Target: 95%',
            line=dict(color='#3498db', width=2),
            yaxis='y'
        ))

    # Gate strength (magnitude, separate axis)
    if gate_strength:
        fig.add_trace(go.Scatter(
            x=iterations_with_data,
            y=gate_strength,
            mode='lines',
            name='Gate Strength',
            line=dict(color='#2ecc71', width=2, dash='dot'),
            yaxis='y2'
        ))

    fig.update_layout(
        title="Hebbian Learning Dynamics: Activation Rate & Gate Strength",
        xaxis_title="Iteration",
        yaxis=dict(
            title=dict(text="Activation Rate (% Active Neurons)", font=dict(color='#e74c3c')),
            tickfont=dict(color='#e74c3c'),
            range=[0, 100]  # 0-100% range for activation rate
        ),
        yaxis2=dict(
            title=dict(text="Gate Strength (Magnitude)", font=dict(color='#2ecc71')),
            tickfont=dict(color='#2ecc71'),
            overlaying='y',
            side='right'
        ),
        template="plotly_white",
        hovermode='x unified',
        annotations=[dict(
            text="Higher activation rate = better (target: 95%). Activation rate averaged across all IFP iterations. Gate strength = x*y magnitude.",
            xref="paper", yref="paper",
            x=0.5, y=1.05,
            showarrow=False,
            font=dict(size=10, color='gray')
        )]
    )

    return fig


def create_gradient_chart(metrics):
    """Gradient norm tracking - separate lines per dataset"""
    if not metrics:
        return go.Figure()

    # Split metrics by dataset (for curriculum learning)
    tinystories_metrics = [m for m in metrics if 'dataset' in m and 'TinyStories' in m['dataset']]
    instruction_metrics = [m for m in metrics if 'dataset' in m and 'Instruction' in m['dataset']]

    fig = go.Figure()

    # If we have dataset-specific metrics, show separate lines
    if tinystories_metrics or instruction_metrics:
        if tinystories_metrics:
            fig.add_trace(go.Scatter(
                x=[m['iteration'] for m in tinystories_metrics],
                y=[m.get('grad_norm', m.get('gradient_norm', 0)) for m in tinystories_metrics],
                mode='lines+markers',
                name='TinyStories',
                line=dict(color='#3498db', width=2),
                marker=dict(size=4)
            ))

        if instruction_metrics:
            fig.add_trace(go.Scatter(
                x=[m['iteration'] for m in instruction_metrics],
                y=[m.get('grad_norm', m.get('gradient_norm', 0)) for m in instruction_metrics],
                mode='lines+markers',
                name='Instructions',
                line=dict(color='#e74c3c', width=2),
                marker=dict(size=4)
            ))
    else:
        # Fallback: single line for experiments without dataset field
        iterations = [m['iteration'] for m in metrics]
        grad_norm = [m.get('grad_norm', m.get('gradient_norm', 0)) for m in metrics]
        fig.add_trace(go.Scatter(
            x=iterations,
            y=grad_norm,
            mode='lines',
            name='Gradient Norm',
            line=dict(color='#f39c12', width=2)
        ))

    fig.update_layout(
        title="Gradient Norm - Training Stability (per dataset)",
        xaxis_title="Iteration",
        yaxis_title="Gradient Norm",
        template="plotly_white",
        hovermode='x unified',
        legend=dict(x=0.8, y=0.95)
    )

    return fig


def create_system_chart(metrics, config=None):
    """GPU and system metrics"""
    if not metrics:
        return go.Figure()

    iterations = [m['iteration'] for m in metrics]

    # Determine which GPU is being used from config
    gpu_id = None
    if config and 'device' in config:
        device_str = config['device']
        if 'cuda:0' in device_str:
            gpu_id = 0
        elif 'cuda:1' in device_str:
            gpu_id = 1

    # If gpu_id not specified in config, auto-detect from metrics
    # Find the GPU with highest average utilization
    if gpu_id is None and metrics:
        gpu_util_avg = {}
        for gpu_idx in range(3):  # Check GPU 0, 1, 2
            utils = [m.get(f'gpu{gpu_idx}_util', 0) for m in metrics if m.get(f'gpu{gpu_idx}_util') is not None]
            if utils:
                gpu_util_avg[gpu_idx] = sum(utils) / len(utils)

        if gpu_util_avg:
            # Use GPU with highest average utilization
            gpu_id = max(gpu_util_avg.items(), key=lambda x: x[1])[0]

    # Collect GPU metrics from the correct GPU
    gpu_util = []
    gpu_mem = []
    gpu_temp = []

    for m in metrics:
        # If we know which GPU, use those specific metrics
        if gpu_id is not None:
            util = m.get(f'gpu{gpu_id}_util', None)
            mem = m.get(f'gpu{gpu_id}_mem_used', None)
            temp = m.get(f'gpu{gpu_id}_temp', None)
        else:
            # Fallback: try gpu0, gpu1, or generic gpu keys
            util = m.get('gpu0_util', m.get('gpu1_util', m.get('gpu_util', None)))
            mem = m.get('gpu0_mem_used', m.get('gpu1_mem_used', m.get('gpu_mem_used', None)))
            temp = m.get('gpu0_temp', m.get('gpu1_temp', m.get('gpu_temp', None)))

        if util is not None:
            gpu_util.append(util)
        if mem is not None:
            gpu_mem.append(mem)
        if temp is not None:
            gpu_temp.append(temp)

    if not gpu_util and not gpu_mem and not gpu_temp:
        return go.Figure().add_annotation(
            text="No GPU metrics available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )

    # Build subplot titles with GPU ID
    gpu_label = f"GPU {gpu_id}" if gpu_id is not None else "GPU"
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(f"{gpu_label} Utilization %", f"{gpu_label} Memory (MB)", f"{gpu_label} Temperature (°C)"),
        vertical_spacing=0.12
    )

    if gpu_util:
        fig.add_trace(go.Scatter(
            x=iterations[:len(gpu_util)],
            y=gpu_util,
            mode='lines',
            name='GPU Util %',
            line=dict(color='#16a085', width=2)
        ), row=1, col=1)

    if gpu_mem:
        fig.add_trace(go.Scatter(
            x=iterations[:len(gpu_mem)],
            y=gpu_mem,
            mode='lines',
            name='GPU Memory (MB)',
            line=dict(color='#c0392b', width=2)
        ), row=2, col=1)

    if gpu_temp:
        fig.add_trace(go.Scatter(
            x=iterations[:len(gpu_temp)],
            y=gpu_temp,
            mode='lines',
            name='GPU Temp (°C)',
            line=dict(color='#e67e22', width=2)
        ), row=3, col=1)

    # Add GPU ID to title if known
    title = "System Resources"
    if gpu_id is not None:
        title = f"System Resources (GPU {gpu_id})"

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode='x unified',
        height=800
    )

    fig.update_xaxes(title_text="Iteration", row=3, col=1)
    fig.update_yaxes(title_text="%", range=[0, 100], row=1, col=1)  # GPU utilization 0-100%
    fig.update_yaxes(title_text="MB", row=2, col=1)
    fig.update_yaxes(title_text="°C", row=3, col=1)

    return fig


def create_comparison_loss_chart(metrics1, metrics2, name1, name2):
    """Overlay loss curves for two experiments"""
    fig = go.Figure()

    if metrics1:
        iterations1 = [m['iteration'] for m in metrics1]
        loss1 = [m.get('loss', m.get('train_loss', m.get('loss_task', m.get('loss_total', 0)))) for m in metrics1]

        fig.add_trace(go.Scatter(
            x=iterations1,
            y=loss1,
            mode='lines',
            name=name1,
            line=dict(color='#3498db', width=2),
            opacity=0.8
        ))

    if metrics2:
        iterations2 = [m['iteration'] for m in metrics2]
        loss2 = [m.get('loss', m.get('train_loss', m.get('loss_task', m.get('loss_total', 0)))) for m in metrics2]

        fig.add_trace(go.Scatter(
            x=iterations2,
            y=loss2,
            mode='lines',
            name=name2,
            line=dict(color='#e74c3c', width=2),
            opacity=0.8
        ))

    # Add 30K iteration marker
    fig.add_vline(x=30000, line_dash="dash", line_color="gray",
                  annotation_text="LR Decay Start (v2)", annotation_position="top")

    fig.update_layout(
        title="Loss Comparison",
        xaxis_title="Iteration",
        yaxis_title="Loss",
        template="plotly_white",
        hovermode='x unified',
        height=400
    )

    return fig


def create_comparison_accuracy_chart(metrics1, metrics2, name1, name2):
    """Overlay accuracy curves for two experiments"""
    fig = go.Figure()

    if metrics1:
        iterations1 = [m['iteration'] for m in metrics1]
        acc1 = [m.get('accuracy', m.get('train_accuracy', 0)) for m in metrics1]

        fig.add_trace(go.Scatter(
            x=iterations1,
            y=acc1,
            mode='lines',
            name=name1,
            line=dict(color='#3498db', width=2),
            opacity=0.8
        ))

    if metrics2:
        iterations2 = [m['iteration'] for m in metrics2]
        acc2 = [m.get('accuracy', m.get('train_accuracy', 0)) for m in metrics2]

        fig.add_trace(go.Scatter(
            x=iterations2,
            y=acc2,
            mode='lines',
            name=name2,
            line=dict(color='#e74c3c', width=2),
            opacity=0.8
        ))

    # Add 30K iteration marker
    fig.add_vline(x=30000, line_dash="dash", line_color="gray",
                  annotation_text="LR Decay Start (v2)", annotation_position="top")

    fig.update_layout(
        title="Accuracy Comparison",
        xaxis_title="Iteration",
        yaxis_title="Accuracy (%)",
        template="plotly_white",
        hovermode='x unified',
        height=400
    )

    return fig


# Layout
layout = html.Div([
    dcc.Location(id='url', refresh=False),

    dbc.Row([
        dbc.Col([
            html.Div(id='experiment-header'),
            html.Hr()
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                dbc.Tab(label="README", tab_id="readme"),
                dbc.Tab(label="Training Metrics", tab_id="metrics"),
                dbc.Tab(label="Compare", tab_id="compare"),
                dbc.Tab(label="Fisher Manifold", tab_id="fisher"),
                dbc.Tab(label="Inference", tab_id="inference"),
                dbc.Tab(label="Neural Microscope", tab_id="microscope"),
                dbc.Tab(label="Config", tab_id="config"),
            ], id="tabs", active_tab="readme"),
            html.Div(id='tab-content', className="mt-3")
        ])
    ]),

    # Auto-refresh metrics - only when viewing metrics tab AND experiment is running
    # Disabled by default, enabled by callback when conditions met
    dcc.Interval(id='metrics-refresh', interval=10000, n_intervals=0, disabled=True),

    # Persistent model state store (survives tab switches)
    # Shared across Inference and Neural Microscope tabs
    dcc.Store(id='experiment-model-store', storage_type='session')
])


@callback(
    Output('experiment-header', 'children'),
    Input('url', 'search')
)
def update_header(search):
    """Update experiment header"""
    if not search or '?path=' not in search:
        return dbc.Alert("No experiment selected", color="warning")

    exp_path = search.split('?path=')[1]
    exp_name = Path(exp_path).name

    # Get latest run
    runs_dir = Path(exp_path) / "runs"
    if not runs_dir.exists():
        return dbc.Alert("No runs found", color="warning")

    runs = sorted([r for r in runs_dir.iterdir() if r.is_dir()], key=lambda r: r.stat().st_mtime, reverse=True)
    if not runs:
        return dbc.Alert("No runs found", color="warning")

    latest_run = runs[0]
    config = load_config(latest_run)

    # Model info
    model_info = "Unknown"
    if 'model' in config:
        m = config['model']
        n_layer = m.get('n_layer', '?')
        n_embd = m.get('n_embd', '?')
        n_head = m.get('n_head', '?')
        model_info = f"{n_layer} iterations × {n_embd}D × {n_head} heads"

    return html.Div([
        html.H2([
            html.I(className="bi bi-folder me-3"),
            exp_name
        ]),
        html.P([
            html.Span(model_info, className="badge bg-primary me-2"),
            html.Span(f"Run: {latest_run.name}", className="badge bg-secondary me-2"),
        ]),
        html.Div([
            dbc.Button([
                html.I(className="bi bi-arrow-left me-2"),
                "Back to Gallery"
            ], href="/", color="light", size="sm", outline=True, className="me-2"),
            dbc.Button([
                html.I(className="bi bi-file-earmark-pdf me-2"),
                "Generate Report"
            ], id="generate-report-btn", color="success", size="sm", outline=True),
            dcc.Download(id="download-report")
        ])
    ])


@callback(
    Output('metrics-refresh', 'disabled'),
    Input('tabs', 'active_tab'),
    Input('url', 'search')
)
def control_auto_refresh(active_tab, search):
    """
    Enable auto-refresh ONLY when:
    1. User is viewing the metrics tab
    2. Experiment has runs (is active or has been run)

    Otherwise disable to prevent unnecessary refreshes.
    """
    import time

    # Disable if not on metrics tab
    if active_tab != 'metrics':
        return True  # Disabled

    # Disable if no experiment selected
    if not search or '?path=' not in search:
        return True  # Disabled

    # Check if experiment is actively running (metrics updated in last 2 minutes)
    exp_path = search.split('?path=')[1]
    runs_dir = Path(exp_path) / "runs"

    if not runs_dir.exists():
        return True  # Disabled - no runs

    runs = [r for r in runs_dir.iterdir() if r.is_dir() and not r.name.startswith('.')]
    if not runs:
        return True  # Disabled - no runs

    # Check latest run for recent activity
    latest_run = max(runs, key=lambda r: r.stat().st_mtime)

    # Check for metrics files
    metrics_files = [
        latest_run / "metrics.jsonl",
        latest_run / "phase1_metrics.jsonl",
        latest_run / "phase2_metrics.jsonl",
    ]

    for metrics_file in metrics_files:
        if metrics_file.exists():
            age_seconds = time.time() - metrics_file.stat().st_mtime
            if age_seconds < 120:  # Updated in last 2 minutes
                return False  # Enabled - experiment is running!

    # No recent activity - disable refresh
    return True  # Disabled


@callback(
    Output('tab-content', 'children'),
    Input('tabs', 'active_tab'),
    Input('metrics-refresh', 'n_intervals'),
    State('url', 'search')
)
def update_tab(active_tab, n, search):
    """Update tab content"""
    from dash import ctx

    # Only refresh metrics tab, never refresh inference tab
    if ctx.triggered_id == 'metrics-refresh' and active_tab != 'metrics':
        from dash import no_update
        return no_update

    if not search or '?path=' not in search:
        return dbc.Alert("No experiment selected", color="warning")

    exp_path = search.split('?path=')[1]

    # Get latest run
    runs_dir = Path(exp_path) / "runs"
    if not runs_dir.exists():
        return dbc.Alert("No runs found", color="warning")

    runs = sorted([r for r in runs_dir.iterdir() if r.is_dir()], key=lambda r: r.stat().st_mtime, reverse=True)
    if not runs:
        return dbc.Alert("No runs found", color="warning")

    latest_run = runs[0]

    if active_tab == "metrics":
        metrics = load_metrics(latest_run)

        if not metrics:
            return dbc.Alert("No metrics found. Training may not have started yet.", color="info")

        latest = metrics[-1]
        config = load_config(latest_run)

        # Calculate progress metrics
        current_iter = latest['iteration']
        max_iters = config.get('max_iters', 58593)
        progress_pct = (current_iter / max_iters) * 100
        remaining_iters = max_iters - current_iter

        # Calculate elapsed time from timestamps
        first_timestamp = metrics[0].get('timestamp', 0)
        last_timestamp = latest.get('timestamp', 0)
        elapsed_sec = last_timestamp - first_timestamp
        hours_elapsed = elapsed_sec / 3600

        # Calculate iterations per second
        if current_iter > 0 and elapsed_sec > 0:
            iters_per_sec = current_iter / elapsed_sec
        else:
            # Fallback to reported iter_time_ms
            iter_time_ms = latest.get('iter_time_ms', 800)
            iters_per_sec = 1000 / iter_time_ms if iter_time_ms > 0 else 0

        # Estimate time remaining based on average iteration time
        if current_iter > 0 and elapsed_sec > 0:
            avg_iter_time_sec = elapsed_sec / current_iter
            time_remaining_sec = remaining_iters * avg_iter_time_sec
        else:
            # Fallback to reported iter_time_ms
            iter_time_ms = latest.get('iter_time_ms', 800)
            time_remaining_sec = (remaining_iters * iter_time_ms) / 1000
        hours_remaining = time_remaining_sec / 3600

        # Recent trend analysis (last 10 metrics)
        recent_metrics = metrics[-10:] if len(metrics) >= 10 else metrics
        # Support multiple field name formats: 'loss', 'train_loss'
        recent_losses = [m.get('loss', m.get('train_loss', m.get('loss_task', m.get('loss_total', 0)))) for m in recent_metrics]
        # Support multiple field name formats: 'accuracy', 'train_accuracy', 'train_acc'
        recent_accs = [m.get('accuracy', m.get('train_accuracy', m.get('train_acc', 0))) for m in recent_metrics]

        # Volatility indicators
        loss_volatility = max(recent_losses) - min(recent_losses) if recent_losses else 0
        acc_volatility = max(recent_accs) - min(recent_accs) if recent_accs else 0
        loss_trend = "↗" if len(recent_losses) >= 2 and recent_losses[-1] > recent_losses[0] else "↘"
        acc_trend = "↗" if len(recent_accs) >= 2 and recent_accs[-1] > recent_accs[0] else "↘"

        # Stability assessment
        if loss_volatility > 0.5:
            stability_color = "danger"
            stability_text = "High Volatility"
            stability_icon = "bi-exclamation-triangle-fill"
        elif loss_volatility > 0.2:
            stability_color = "warning"
            stability_text = "Moderate Volatility"
            stability_icon = "bi-exclamation-circle"
        else:
            stability_color = "success"
            stability_text = "Stable"
            stability_icon = "bi-check-circle-fill"

        return html.Div([
            # Progress Bar
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H6([
                            html.I(className="bi bi-clock-history me-2"),
                            "Training Progress"
                        ], className="mb-2"),
                        dbc.Progress(
                            value=progress_pct,
                            label=f"{progress_pct:.1f}%",
                            color="primary",
                            className="mb-2",
                            style={"height": "25px"}
                        ),
                        dbc.Row([
                            dbc.Col([
                                html.Small([
                                    html.Strong("Current: "),
                                    f"{current_iter:,}/{max_iters:,} iterations"
                                ])
                            ], md=3),
                            dbc.Col([
                                html.Small([
                                    html.Strong("Speed: "),
                                    f"{iters_per_sec:.2f} iter/s"
                                ])
                            ], md=3),
                            dbc.Col([
                                html.Small([
                                    html.Strong("Elapsed: "),
                                    f"{hours_elapsed:.1f}h"
                                ])
                            ], md=3),
                            dbc.Col([
                                html.Small([
                                    html.Strong("Remaining: "),
                                    f"~{hours_remaining:.1f}h ({remaining_iters:,} iters)"
                                ])
                            ], md=3),
                        ])
                    ])
                ])
            ], className="mb-3"),

            # Summary stats with trend indicators
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Loss", className="text-muted mb-1"),
                            html.Div([
                                html.H3(f"{latest.get('loss', latest.get('train_loss', latest.get('loss_task', latest.get('loss_total', 0)))):.4f}", className="d-inline me-2"),
                                html.Span(loss_trend, className="fs-4")
                            ])
                        ])
                    ])
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Accuracy", className="text-muted mb-1"),
                            html.Div([
                                html.H3(f"{latest.get('accuracy', latest.get('train_accuracy', latest.get('train_acc', 0))) * 100:.2f}%", className="d-inline me-2"),
                                html.Span(acc_trend, className="fs-4")
                            ])
                        ])
                    ])
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Stability (Last 10)", className="text-muted mb-1"),
                            html.Div([
                                html.I(className=f"bi {stability_icon} me-2"),
                                html.Span(stability_text, className=f"text-{stability_color}")
                            ]),
                            html.Small(f"Loss range: {loss_volatility:.3f}", className="text-muted")
                        ])
                    ])
                ], md=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Gradient Norm", className="text-muted mb-1"),
                            html.H3(f"{latest.get('grad_norm', latest.get('gradient_norm', 0)):.3f}")
                        ])
                    ])
                ], md=3),
            ], className="mb-4"),

            # Recent Trend Table
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="bi bi-graph-up me-2"),
                    "Recent Trend (Last 10 Iterations)"
                ]),
                dbc.CardBody([
                    dbc.Table([
                        html.Thead([
                            html.Tr([
                                html.Th("Iteration"),
                                html.Th("Loss"),
                                html.Th("Accuracy"),
                                html.Th("Perplexity"),
                                html.Th("LR") if recent_metrics and 'learning_rate' in recent_metrics[0] else None
                            ])
                        ]),
                        html.Tbody([
                            html.Tr([
                                html.Td(f"{m['iteration']:,}"),
                                html.Td(f"{m.get('loss', m.get('train_loss', m.get('loss_task', m.get('loss_total', 0)))):.4f}"),
                                html.Td(f"{m.get('accuracy', m.get('train_accuracy', m.get('train_acc', 0))) * 100:.1f}%"),
                                # Calculate perplexity from loss if not explicitly logged
                                html.Td(f"{m.get('perplexity', 2.718281828 ** m.get('loss', m.get('train_loss', m.get('loss_task', m.get('loss_total', 0)))) if 'loss' in m or 'train_loss' in m else 0):.2f}"),
                                html.Td(f"{m.get('learning_rate', 0):.2e}") if 'learning_rate' in m else None
                            ]) for m in reversed(recent_metrics)
                        ])
                    ], bordered=True, hover=True, size="sm", className="mb-0")
                ])
            ], className="mb-4"),

            # Charts
            dbc.Row([
                dbc.Col([dcc.Graph(figure=create_loss_chart(metrics))], md=6),
                dbc.Col([dcc.Graph(figure=create_accuracy_chart(metrics))], md=6),
            ]),
            dbc.Row([
                dbc.Col([dcc.Graph(figure=create_hebbian_chart(metrics))], md=6),
                dbc.Col([dcc.Graph(figure=create_gradient_chart(metrics))], md=6),
            ]),
            dbc.Row([
                dbc.Col([dcc.Graph(figure=create_system_chart(metrics, config))], md=12),
            ])
        ])

    elif active_tab == "readme":
        readme_content = load_readme(exp_path)
        return dbc.Card([
            dbc.CardBody([
                dcc.Markdown(readme_content, className="markdown-content")
            ])
        ])

    elif active_tab == "fisher":
        # Fisher Manifold Visualization Tab
        checkpoints_dir = Path(exp_path) / "checkpoints"

        # Check for Fisher matrices
        fisher_phase1 = checkpoints_dir / "fisher_phase1.pt"
        fisher_phase2 = checkpoints_dir / "fisher_phase2.pt"

        if not fisher_phase1.exists() and not fisher_phase2.exists():
            return dbc.Alert([
                html.H4("Fisher Information Not Available", className="alert-heading"),
                html.P("Fisher matrices will be generated after Phase 1 and Phase 2 training completes."),
                html.Hr(),
                html.P([
                    "Fisher information captures parameter importance and enables visualization of the ",
                    html.Strong("semantic manifold"), " - the low-dimensional structure where BDH neurons concentrate."
                ], className="mb-0")
            ], color="info")

        # Default to Phase 2 if available, else Phase 1
        fisher_path = fisher_phase2 if fisher_phase2.exists() else fisher_phase1
        phase_label = "Phase 2" if fisher_phase2.exists() else "Phase 1"

        try:
            # Generate visualization
            fig, stats = generate_fisher_visualization(
                fisher_path,
                layer_idx=None,  # All layers
                top_k=500,  # Top 500 parameters
                method='mds',
                k_neighbors=10
            )

            return dbc.Container([
                dbc.Row([
                    dbc.Col([
                        html.H3([
                            html.I(className="bi bi-diagram-3 me-2"),
                            "Fisher-Informed Semantic Manifold"
                        ]),
                        html.P([
                            "3D visualization of BDH's parameter space using ",
                            html.Strong("Fisher information distance"),
                            " (not Euclidean). Colors show parameter importance, borders show local curvature."
                        ], className="text-muted"),
                    ]),
                ]),

                html.Hr(),

                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.Strong("Visualization Controls")),
                            dbc.CardBody([
                                dbc.Row([
                                    dbc.Col([
                                        html.Label("Phase:"),
                                        dcc.Dropdown(
                                            id='fisher-phase-selector',
                                            options=[
                                                {'label': 'Phase 1', 'value': 'phase1', 'disabled': not fisher_phase1.exists()},
                                                {'label': 'Phase 2', 'value': 'phase2', 'disabled': not fisher_phase2.exists()},
                                            ],
                                            value='phase2' if fisher_phase2.exists() else 'phase1'
                                        ),
                                    ], md=6),

                                    dbc.Col([
                                        html.Label("Layer Filter:"),
                                        dcc.Dropdown(
                                            id='fisher-layer-selector',
                                            options=[
                                                {'label': 'All Layers', 'value': 'all'},
                                                {'label': 'Layer 0', 'value': 0},
                                                {'label': 'Layer 1', 'value': 1},
                                                {'label': 'Layer 2', 'value': 2},
                                                {'label': 'Layer 3', 'value': 3},
                                                {'label': 'Layer 4', 'value': 4},
                                                {'label': 'Layer 5', 'value': 5},
                                            ],
                                            value='all'
                                        ),
                                    ], md=6),
                                ]),
                            ])
                        ], className="mb-3"),
                    ]),
                ]),

                dbc.Row([
                    dbc.Col([
                        dcc.Graph(figure=fig, style={'height': '800px'}),
                    ], md=12),
                ]),

                html.Hr(),

                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.Strong(f"Fisher Statistics ({phase_label})")),
                            dbc.CardBody([
                                dbc.Table([
                                    html.Tbody([
                                        html.Tr([
                                            html.Td(html.Strong("Parameters Visualized:")),
                                            html.Td(f"{stats['n_parameters']:,}")
                                        ]),
                                        html.Tr([
                                            html.Td(html.Strong("Mean Importance:")),
                                            html.Td(f"{stats['mean_importance']:.6f}")
                                        ]),
                                        html.Tr([
                                            html.Td(html.Strong("Max Importance:")),
                                            html.Td(f"{stats['max_importance']:.6f}")
                                        ]),
                                        html.Tr([
                                            html.Td(html.Strong("Mean Curvature:")),
                                            html.Td(f"{stats['mean_curvature']:.4f}")
                                        ]),
                                        html.Tr([
                                            html.Td(html.Strong("Max Curvature:")),
                                            html.Td(f"{stats['max_curvature']:.4f}")
                                        ]),
                                    ])
                                ], bordered=True, striped=True),
                            ])
                        ]),
                    ], md=6),

                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.Strong("Top 10 Most Important Parameters")),
                            dbc.CardBody([
                                html.Ul([
                                    html.Li(param, style={'font-family': 'monospace', 'font-size': '0.9em'})
                                    for param in stats['top_parameters']
                                ])
                            ])
                        ]),
                    ], md=6),
                ]),

                html.Hr(),

                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardHeader(html.Strong("Interpretation Guide")),
                            dbc.CardBody([
                                html.H6("What This Shows:", className="text-primary"),
                                html.Ul([
                                    html.Li([html.Strong("Point Position: "), "Fisher information distance (semantic similarity)"]),
                                    html.Li([html.Strong("Color Intensity: "), "Parameter importance (darker = more critical)"]),
                                    html.Li([html.Strong("Border Color (red): "), "Local curvature (high = category boundary)"]),
                                    html.Li([html.Strong("Clustering: "), "Semantically related parameters group together"]),
                                ]),

                                html.H6("Key Concepts:", className="text-primary mt-3"),
                                html.Ul([
                                    html.Li([html.Strong("Semantic Manifold: "), "Low-dimensional structure where neurons concentrate"]),
                                    html.Li([html.Strong("Fisher Distance: "), "Information-theoretic distance (superior to Euclidean)"]),
                                    html.Li([html.Strong("High Curvature: "), "Sharp semantic transitions (e.g., concept boundaries)"]),
                                    html.Li([html.Strong("High Importance: "), "Parameters critical for preserving learned structure"]),
                                ]),
                            ])
                        ]),
                    ], md=12),
                ]),

            ], fluid=True)

        except Exception as e:
            return dbc.Alert([
                html.H4("Visualization Error", className="alert-heading"),
                html.P(f"Failed to generate Fisher visualization: {str(e)}"),
                html.Hr(),
                html.P("This may occur if Fisher matrices are corrupted or incompatible.", className="mb-0")
            ], color="danger")

    elif active_tab == "inference":
        # Inference tab - use full working interface
        config = load_config(latest_run)
        return create_inference_interface(latest_run, config)

    elif active_tab == "microscope":
        # Neural Microscope tab - integrated visualization
        return create_neural_microscope_interface(exp_path, latest_run.name)

    elif active_tab == "config":
        config = load_config(latest_run)

        # Get GPU allocation info
        try:
            from src.utils.neural_microscope_config import GPUAllocator
            allocator = GPUAllocator()
            inference_device = allocator.get_inference_device()
            rendering_device = allocator.get_rendering_device() or "CPU OpenGL"
            gpu_config = allocator.config.get('gpu_allocation', {})
            available_gpus = allocator.available_gpus
        except Exception as e:
            inference_device = "Unknown"
            rendering_device = "Unknown"
            gpu_config = {}
            available_gpus = {}

        # Build structured config display
        return html.Div([
            # GPU Allocation Section
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="bi bi-gpu-card me-2"),
                    "GPU Allocation (Dashboard-Level)"
                ]),
                dbc.CardBody([
                    html.P("GPU assignments are managed at the dashboard application level, not per-experiment.", className="text-muted mb-3"),

                    dbc.Row([
                        dbc.Col([
                            html.H6("Available GPUs:", className="mb-2"),
                            html.Ul([
                                html.Li([
                                    html.Strong(f"GPU {gpu_id}: "),
                                    f"{props['name']} - ",
                                    html.Span(f"{props['memory_free']:.1f}/{props['memory_total']:.1f} GB free", className="text-muted")
                                ]) for gpu_id, props in available_gpus.items()
                            ]) if available_gpus else html.P("No CUDA GPUs detected", className="text-muted")
                        ], md=6),
                        dbc.Col([
                            html.H6("Current Allocation:", className="mb-2"),
                            dbc.Table([
                                html.Tbody([
                                    html.Tr([html.Td("Inference:"), html.Td(html.Code(inference_device))]),
                                    html.Tr([html.Td("Rendering (UMAP):"), html.Td(html.Code(rendering_device))]),
                                ])
                            ], bordered=True, size="sm")
                        ], md=6)
                    ]),

                    html.Hr(),

                    html.P([
                        html.I(className="bi bi-info-circle me-2"),
                        "Configuration file: ",
                        html.Code("src/utils/neural_microscope_config.py")
                    ], className="mb-0 small text-muted")
                ])
            ], className="mb-3"),

            # Model Configuration Section
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="bi bi-diagram-3 me-2"),
                    "Model Architecture"
                ]),
                dbc.CardBody([
                    dbc.Table([
                        html.Tbody([
                            html.Tr([html.Td(html.Strong("Layers:")), html.Td(config.get('model', {}).get('n_layer', 'N/A'))]),
                            html.Tr([html.Td(html.Strong("Embedding Dim:")), html.Td(config.get('model', {}).get('n_embd', 'N/A'))]),
                            html.Tr([html.Td(html.Strong("Attention Heads:")), html.Td(config.get('model', {}).get('n_head', 'N/A'))]),
                            html.Tr([html.Td(html.Strong("Vocabulary Size:")), html.Td(config.get('model', {}).get('vocab_size', 'N/A'))]),
                            html.Tr([html.Td(html.Strong("Dropout:")), html.Td(config.get('model', {}).get('dropout', 'N/A'))]),
                        ])
                    ], bordered=True, hover=True)
                ])
            ], className="mb-3"),

            # Training Configuration Section
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="bi bi-lightning me-2"),
                    "Training Configuration"
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Hyperparameters:", className="mb-2"),
                            dbc.Table([
                                html.Tbody([
                                    html.Tr([html.Td(html.Strong("Batch Size:")), html.Td(config.get('batch_size', 'N/A'))]),
                                    html.Tr([html.Td(html.Strong("Block Size:")), html.Td(config.get('block_size', 'N/A'))]),
                                    html.Tr([html.Td(html.Strong("Learning Rate:")), html.Td(f"{config.get('learning_rate', 'N/A'):.0e}" if isinstance(config.get('learning_rate'), (int, float)) else 'N/A')]),
                                    html.Tr([html.Td(html.Strong("Max Iterations:")), html.Td(f"{config.get('max_iters', 'N/A'):,}" if isinstance(config.get('max_iters'), int) else 'N/A')]),
                                    html.Tr([html.Td(html.Strong("Training Device:")), html.Td(html.Code(config.get('device', 'N/A')))]),
                                ])
                            ], bordered=True, size="sm")
                        ], md=6),
                        dbc.Col([
                            html.H6("Adaptive Stopping:", className="mb-2"),
                            dbc.Table([
                                html.Tbody([
                                    html.Tr([html.Td(html.Strong("Convergence Window:")), html.Td(f"{config.get('convergence_window', 'N/A'):,}" if isinstance(config.get('convergence_window'), int) else 'N/A')]),
                                    html.Tr([html.Td(html.Strong("Convergence Threshold:")), html.Td(config.get('convergence_threshold', 'N/A'))]),
                                    html.Tr([html.Td(html.Strong("Grokking Window:")), html.Td(f"{config.get('grokking_window', 'N/A'):,}" if isinstance(config.get('grokking_window'), int) else 'N/A')]),
                                    html.Tr([html.Td(html.Strong("Grokking Threshold:")), html.Td(f"{config.get('grokking_threshold', 'N/A')}%" if config.get('grokking_threshold') else 'N/A')]),
                                    html.Tr([html.Td(html.Strong("Min Iterations:")), html.Td(f"{config.get('min_iterations', 'N/A'):,}" if isinstance(config.get('min_iterations'), int) else 'N/A')]),
                                ])
                            ], bordered=True, size="sm")
                        ], md=6)
                    ])
                ])
            ], className="mb-3"),

            # Dataset Section
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="bi bi-file-text me-2"),
                    "Dataset"
                ]),
                dbc.CardBody([
                    dbc.Table([
                        html.Tbody([
                            html.Tr([html.Td(html.Strong("Path:")), html.Td(html.Code(config.get('dataset_path', 'N/A')))]),
                        ])
                    ], bordered=True)
                ])
            ], className="mb-3"),

            # Raw JSON (collapsible)
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="bi bi-code-square me-2"),
                    "Raw Configuration (JSON)"
                ]),
                dbc.CardBody([
                    html.Details([
                        html.Summary("Click to expand", style={"cursor": "pointer", "color": "#0d6efd"}),
                        html.Pre(json.dumps(config, indent=2), className="bg-light p-3 mt-2")
                    ])
                ])
            ])
        ])

    elif active_tab == "compare":
        # Compare tab - find related experiments for comparison
        current_exp_name = Path(exp_path).name

        # Find companion experiments (e.g., v1 vs v2)
        related_experiments = []

        # Check for v1/v2 variants
        if current_exp_name.endswith('_v2'):
            base_name = current_exp_name[:-3]  # Remove _v2
            v1_path = EXPERIMENTS_DIR / base_name
            if v1_path.exists():
                related_experiments.append(('v1', base_name, v1_path))
        elif not current_exp_name.endswith('_v2'):
            # This might be v1, check for v2
            v2_path = EXPERIMENTS_DIR / f"{current_exp_name}_v2"
            if v2_path.exists():
                related_experiments.append(('v2', f"{current_exp_name}_v2", v2_path))

        if not related_experiments:
            return dbc.Alert([
                html.I(className="bi bi-info-circle me-2"),
                "No related experiments found for comparison. ",
                html.Br(),
                "Comparison works for experiment variants (e.g., POC v1 vs POC_v2)."
            ], color="info")

        # Load metrics for current and related experiments
        current_metrics = load_metrics(latest_run)

        comparison_cards = []

        for variant, exp_name, exp_path_obj in related_experiments:
            runs_dir = exp_path_obj / "runs"
            if not runs_dir.exists():
                continue

            runs = sorted([r for r in runs_dir.iterdir() if r.is_dir() and not r.name.startswith('.')],
                         key=lambda r: r.stat().st_mtime, reverse=True)

            if not runs:
                continue

            related_run = runs[0]
            related_metrics = load_metrics(related_run)
            related_config = load_config(related_run)

            if not related_metrics:
                continue

            # Calculate comparison metrics
            related_latest = related_metrics[-1]
            current_latest = current_metrics[-1] if current_metrics else {}

            # Progress comparison
            current_iter = current_latest.get('iteration', 0)
            related_iter = related_latest.get('iteration', 0)
            current_max = load_config(latest_run).get('max_iters', 58593)
            related_max = related_config.get('max_iters', 58593)
            current_pct = (current_iter / current_max) * 100
            related_pct = (related_iter / related_max) * 100

            # Performance comparison at similar iterations
            # Find closest iterations
            current_at_30k = next((m for m in current_metrics if m['iteration'] >= 30000), None) if current_metrics else None
            related_at_30k = next((m for m in related_metrics if m['iteration'] >= 30000), None) if related_metrics else None

            comparison_cards.append(
                dbc.Card([
                    dbc.CardHeader([
                        html.H5([
                            html.I(className="bi bi-arrow-left-right me-2"),
                            f"Comparing with: {exp_name}"
                        ], className="mb-0")
                    ]),
                    dbc.CardBody([
                        # Side-by-side progress
                        dbc.Row([
                            dbc.Col([
                                html.H6([
                                    html.I(className="bi bi-file-earmark me-2"),
                                    "Current Experiment"
                                ], className="text-primary mb-3"),
                                dbc.ListGroup([
                                    dbc.ListGroupItem([
                                        html.Strong("Progress: "),
                                        f"{current_iter:,}/{current_max:,} ({current_pct:.1f}%)"
                                    ]),
                                    dbc.ListGroupItem([
                                        html.Strong("Current Loss: "),
                                        f"{current_latest.get('loss', current_latest.get('train_loss', 0)):.4f}"
                                    ]),
                                    dbc.ListGroupItem([
                                        html.Strong("Current Accuracy: "),
                                        f"{current_latest.get('accuracy', current_latest.get('train_accuracy', 0)) * 100:.2f}%"
                                    ]),
                                    dbc.ListGroupItem([
                                        html.Strong("LR Decay: "),
                                        "Yes" if 'learning_rate' in current_latest and current_latest.get('learning_rate') < load_config(latest_run).get('learning_rate', 1e-4) else "No"
                                    ])
                                ])
                            ], md=6),
                            dbc.Col([
                                html.H6([
                                    html.I(className="bi bi-file-earmark me-2"),
                                    f"{exp_name}"
                                ], className="text-success mb-3"),
                                dbc.ListGroup([
                                    dbc.ListGroupItem([
                                        html.Strong("Progress: "),
                                        f"{related_iter:,}/{related_max:,} ({related_pct:.1f}%)"
                                    ]),
                                    dbc.ListGroupItem([
                                        html.Strong("Current Loss: "),
                                        f"{related_latest.get('loss', related_latest.get('train_loss', 0)):.4f}"
                                    ]),
                                    dbc.ListGroupItem([
                                        html.Strong("Current Accuracy: "),
                                        f"{related_latest.get('accuracy', related_latest.get('train_accuracy', 0)) * 100:.2f}%"
                                    ]),
                                    dbc.ListGroupItem([
                                        html.Strong("LR Decay: "),
                                        "Yes" if 'learning_rate' in related_latest and related_latest.get('learning_rate') < related_config.get('learning_rate', 1e-4) else "No"
                                    ])
                                ])
                            ], md=6)
                        ], className="mb-4"),

                        # Performance at 30K iterations (if available)
                        html.H6("Performance Comparison @ Iteration 30K", className="mt-4 mb-3") if current_at_30k and related_at_30k else None,
                        dbc.Row([
                            dbc.Col([
                                dbc.ListGroup([
                                    dbc.ListGroupItem([
                                        html.Strong("Loss: "),
                                        f"{current_at_30k.get('loss', current_at_30k.get('train_loss', 0)):.4f}" if current_at_30k else "N/A"
                                    ]),
                                    dbc.ListGroupItem([
                                        html.Strong("Accuracy: "),
                                        f"{current_at_30k.get('accuracy', current_at_30k.get('train_accuracy', 0)) * 100:.2f}%" if current_at_30k else "N/A"
                                    ])
                                ])
                            ], md=6),
                            dbc.Col([
                                dbc.ListGroup([
                                    dbc.ListGroupItem([
                                        html.Strong("Loss: "),
                                        f"{related_at_30k.get('loss', related_at_30k.get('train_loss', 0)):.4f}" if related_at_30k else "N/A"
                                    ]),
                                    dbc.ListGroupItem([
                                        html.Strong("Accuracy: "),
                                        f"{related_at_30k.get('accuracy', related_at_30k.get('train_accuracy', 0)) * 100:.2f}%" if related_at_30k else "N/A"
                                    ])
                                ])
                            ], md=6)
                        ]) if current_at_30k and related_at_30k else None,

                        # Overlay charts
                        html.H6("Loss Comparison", className="mt-4 mb-3"),
                        dcc.Graph(figure=create_comparison_loss_chart(
                            current_metrics, related_metrics,
                            Path(exp_path).name, exp_name
                        )),

                        html.H6("Accuracy Comparison", className="mt-4 mb-3"),
                        dcc.Graph(figure=create_comparison_accuracy_chart(
                            current_metrics, related_metrics,
                            Path(exp_path).name, exp_name
                        ))
                    ])
                ], className="mb-3")
            )

        return html.Div(comparison_cards)

    return html.Div("Unknown tab")


# ============================================================================
# Neural Microscope Callbacks (Phase 2A - Embedded Mode)
# ============================================================================

@callback(
    Output('microscope-model-status', 'children'),
    Output('microscope-initial-info', 'style'),
    Input('experiment-model-store', 'data')
)
def microscope_show_model_status(model_data):
    """Show model status from Inference tab in Neural Microscope"""
    if model_data and model_data.get('loaded') and model_data.get('checkpoint_path'):
        checkpoint = Path(model_data['checkpoint_path']).name
        device = model_data.get('device', 'unknown')
        n_params = model_data.get('n_params', 0)

        status = dbc.Alert([
            html.I(className="bi bi-check-circle me-2"),
            f"Model loaded: {checkpoint} ({n_params:,} params) on {device}"
        ], color="success")

        # Hide initial info when model loaded
        return status, {'display': 'none'}

    # No model loaded - show initial info
    return None, {}


# Clientside callback for instant button feedback (before server processing)
# Note: We removed this approach to avoid duplicate output conflicts.
# Instead, the button feedback is now handled entirely by the server callback.
# The slight delay is acceptable given the ~30s total processing time.


@callback(
    Output('microscope-activations-store', 'data'),
    Output('microscope-run-status', 'children', allow_duplicate=True),
    Output('microscope-run-btn', 'disabled'),
    Output('microscope-run-btn', 'children'),
    Input('microscope-run-btn', 'n_clicks'),
    State('experiment-model-store', 'data'),
    State('microscope-text-input', 'value'),
    State('microscope-activation-threshold', 'value'),
    State('microscope-hebbian-toggle', 'value'),
    prevent_initial_call=True
)
def microscope_run_inference(n_clicks, model_data, text_input, threshold, hebbian_toggle):
    """Run inference and capture activations with progress feedback"""
    default_button = [html.I(className="bi bi-play-fill me-2"), "Run Inference & Visualize"]

    if not model_data or not model_data.get('loaded'):
        return (
            None,
            dbc.Alert([
                html.I(className="bi bi-exclamation-triangle me-2"),
                "Please load a model in the Inference tab first"
            ], color="warning"),
            False,  # Enable button
            default_button
        )

    if not text_input:
        return (
            None,
            dbc.Alert("Please enter text input", color="warning"),
            False,  # Enable button
            default_button
        )

    try:
        # Show loading state
        print("🚀 Starting inference process...")

        # Import BDH Inference Engine
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from neural_microscope import BDHInferenceEngine

        # Initialize engine with checkpoint from Inference tab
        print(f"📂 Loading model from checkpoint...")
        engine = BDHInferenceEngine(
            checkpoint_path=model_data['checkpoint_path'],
            device=model_data.get('device', 'cpu'),
            activation_threshold=threshold
        )

        # Run inference
        print(f"🧠 Running inference on '{text_input}'...")
        result = engine.run_inference(text_input, hebbian_learning=hebbian_toggle)
        print(f"✓ Inference complete, captured {len(result['activations'])} activations")

        # Pre-compute all position modes for each activation
        print("📍 Pre-computing position modes...")
        activations = result['activations']

        # Store weight-space positions (default x/y/z from BDHInferenceEngine)
        for act in activations:
            act['weight_x'] = act['x']
            act['weight_y'] = act['y']
            act['weight_z'] = act['z']

            # Grid positions (simple fallback)
            act['grid_x'] = act['neuron_id'] % 90
            act['grid_y'] = act['neuron_id'] // 90
            act['grid_z'] = act['iteration'] * 4 + act['head']
        print("  ✓ Weight-space and grid positions ready")

        # Compute activation-space positions (input-specific manifold)
        activation_space_computed = False
        try:
            print("  🧮 Computing activation-space positions (UMAP, ~30s)...")
            activation_space = engine.compute_activation_space_positions(activations, method='umap')
            # Update activation records with activation-space positions
            for orig_act, new_act in zip(activations, activation_space):
                orig_act['activation_x'] = new_act['x']
                orig_act['activation_y'] = new_act['y']
                orig_act['activation_z'] = new_act['z']
            print("  ✓ Activation-space positions computed successfully")
            activation_space_computed = True
        except Exception as e:
            print(f"  ⚠️  Could not compute activation-space positions: {e}")
            print("  → Falling back to weight-space for activation mode")
            # Fallback: use weight-space for activation mode
            for act in activations:
                act['activation_x'] = act['weight_x']
                act['activation_y'] = act['weight_y']
                act['activation_z'] = act['weight_z']

        result['activations'] = activations
        print("✅ All position modes ready")

        # Build detailed success message with model output and position mode status
        position_modes_status = []
        if activation_space_computed:
            position_modes_status = [
                html.Br(),
                html.Strong("Position modes: "),
                html.Span("Weight ✓", className="text-success me-2"),
                html.Span("Activation ✓", className="text-success me-2"),
                html.Span("Grid ✓", className="text-success")
            ]
        else:
            position_modes_status = [
                html.Br(),
                html.Strong("Position modes: "),
                html.Span("Weight ✓", className="text-success me-2"),
                html.Span("Activation ⚠ (fallback to weight)", className="text-warning me-2"),
                html.Span("Grid ✓", className="text-success")
            ]

        success_msg = dbc.Alert([
            html.H6([
                html.I(className="bi bi-check-circle me-2"),
                "Inference Complete"
            ], className="mb-2"),
            html.Div([
                html.Strong("Input: "), f'"{result["input_text"]}"',
                html.Br(),
                html.Strong("Output: "), f'"{result["output_text"]}"',
                html.Br(),
                html.Strong("Activations captured: "), f"{len(result['activations']):,}",
                html.Br(),
                html.Strong("Sparsity: "), f"{result['summary_stats']['sparsity_percent']:.4f}%",
                html.Br(),
                html.Strong("Characters processed: "), str(result['num_chars_processed']),
                *position_modes_status
            ], className="small")
        ], color="success")

        return (
            result,
            success_msg,
            False,  # Re-enable button
            default_button  # Reset button text
        )

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"❌ Inference failed: {str(e)}")
        print(error_details)

        return (
            None,
            dbc.Alert([
                html.H6([
                    html.I(className="bi bi-exclamation-triangle me-2"),
                    "Inference Failed"
                ], className="mb-2"),
                html.Div([
                    html.Strong("Error: "), str(e),
                    html.Br(),
                    html.Details([
                        html.Summary("Show full traceback", className="text-muted"),
                        html.Pre(error_details, className="small mt-2")
                    ])
                ])
            ], color="danger"),
            False,  # Re-enable button
            default_button  # Reset button text
        )


@callback(
    Output('microscope-3d-plot', 'figure'),
    Output('microscope-char-selector', 'options'),
    Input('microscope-activations-store', 'data'),
    Input('microscope-activation-type', 'value'),
    Input('microscope-iteration-selector', 'value'),
    Input('microscope-char-selector', 'value'),
    Input('microscope-topk-slider', 'value'),
    Input('microscope-position-mode', 'value')
)
def microscope_update_plot(result, activation_type, selected_iteration, selected_char, topk_percent, position_mode):
    """Update 3D visualization from activations with filtering"""
    # Default char options
    default_char_options = [{'label': 'All Characters', 'value': 'all'}]

    if not result or 'activations' not in result:
        # Empty plot (silently return - this is normal when microscope not in use)
        fig = go.Figure()
        fig.update_layout(
            title="No activations yet. Run inference to visualize.",
            scene=dict(
                xaxis_title='Position X',
                yaxis_title='Position Y',
                zaxis_title='Iteration × Head'
            ),
            height=700
        )
        return fig, default_char_options

    activations = result['activations']

    if not activations:
        print("⚠️  Activations list is empty - returning empty plot")
        fig = go.Figure()
        fig.update_layout(title="No activations captured")
        return fig, default_char_options

    # Build character position options from data
    char_positions = sorted(set(act['char_position'] for act in activations))
    char_options = [{'label': 'All Characters', 'value': 'all'}] + \
                   [{'label': f'Char {i}', 'value': str(i)} for i in char_positions]

    # Filter activations
    print(f"  - Total activations before filtering: {len(activations)}")
    print(f"  - Filter criteria: type={activation_type}, iter={selected_iteration}, char={selected_char}")

    filtered = [
        act for act in activations
        if act['activation_type'] == activation_type and
           (selected_iteration == 'all' or act['iteration'] == int(selected_iteration)) and
           (selected_char == 'all' or act['char_position'] == int(selected_char))
    ]

    print(f"  - Activations after filtering: {len(filtered)}")

    if not filtered:
        print("⚠️  No activations match filters - returning empty plot")
        fig = go.Figure()
        fig.update_layout(title="No activations match the selected filters")
        return fig, char_options

    # Apply Top-K filtering
    original_count = len(filtered)
    if topk_percent < 100:
        # Sort by activation value descending
        filtered.sort(key=lambda x: x['activation_value'], reverse=True)
        # Keep only top K%
        k = max(1, int(len(filtered) * topk_percent / 100))
        filtered = filtered[:k]
        print(f"  - After Top-K ({topk_percent}%): {len(filtered)} of {original_count} activations")

    # Debug: Check if position mode coordinates exist
    if filtered:
        sample = filtered[0]
        print(f"  - Sample activation keys: {list(sample.keys())}")
        print(f"  - Has weight coords: {all(k in sample for k in ['weight_x', 'weight_y', 'weight_z'])}")
        print(f"  - Has activation coords: {all(k in sample for k in ['activation_x', 'activation_y', 'activation_z'])}")
        print(f"  - Has grid coords: {all(k in sample for k in ['grid_x', 'grid_y', 'grid_z'])}")

    # Extract coordinates based on position mode
    if position_mode == 'activation':
        # Activation-space positions (UMAP of activation patterns)
        x_coords = [act.get('activation_x', act['x']) for act in filtered]
        y_coords = [act.get('activation_y', act['y']) for act in filtered]
        z_coords = [act.get('activation_z', act['z']) for act in filtered]
        print(f"  - Using activation-space coordinates")
    elif position_mode == 'grid':
        # Grid positions (simple fallback)
        x_coords = [act.get('grid_x', act['x']) for act in filtered]
        y_coords = [act.get('grid_y', act['y']) for act in filtered]
        z_coords = [act.get('grid_z', act['z']) for act in filtered]
        print(f"  - Using grid coordinates")
    else:  # 'weight' (default)
        # Weight-space positions (PCA of attention weights)
        x_coords = [act.get('weight_x', act['x']) for act in filtered]
        y_coords = [act.get('weight_y', act['y']) for act in filtered]
        z_coords = [act.get('weight_z', act['z']) for act in filtered]
        print(f"  - Using weight-space coordinates")

    values = [act['activation_value'] for act in filtered]
    print(f"  - Extracted {len(x_coords)} coordinates for plotting")

    # Create hover text
    hover_text = [
        f"Iter: {act['iteration']}, Head: {act['head']}<br>"
        f"Neuron: {act['neuron_id']}<br>"
        f"Activation: {act['activation_value']:.4f}<br>"
        f"Char Pos: {act['char_position']}"
        for act in filtered
    ]

    # Create 3D scatter plot
    fig = go.Figure(data=[go.Scatter3d(
        x=x_coords,
        y=y_coords,
        z=z_coords,
        mode='markers',
        marker=dict(
            size=3,
            color=values,
            colorscale='Hot',
            showscale=True,
            colorbar=dict(title="Activation"),
            opacity=0.8
        ),
        text=hover_text,
        hoverinfo='text'
    )])

    # Build title with filter info
    position_mode_label = {
        'weight': 'Weight Space',
        'activation': 'Activation Space (Validated)',
        'grid': 'Grid'
    }.get(position_mode, 'Weight Space')

    title_parts = [f"{activation_type} | {position_mode_label}"]
    if selected_iteration != 'all':
        title_parts.append(f"Iteration {selected_iteration}")
    if selected_char != 'all':
        title_parts.append(f"Char {selected_char}")
    if topk_percent < 100:
        title_parts.append(f"Top {topk_percent:.1f}%: {len(filtered):,} of {original_count:,} neurons")
    else:
        title_parts.append(f"({len(filtered):,} neurons)")

    # Set axis labels based on position mode
    if position_mode == 'activation':
        x_label = 'Activation Space X (UMAP)'
        y_label = 'Activation Space Y (UMAP)'
    elif position_mode == 'grid':
        x_label = 'Grid X (neuron_id % 90)'
        y_label = 'Grid Y (neuron_id // 90)'
    else:  # 'weight'
        x_label = 'Weight Space X (PCA)'
        y_label = 'Weight Space Y (PCA)'

    fig.update_layout(
        title=" | ".join(title_parts),
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title='Z = Iteration × 4 + Head',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        height=700,
        template='plotly_white'
    )

    return fig, char_options


@callback(
    Output('download-report', 'data'),
    Input('generate-report-btn', 'n_clicks'),
    State('url', 'search'),
    prevent_initial_call=True
)
def generate_and_download_report(n_clicks, search):
    """Generate PDF report and trigger download"""
    if not search or '?path=' not in search:
        return None

    exp_path = search.split('?path=')[1]

    # Get latest run
    runs_dir = Path(exp_path) / "runs"
    if not runs_dir.exists():
        return None

    runs = sorted([r for r in runs_dir.iterdir() if r.is_dir()],
                  key=lambda r: r.stat().st_mtime, reverse=True)
    if not runs:
        return None

    latest_run = runs[0]

    try:
        # Import report generator
        sys.path.insert(0, str(BDH_ROOT / "src" / "utils"))
        from report_generator import ExperimentReportGenerator

        # Generate report
        generator = ExperimentReportGenerator(latest_run)
        pdf_path = generator.generate()

        # Return for download
        return dcc.send_file(str(pdf_path))

    except Exception as e:
        print(f"Error generating report: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None
