"""
Settings Page - Application-Level Configuration

Global settings that apply across all experiments:
- GPU assignment preferences
- Dashboard refresh rates
- Default visualization settings
- System resource monitoring
"""

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from pathlib import Path
import json
import subprocess

dash.register_page(__name__, path='/settings', name='Settings', order=2)

# Config file location
CONFIG_FILE = Path(__file__).parent.parent / "config.yaml"


def get_gpu_info():
    """Get available GPU information"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,temperature.gpu,utilization.gpu', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )

        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    gpus.append({
                        'index': int(parts[0]),
                        'name': parts[1],
                        'memory': parts[2],
                        'temp': parts[3],
                        'util': parts[4]
                    })
            return gpus
    except:
        pass

    return []


def load_app_config():
    """Load application configuration"""
    default_config = {
        'default_gpu': 0,
        'refresh_interval': 5000,  # milliseconds
        'max_experiments_shown': 50,
        'theme': 'dark',
        'enable_neural_microscope': False,
        'checkpoint_auto_backup': False,
    }

    if CONFIG_FILE.exists():
        try:
            import yaml
            with open(CONFIG_FILE) as f:
                loaded = yaml.safe_load(f) or {}
                return {**default_config, **loaded}
        except:
            pass

    return default_config


def save_app_config(config):
    """Save application configuration"""
    try:
        import yaml
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return True
    except Exception as e:
        print(f"Failed to save config: {e}")
        return False


# Layout
layout = html.Div([
    dbc.Row([
        dbc.Col([
            html.H2([
                html.I(className="bi bi-gear-fill me-3"),
                "Application Settings"
            ]),
            html.P("Global configuration affecting all experiments", className="text-muted"),
            html.Hr(),
        ])
    ]),

    # GPU Configuration Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="bi bi-gpu-card me-2"),
                    html.Strong("GPU Assignment")
                ]),
                dbc.CardBody([
                    html.P([
                        "Configure default GPU for new experiments. ",
                        "Individual experiments can override this in their config."
                    ], className="text-muted small"),

                    html.Div(id='gpu-cards', className="mt-3"),

                    dbc.Row([
                        dbc.Col([
                            html.Label("Default GPU:", className="fw-bold"),
                            dcc.Dropdown(
                                id='default-gpu-select',
                                options=[],  # Populated by callback
                                value=0,
                                className="mb-3"
                            ),
                        ], md=6),
                    ]),

                    dbc.Alert(
                        id='gpu-save-status',
                        is_open=False,
                        duration=3000,
                        color='success'
                    ),
                ])
            ], className="mb-4"),
        ], md=12),
    ]),

    # Dashboard Settings Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="bi bi-speedometer2 me-2"),
                    html.Strong("Dashboard Settings")
                ]),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Metrics Refresh Interval:", className="fw-bold"),
                            dcc.Slider(
                                id='refresh-interval-slider',
                                min=1,
                                max=30,
                                step=1,
                                value=5,
                                marks={
                                    1: '1s',
                                    5: '5s',
                                    10: '10s',
                                    15: '15s',
                                    30: '30s'
                                },
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            html.P("How often to refresh metrics during active training",
                                   className="text-muted small mt-2"),
                        ], md=6),

                        dbc.Col([
                            html.Label("Max Experiments in Gallery:", className="fw-bold"),
                            dcc.Input(
                                id='max-experiments-input',
                                type='number',
                                value=50,
                                min=10,
                                max=500,
                                className="form-control mb-2"
                            ),
                            html.P("Limit number of experiments shown in gallery",
                                   className="text-muted small"),
                        ], md=6),
                    ]),

                    html.Hr(),

                    dbc.Row([
                        dbc.Col([
                            dbc.Checklist(
                                id='feature-toggles',
                                options=[
                                    {'label': ' Enable Neural Microscope (experimental)', 'value': 'neural_microscope'},
                                    {'label': ' Auto-backup checkpoints to external storage', 'value': 'checkpoint_backup'},
                                    {'label': ' Show debug information', 'value': 'debug_mode'},
                                ],
                                value=[],
                                className="mb-3"
                            ),
                        ]),
                    ]),

                    dbc.Button(
                        "Save Dashboard Settings",
                        id='save-dashboard-btn',
                        color="primary",
                        className="mt-3"
                    ),

                    dbc.Alert(
                        id='dashboard-save-status',
                        is_open=False,
                        duration=3000,
                        color='success'
                    ),
                ])
            ], className="mb-4"),
        ], md=12),
    ]),

    # System Information Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.I(className="bi bi-info-circle me-2"),
                    html.Strong("System Information")
                ]),
                dbc.CardBody([
                    html.Div(id='system-info'),

                    dbc.Button(
                        [html.I(className="bi bi-arrow-clockwise me-2"), "Refresh"],
                        id='refresh-system-btn',
                        color="secondary",
                        size="sm",
                        className="mt-3"
                    ),
                ])
            ], className="mb-4"),
        ], md=12),
    ]),

    # Store for configuration
    dcc.Store(id='app-config-store', data=load_app_config()),

    # Manual refresh only - no auto-refresh for settings page
    # System info updated only when user clicks refresh button
    dcc.Interval(id='system-refresh-interval', interval=999999999, n_intervals=0, disabled=True),
])


@callback(
    [Output('gpu-cards', 'children'),
     Output('default-gpu-select', 'options'),
     Output('default-gpu-select', 'value')],
    [Input('system-refresh-interval', 'n_intervals'),
     Input('refresh-system-btn', 'n_clicks')],
    State('app-config-store', 'data')
)
def update_gpu_info(n_intervals, n_clicks, config):
    """Update GPU information cards"""
    gpus = get_gpu_info()

    if not gpus:
        return [
            dbc.Alert("No GPUs detected or nvidia-smi not available", color="warning"),
            [],
            0
        ]

    # Create GPU cards
    cards = []
    for gpu in gpus:
        # Determine status color
        try:
            util = float(gpu['util'].replace('%', ''))
            if util > 80:
                color = "danger"
                status = "Heavy Load"
            elif util > 30:
                color = "warning"
                status = "In Use"
            else:
                color = "success"
                status = "Available"
        except:
            color = "secondary"
            status = "Unknown"

        card = dbc.Card([
            dbc.CardBody([
                dbc.Row([
                    dbc.Col([
                        html.H5(f"GPU {gpu['index']}", className="mb-0"),
                        html.P(gpu['name'], className="text-muted small mb-0"),
                    ], width=6),
                    dbc.Col([
                        dbc.Badge(status, color=color, className="float-end"),
                    ], width=6),
                ]),
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.Small("Memory:"),
                        html.Div(gpu['memory'], className="fw-bold"),
                    ], width=4),
                    dbc.Col([
                        html.Small("Temperature:"),
                        html.Div(gpu['temp'], className="fw-bold"),
                    ], width=4),
                    dbc.Col([
                        html.Small("Utilization:"),
                        html.Div(gpu['util'], className="fw-bold"),
                    ], width=4),
                ]),
            ])
        ], className="mb-2", color=color, outline=True)

        cards.append(card)

    # Create dropdown options
    options = [
        {'label': f"GPU {gpu['index']}: {gpu['name']}", 'value': gpu['index']}
        for gpu in gpus
    ]

    default_gpu = config.get('default_gpu', 0)

    return cards, options, default_gpu


@callback(
    Output('system-info', 'children'),
    [Input('system-refresh-interval', 'n_intervals'),
     Input('refresh-system-btn', 'n_clicks')]
)
def update_system_info(n_intervals, n_clicks):
    """Update system information"""
    info = []

    # Dashboard version
    info.append(html.Div([
        html.Strong("Dashboard Version: "),
        html.Span("v2.0 - BDH-Native")
    ], className="mb-2"))

    # Config file location
    info.append(html.Div([
        html.Strong("Config File: "),
        html.Code(str(CONFIG_FILE)),
        html.Span(" ✓ Exists" if CONFIG_FILE.exists() else " ✗ Not Found",
                 className="ms-2 text-success" if CONFIG_FILE.exists() else "ms-2 text-muted")
    ], className="mb-2"))

    # Experiments directory
    exp_dir = Path(__file__).parent.parent.parent.parent.parent / "experiments"
    info.append(html.Div([
        html.Strong("Experiments Directory: "),
        html.Code(str(exp_dir))
    ], className="mb-2"))

    # Count experiments
    if exp_dir.exists():
        exp_count = len([d for d in exp_dir.iterdir() if d.is_dir() and not d.name.startswith('.')])
        info.append(html.Div([
            html.Strong("Total Experiments: "),
            html.Span(str(exp_count))
        ], className="mb-2"))

    return info


@callback(
    Output('gpu-save-status', 'children'),
    Output('gpu-save-status', 'is_open'),
    Output('app-config-store', 'data', allow_duplicate=True),
    Input('default-gpu-select', 'value'),
    State('app-config-store', 'data'),
    prevent_initial_call=True
)
def save_gpu_setting(gpu_value, config):
    """Save GPU assignment"""
    if gpu_value is None:
        return "No GPU selected", False, config

    config['default_gpu'] = gpu_value

    if save_app_config(config):
        return f"Default GPU set to GPU {gpu_value}", True, config
    else:
        return "Failed to save configuration", True, config


@callback(
    Output('dashboard-save-status', 'children'),
    Output('dashboard-save-status', 'is_open'),
    Output('app-config-store', 'data', allow_duplicate=True),
    Input('save-dashboard-btn', 'n_clicks'),
    [State('refresh-interval-slider', 'value'),
     State('max-experiments-input', 'value'),
     State('feature-toggles', 'value'),
     State('app-config-store', 'data')],
    prevent_initial_call=True
)
def save_dashboard_settings(n_clicks, refresh_interval, max_experiments, features, config):
    """Save dashboard settings"""
    if n_clicks is None:
        return "", False, config

    config['refresh_interval'] = refresh_interval * 1000  # Convert to milliseconds
    config['max_experiments_shown'] = max_experiments
    config['enable_neural_microscope'] = 'neural_microscope' in (features or [])
    config['checkpoint_auto_backup'] = 'checkpoint_backup' in (features or [])
    config['debug_mode'] = 'debug_mode' in (features or [])

    if save_app_config(config):
        return "Dashboard settings saved successfully", True, config
    else:
        return "Failed to save configuration", True, config
