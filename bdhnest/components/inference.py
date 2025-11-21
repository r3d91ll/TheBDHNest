"""
Inference Component for BDH Dashboard v2

Text generation interface with BDH-correct terminology.
- "Iterations" not "Layers"
- Byte-level character encoding
- Hebbian state visualization
"""

from pathlib import Path
from typing import Dict, List
import torch
import sys

from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go

# Add BDH root to path
BDH_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(BDH_ROOT))

try:
    from bdh import BDH, BDHConfig
    BDH_AVAILABLE = True
except ImportError:
    BDH_AVAILABLE = False


def get_available_checkpoints(run_path: Path) -> List[Dict[str, str]]:
    """
    Find all checkpoints for this run.
    Checks both experiment-level and run-level checkpoint directories.
    """
    checkpoints = []

    # 1. Check experiment-level checkpoints (e.g., experiments/BDH_v8/checkpoints/)
    #    Common for experiments that save phase checkpoints
    exp_path = run_path.parent.parent  # runs/{timestamp} -> runs -> experiment
    exp_checkpoints_dir = exp_path / 'checkpoints'

    if exp_checkpoints_dir.exists():
        for ckpt_file in sorted(exp_checkpoints_dir.glob('*.pt'), reverse=True):
            checkpoints.append({
                'label': f"[Experiment] {ckpt_file.stem}",
                'value': str(ckpt_file)
            })

    # 2. Check run-level checkpoints (e.g., runs/{timestamp}/checkpoints/)
    #    Common for experiments that save iteration checkpoints
    run_checkpoints_dir = run_path / 'checkpoints'

    if run_checkpoints_dir.exists():
        for ckpt_file in sorted(run_checkpoints_dir.glob('*.pt'), reverse=True):
            checkpoints.append({
                'label': f"[Run] {ckpt_file.stem}",
                'value': str(ckpt_file)
            })

    return checkpoints


def create_inference_interface(run_path: Path, config: Dict) -> html.Div:
    """
    Create inference interface for experiment.

    BDH-specific:
    - Character-level generation (vocab_size=256)
    - N iterations through shared encoder per token
    - Hebbian state maintains context
    """
    checkpoints = get_available_checkpoints(run_path)

    if not checkpoints:
        exp_path = run_path.parent.parent
        return dbc.Alert([
            html.H4("No Checkpoints Available", className="alert-heading"),
            html.P("Training needs to progress further before checkpoints are saved."),
            html.Hr(),
            html.P("Looking for checkpoints in:", className="fw-bold mb-1"),
            html.Ul([
                html.Li(html.Code(str(exp_path / 'checkpoints'), className="small")),
                html.Li(html.Code(str(run_path / 'checkpoints'), className="small")),
            ], className="mb-0")
        ], color="info")

    n_iterations = config.get('model', {}).get('n_layer', 12)

    # Get GPU allocation info from app settings
    try:
        import yaml
        config_file = Path(__file__).parent.parent / "config.yaml"
        if config_file.exists():
            with open(config_file) as f:
                app_config = yaml.safe_load(f)
                default_gpu = app_config.get('default_gpu', 0)
                inference_device = f'cuda:{default_gpu}' if torch.cuda.is_available() else 'cpu'
                gpu_note = f"Inference runs on {inference_device} (from Settings page)"
        else:
            inference_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            gpu_note = f"Inference runs on {inference_device} (default, configure in Settings)"
    except Exception:
        inference_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        gpu_note = f"Inference runs on {inference_device} (default)"

    return html.Div([
        # Info Banner
        dbc.Alert([
            html.H5([html.I(className="bi bi-chat-dots me-2"), "Text Generation"], className="alert-heading"),
            html.P([
                "Test model generation with real-time monitoring. ",
                f"Model processes each token through {n_iterations} iterations of the shared encoder."
            ], className="mb-2"),
            html.P([
                html.I(className="bi bi-info-circle me-2"),
                gpu_note
            ], className="mb-0 small text-muted")
        ], color="info", className="mb-3"),

        dbc.Row([
            # Left: Generation Interface
            dbc.Col([
                # Checkpoint Selection
                dbc.Card([
                    dbc.CardHeader("Model Checkpoint"),
                    dbc.CardBody([
                        html.Label("Select Checkpoint:"),
                        dcc.Dropdown(
                            id='inf-checkpoint',
                            options=checkpoints,
                            value=checkpoints[0]['value'] if checkpoints else None,
                            clearable=False,
                            className="mb-3"
                        ),

                        dbc.ButtonGroup([
                            dbc.Button("Load Model", id='inf-load-btn', color="primary"),
                            dbc.Button("Unload", id='inf-unload-btn', color="danger", outline=True)
                        ], className="d-grid")
                    ])
                ], className="mb-3"),

                # Model Status
                html.Div(id='inf-status', className="mb-3"),

                # Generation Controls
                dbc.Card([
                    dbc.CardHeader("Generation Settings"),
                    dbc.CardBody([
                        html.Label("Max Tokens:"),
                        dbc.Input(id='inf-max-tokens', type='number', value=200, min=10, max=1000, className="mb-3"),

                        html.Label("Temperature:"),
                        dbc.Input(id='inf-temperature', type='number', value=1.0, min=0.1, max=2.0, step=0.1, className="mb-3"),

                        html.Label("Prompt:"),
                        dbc.Textarea(
                            id='inf-prompt',
                            placeholder='Enter your prompt...',
                            style={'height': '120px'},
                            className="mb-3"
                        ),

                        dbc.Button(
                            [html.I(className="bi bi-play-fill me-2"), "Generate"],
                            id='inf-generate-btn',
                            color="success",
                            size="lg",
                            className="w-100",
                            disabled=True
                        )
                    ])
                ], className="mb-3"),

                # Output
                dbc.Card([
                    dbc.CardHeader("Generated Output"),
                    dbc.CardBody([
                        html.Pre(
                            id='inf-output',
                            style={
                                'minHeight': '200px',
                                'maxHeight': '400px',
                                'overflowY': 'scroll',
                                'whiteSpace': 'pre-wrap',
                                'backgroundColor': '#f8f9fa',
                                'padding': '10px',
                                'borderRadius': '4px'
                            },
                            children="Generated text will appear here..."
                        )
                    ])
                ])

            ], md=6),

            # Right: Stats and Monitoring
            dbc.Col([
                # Generation Stats
                dbc.Card([
                    dbc.CardHeader("Generation Statistics"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6("Tokens Generated", className="text-muted"),
                                html.H3(id='inf-tokens-generated', children="—")
                            ], md=6),
                            dbc.Col([
                                html.H6("Generation Time", className="text-muted"),
                                html.H3(id='inf-gen-time', children="—")
                            ], md=6)
                        ])
                    ])
                ], className="mb-3"),

                # BDH Architecture Info
                dbc.Card([
                    dbc.CardHeader("BDH Architecture"),
                    dbc.CardBody([
                        html.P([
                            html.Strong("Iterations: "),
                            f"{n_iterations} (passes through shared encoder)"
                        ]),
                        html.P([
                            html.Strong("Encoding: "),
                            "Byte-level (vocab_size=256, 1 char = 1 token)"
                        ]),
                        html.P([
                            html.Strong("Context: "),
                            "Unlimited (Hebbian state maintains long-range dependencies)"
                        ], className="mb-0")
                    ])
                ], className="mb-3"),

                # Generation Tips
                dbc.Card([
                    dbc.CardHeader("Generation Tips"),
                    dbc.CardBody([
                        html.Ul([
                            html.Li("Temperature 0.7-1.0: Balanced creativity"),
                            html.Li("Temperature >1.0: More random/creative"),
                            html.Li("Temperature <0.7: More conservative/repetitive"),
                            html.Li("Character-level: May generate incomplete words at boundaries"),
                            html.Li("First generations may be slow (model loading)")
                        ], className="small mb-0")
                    ])
                ])

            ], md=6)
        ])

        # Note: experiment-model-store is defined at page level to persist across tab switches
        # This store is shared between Inference and Neural Microscope tabs
    ])


# Global model cache (for this process)
_MODEL_CACHE = {}


@callback(
    Output('inf-status', 'children', allow_duplicate=True),
    Output('inf-generate-btn', 'disabled', allow_duplicate=True),
    Input('experiment-model-store', 'data'),
    prevent_initial_call='initial_duplicate'
)
def restore_ui_state(model_data):
    """Restore UI state from stored model data (after tab switch)"""
    if model_data and model_data.get('loaded') and 'model' in _MODEL_CACHE:
        # Model is still loaded in cache
        device = model_data.get('device', 'unknown')
        checkpoint = _MODEL_CACHE.get('checkpoint', 'unknown')
        model = _MODEL_CACHE['model']
        n_params = sum(p.numel() for p in model.parameters())

        return (
            dbc.Alert([
                html.I(className="bi bi-check-circle me-2"),
                f"Model loaded: {checkpoint} ({n_params:,} params) on {device}"
            ], color="success"),
            False  # Enable generate button
        )

    # No model loaded
    return dbc.Alert("Click 'Load Model' to begin", color="light"), True


@callback(
    Output('inf-status', 'children', allow_duplicate=True),
    Output('inf-generate-btn', 'disabled', allow_duplicate=True),
    Output('experiment-model-store', 'data'),
    Input('inf-load-btn', 'n_clicks'),
    Input('inf-unload-btn', 'n_clicks'),
    State('inf-checkpoint', 'value'),
    prevent_initial_call=True
)
def handle_model_loading(load_clicks, unload_clicks, checkpoint_path):
    """Load or unload model (device managed by dashboard config)"""
    from dash import ctx

    if not BDH_AVAILABLE:
        return dbc.Alert("BDH model not available", color="danger"), True, None

    # Get device from dashboard configuration
    # Load default GPU from app settings
    try:
        import yaml
        config_file = Path(__file__).parent.parent / "config.yaml"
        if config_file.exists():
            with open(config_file) as f:
                app_config = yaml.safe_load(f)
                default_gpu = app_config.get('default_gpu', 0)
                device = f'cuda:{default_gpu}' if torch.cuda.is_available() else 'cpu'
        else:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    except Exception:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # Determine which button was clicked
    button_id = ctx.triggered_id

    if button_id == 'inf-unload-btn':
        # Unload model
        _MODEL_CACHE.clear()
        torch.cuda.empty_cache()
        return dbc.Alert("Model unloaded", color="info"), True, None

    if button_id == 'inf-load-btn':
        if not checkpoint_path:
            return dbc.Alert("Select a checkpoint first", color="warning"), True, None

        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')

            # Build config - try checkpoint first, then config.json
            config_dict = checkpoint.get('config', {})
            if 'model' not in config_dict:
                # Try loading from config.json in the run directory
                checkpoint_file = Path(checkpoint_path)
                run_dir = checkpoint_file.parent.parent  # checkpoints/ -> run_dir/
                config_json_path = run_dir / "config.json"

                if config_json_path.exists():
                    import json
                    with open(config_json_path) as f:
                        config_dict = json.load(f)

                if 'model' not in config_dict:
                    return dbc.Alert("No model config in checkpoint or config.json", color="danger"), True, None

            m = config_dict['model']
            config = BDHConfig(
                n_layer=m.get('n_layer', 6),
                n_embd=m.get('n_embd', 256),
                n_head=m.get('n_head', 4),
                vocab_size=m.get('vocab_size', 256),
                dropout=0.0
            )

            # Initialize model
            model = BDH(config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()

            # Cache model
            _MODEL_CACHE['model'] = model
            _MODEL_CACHE['device'] = device
            _MODEL_CACHE['checkpoint'] = Path(checkpoint_path).stem

            # Get param count
            n_params = sum(p.numel() for p in model.parameters())

            return (
                dbc.Alert([
                    html.I(className="bi bi-check-circle me-2"),
                    f"Model loaded: {Path(checkpoint_path).stem} ({n_params:,} params) on {device}"
                ], color="success"),
                False,  # Enable generate button
                {
                    'loaded': True,
                    'device': device,
                    'checkpoint': Path(checkpoint_path).stem,
                    'checkpoint_path': checkpoint_path,  # Full path for Neural Microscope
                    'n_params': n_params
                }
            )

        except Exception as e:
            _MODEL_CACHE.clear()
            return dbc.Alert(f"Error loading model: {str(e)}", color="danger"), True, None

    return dbc.Alert("Click 'Load Model' to begin", color="light"), True, None


@callback(
    Output('inf-output', 'children'),
    Output('inf-tokens-generated', 'children'),
    Output('inf-gen-time', 'children'),
    Input('inf-generate-btn', 'n_clicks'),
    State('inf-prompt', 'value'),
    State('inf-max-tokens', 'value'),
    State('inf-temperature', 'value'),
    State('experiment-model-store', 'data'),
    prevent_initial_call=True
)
def generate_text(n_clicks, prompt, max_tokens, temperature, model_data):
    """Generate text from prompt"""
    import time

    if not model_data or not model_data.get('loaded'):
        return "⚠️ Load a model first", "—", "—"

    if not prompt:
        return "⚠️ Enter a prompt", "—", "—"

    if 'model' not in _MODEL_CACHE:
        return "⚠️ Model not found in cache. Click 'Load Model' again.", "—", "—"

    try:
        model = _MODEL_CACHE['model']
        device = _MODEL_CACHE['device']

        # Encode prompt (byte-level)
        prompt_tokens = torch.tensor(
            [ord(c) % 256 for c in prompt],
            dtype=torch.long
        ).unsqueeze(0).to(device)

        # Generate
        start_time = time.time()
        with torch.no_grad():
            output_tokens = model.generate(
                prompt_tokens,
                max_new_tokens=max_tokens,
                temperature=temperature
            )
        gen_time = time.time() - start_time

        # Decode
        output_text = ''.join([chr(int(t)) for t in output_tokens[0].cpu().numpy()])

        tokens_generated = len(output_tokens[0]) - len(prompt_tokens[0])

        return (
            output_text,
            f"{tokens_generated}",
            f"{gen_time:.2f}s"
        )

    except Exception as e:
        return f"❌ Generation error: {str(e)}", "—", "—"
