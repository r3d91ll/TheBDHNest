"""
Inference Tab - Text Generation

Test model generation quality directly from the dashboard.

BDH-Specific:
- Unlimited sequence length (recurrent Hebbian state)
- Generation continues through N iterations per token
- No KV-cache (attention state is the memory)
"""

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
from pathlib import Path
import json
import torch
import sys

# Add BDH root to path for model imports
BDH_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(BDH_ROOT))

try:
    from bdh import BDH, BDHConfig
    BDH_AVAILABLE = True
except ImportError:
    BDH_AVAILABLE = False

dash.register_page(__name__, path='/inference')


def load_model(checkpoint_path):
    """Load BDH model from checkpoint"""
    if not BDH_AVAILABLE:
        return None, "BDH model not available"

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Get config from checkpoint OR from config.json in run directory
        config_dict = checkpoint.get('config', {})

        if 'model' not in config_dict:
            # Try loading from config.json in the run directory
            checkpoint_file = Path(checkpoint_path)
            # Navigate up to find config.json (checkpoints/ -> run_dir/)
            run_dir = checkpoint_file.parent.parent
            config_json_path = run_dir / "config.json"

            if config_json_path.exists():
                try:
                    with open(config_json_path) as f:
                        config_dict = json.load(f)
                except Exception:
                    pass

        if 'model' in config_dict:
            model_cfg = config_dict['model']
            config = BDHConfig(
                n_layer=model_cfg.get('n_layer', 6),
                n_embd=model_cfg.get('n_embd', 256),
                n_head=model_cfg.get('n_head', 4),
                vocab_size=model_cfg.get('vocab_size', 256),
                dropout=0.0  # Inference mode
            )
        else:
            return None, "No model config found in checkpoint or config.json"

        # Initialize model
        model = BDH(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        return model, None

    except Exception as e:
        return None, f"Error loading model: {str(e)}"


def generate_text(model, prompt, max_tokens=200, temperature=1.0, device='cpu'):
    """
    Generate text from prompt with metrics.

    BDH-specific:
    - Character-level generation (vocab_size=256)
    - Each token processed through N iterations
    - Hebbian state maintains context

    Returns:
        tuple: (output_text, metrics_dict)
    """
    if model is None:
        return "Model not loaded", {}

    try:
        model = model.to(device)

        # Encode prompt (byte-level)
        prompt_tokens = torch.tensor([ord(c) % 256 for c in prompt], dtype=torch.long).unsqueeze(0).to(device)

        # Track metrics
        import time
        import torch.nn.functional as F

        metrics = {
            'prompt_length': len(prompt),
            'tokens_generated': 0,
            'total_time_ms': 0,
            'perplexity': 0.0,
            'token_accuracy': 0.0,
            'avg_loss': 0.0
        }

        start_time = time.time()

        # Generate with loss tracking
        with torch.no_grad():
            # First, compute perplexity on the prompt
            if len(prompt_tokens[0]) > 1:
                logits = model(prompt_tokens)  # (B, T, vocab_size)

                # Compute loss on prompt (next-char prediction)
                targets = prompt_tokens[:, 1:]  # Shift by 1
                logits_for_loss = logits[:, :-1, :]  # Remove last position

                # Flatten for cross entropy
                B, T, V = logits_for_loss.shape
                logits_flat = logits_for_loss.reshape(B * T, V)
                targets_flat = targets.reshape(B * T)

                prompt_loss = F.cross_entropy(logits_flat, targets_flat)
                prompt_perplexity = torch.exp(prompt_loss).item()

                # Token-level accuracy on prompt
                predictions = torch.argmax(logits_for_loss, dim=-1)
                correct = (predictions == targets).sum().item()
                total = targets.numel()
                prompt_accuracy = (correct / total * 100) if total > 0 else 0.0

                metrics['prompt_perplexity'] = prompt_perplexity
                metrics['prompt_accuracy'] = prompt_accuracy

            # Now generate new tokens
            output_tokens = model.generate(
                prompt_tokens,
                max_new_tokens=max_tokens,
                temperature=temperature
            )

            metrics['tokens_generated'] = len(output_tokens[0]) - len(prompt_tokens[0])

        end_time = time.time()
        metrics['total_time_ms'] = (end_time - start_time) * 1000
        metrics['tokens_per_second'] = metrics['tokens_generated'] / ((end_time - start_time) if (end_time - start_time) > 0 else 1)

        # Decode (byte-level)
        output_text = ''.join([chr(int(t)) for t in output_tokens[0].cpu().numpy()])

        return output_text, metrics

    except Exception as e:
        return f"Generation error: {str(e)}", {}


# Layout
layout = html.Div([
    dcc.Location(id='inference-url', refresh=False),
    dcc.Store(id='loaded-model-store'),  # Store loaded model info

    dbc.Row([
        dbc.Col([
            html.H2([
                html.I(className="bi bi-chat-dots me-3"),
                "Text Generation (Inference)"
            ]),
            html.P("Test model generation quality with custom prompts", className="lead text-muted"),
            html.Hr()
        ])
    ]),

    # Model loading section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Model Selection"),
                dbc.CardBody([
                    html.Div(id='inference-model-info'),
                    html.Hr(),
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Checkpoint:"),
                            dcc.Dropdown(
                                id='checkpoint-selector',
                                placeholder="Select checkpoint...",
                                className="mb-2"
                            ),
                        ], md=8),
                        dbc.Col([
                            dbc.Button(
                                [html.I(className="bi bi-upload me-2"), "Load Model"],
                                id='load-model-btn',
                                color="primary",
                                className="w-100"
                            )
                        ], md=4)
                    ]),
                    html.Div(id='model-load-status', className="mt-2")
                ])
            ], className="mb-3")
        ])
    ]),

    # Generation interface
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Text Generation"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Prompt:"),
                            dbc.Textarea(
                                id='generation-prompt',
                                placeholder="Enter your prompt here...",
                                style={'height': '100px'},
                                className="mb-3"
                            ),
                        ], md=12)
                    ]),

                    dbc.Row([
                        dbc.Col([
                            dbc.Label("Max Tokens:"),
                            dbc.Input(
                                id='max-tokens',
                                type='number',
                                value=200,
                                min=10,
                                max=1000,
                                step=10
                            )
                        ], md=4),
                        dbc.Col([
                            dbc.Label("Temperature:"),
                            dbc.Input(
                                id='temperature',
                                type='number',
                                value=1.0,
                                min=0.1,
                                max=2.0,
                                step=0.1
                            )
                        ], md=4),
                        dbc.Col([
                            dbc.Label("Device:"),
                            dcc.Dropdown(
                                id='device-selector',
                                options=[
                                    {'label': 'CPU', 'value': 'cpu'},
                                    {'label': 'CUDA (GPU)', 'value': 'cuda'}
                                ],
                                value='cpu'
                            )
                        ], md=4),
                    ], className="mb-3"),

                    dbc.Button(
                        [html.I(className="bi bi-play-fill me-2"), "Generate"],
                        id='generate-btn',
                        color="success",
                        size="lg",
                        className="w-100 mb-3"
                    ),

                    html.Hr(),

                    dbc.Label("Generated Output:"),
                    html.Pre(
                        id='generation-output',
                        className="bg-light p-3",
                        style={'minHeight': '200px', 'whiteSpace': 'pre-wrap'}
                    ),

                    html.Hr(),

                    dbc.Label("Performance Metrics:"),
                    html.Div(id='generation-metrics', className="mt-2")
                ])
            ])
        ])
    ]),

    # BDH-specific notes
    dbc.Row([
        dbc.Col([
            dbc.Alert([
                html.H5([html.I(className="bi bi-info-circle me-2"), "BDH Generation & Metrics"], className="alert-heading"),
                html.P([
                    html.Strong("Character-Level Generation: "),
                    "BDH uses byte-level encoding (vocab_size=256). Each character is a separate token."
                ], className="mb-2"),
                html.P([
                    html.Strong("Iteration Processing: "),
                    f"Each token is processed through N iterations of the shared encoder/decoder."
                ], className="mb-2"),
                html.P([
                    html.Strong("Unlimited Sequence: "),
                    "Unlike Transformers, BDH has no context window limit. Hebbian attention state maintains long-range dependencies."
                ], className="mb-2"),
                html.Hr(),
                html.P([
                    html.Strong("Performance Metrics: "),
                    "The dashboard displays:"
                ], className="mb-1"),
                html.Ul([
                    html.Li([html.Strong("Prompt Perplexity: "), "How well the model predicts the next character in your prompt (lower is better)"]),
                    html.Li([html.Strong("Prompt Accuracy: "), "Token-level next-character prediction accuracy on your prompt"]),
                    html.Li([html.Strong("Generation Speed: "), "Tokens generated per second (throughput)"]),
                    html.Li([html.Strong("Total Time: "), "End-to-end generation latency including prompt processing"])
                ], className="mb-0")
            ], color="info", className="mt-3")
        ])
    ])
])


@callback(
    Output('inference-model-info', 'children'),
    Output('checkpoint-selector', 'options'),
    Input('inference-url', 'search')
)
def load_experiment_info(search):
    """Load experiment info and available checkpoints"""
    if not search or '?path=' not in search:
        return dbc.Alert("No experiment selected. Go to an experiment detail page first.", color="warning"), []

    exp_path = search.split('?path=')[1]

    # Get latest run
    runs_dir = Path(exp_path) / "runs"
    if not runs_dir.exists():
        return dbc.Alert("No runs found", color="warning"), []

    runs = sorted([r for r in runs_dir.iterdir() if r.is_dir()], key=lambda r: r.stat().st_mtime, reverse=True)
    if not runs:
        return dbc.Alert("No runs found", color="warning"), []

    latest_run = runs[0]

    # Load config
    config_file = latest_run / "config.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
        except:
            config = {}
    else:
        config = {}

    # Model info
    model_info = []
    if 'model' in config:
        m = config['model']
        model_info.append(html.P([
            html.Strong("Architecture: "),
            f"{m.get('n_layer', '?')} iterations × {m.get('n_embd', '?')}D × {m.get('n_head', '?')} heads"
        ], className="mb-1"))
        model_info.append(html.P([
            html.Strong("Vocab Size: "),
            f"{m.get('vocab_size', '?')} (byte-level)"
        ], className="mb-1"))

    model_info.append(html.P([
        html.Strong("Run: "),
        latest_run.name
    ], className="mb-0"))

    # Find checkpoints - check both experiment-level and run-level directories
    checkpoint_options = []
    exp_dir = Path(exp_path)

    # 1. Check experiment-level checkpoints (e.g., experiments/BDH_v8/checkpoints/)
    exp_checkpoint_dir = exp_dir / "checkpoints"
    if exp_checkpoint_dir.exists():
        checkpoints = sorted(exp_checkpoint_dir.glob("*.pt"))
        for ckpt in checkpoints:
            checkpoint_options.append({
                'label': f"[Experiment] {ckpt.name}",
                'value': str(ckpt.absolute())  # Use absolute path
            })

    # 2. Check run-level checkpoints (e.g., runs/{timestamp}/checkpoints/)
    run_checkpoint_dir = latest_run / "checkpoints"
    if run_checkpoint_dir.exists():
        checkpoints = sorted(run_checkpoint_dir.glob("*.pt"))
        for ckpt in checkpoints:
            checkpoint_options.append({
                'label': f"[Run] {ckpt.name}",
                'value': str(ckpt.absolute())  # Use absolute path
            })

    if not checkpoint_options:
        return (
            html.Div(model_info + [
                html.Hr(),
                dbc.Alert([
                    html.H5("No checkpoints found", className="alert-heading"),
                    html.P("Checkpoints will be saved at configured intervals during training."),
                    html.P([
                        "Looking in: ",
                        html.Br(),
                        html.Code(str(exp_checkpoint_dir)),
                        html.Br(),
                        html.Code(str(run_checkpoint_dir))
                    ], className="mb-0 small")
                ], color="warning", className="mb-0")
            ]),
            []
        )

    return html.Div(model_info), checkpoint_options


@callback(
    Output('model-load-status', 'children'),
    Output('loaded-model-store', 'data'),
    Input('load-model-btn', 'n_clicks'),
    State('checkpoint-selector', 'value'),
    prevent_initial_call=True
)
def load_model_callback(n_clicks, checkpoint_path):
    """Load selected checkpoint"""
    import traceback

    print(f"\n[INFERENCE] Load model callback triggered")
    print(f"[INFERENCE] Checkpoint path: {checkpoint_path}")

    if not checkpoint_path:
        msg = "Please select a checkpoint"
        print(f"[INFERENCE] ERROR: {msg}")
        return dbc.Alert(msg, color="warning"), None

    if not BDH_AVAILABLE:
        msg = "BDH model not available. Check imports."
        print(f"[INFERENCE] ERROR: {msg}")
        return dbc.Alert(msg, color="danger"), None

    # Try to load model (just validate, don't keep in memory)
    try:
        print(f"[INFERENCE] Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"[INFERENCE] Checkpoint keys: {list(checkpoint.keys())}")

        # Verify we can load config
        config_dict = checkpoint.get('config', {})
        if 'model' not in config_dict:
            checkpoint_file = Path(checkpoint_path)
            run_dir = checkpoint_file.parent.parent
            config_json_path = run_dir / "config.json"
            print(f"[INFERENCE] Loading config from: {config_json_path}")

            if config_json_path.exists():
                with open(config_json_path) as f:
                    config_dict = json.load(f)
                print(f"[INFERENCE] Config loaded successfully")
            else:
                msg = f"No config.json found at {config_json_path}"
                print(f"[INFERENCE] ERROR: {msg}")
                return dbc.Alert(msg, color="danger", is_open=True, duration=10000), None

        if 'model' not in config_dict:
            msg = "No model config found in checkpoint or config.json"
            print(f"[INFERENCE] ERROR: {msg}")
            return dbc.Alert(msg, color="danger", is_open=True, duration=10000), None

        print(f"[INFERENCE] ✓ Model loaded successfully")
        return (
            dbc.Alert([
                html.I(className="bi bi-check-circle me-2"),
                f"✓ Model loaded successfully from {Path(checkpoint_path).name}"
            ], color="success", is_open=True, duration=5000),
            {'checkpoint_path': checkpoint_path}
        )

    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        print(f"[INFERENCE] ERROR: {error_msg}")
        print(f"[INFERENCE] Traceback:")
        traceback.print_exc()
        return dbc.Alert([
            html.H5("Error Loading Model", className="alert-heading"),
            html.P(error_msg),
            html.Hr(),
            html.P(f"Checkpoint: {checkpoint_path}", className="mb-0 small")
        ], color="danger", is_open=True, duration=10000), None


@callback(
    Output('generation-output', 'children'),
    Output('generation-metrics', 'children'),
    Input('generate-btn', 'n_clicks'),
    State('loaded-model-store', 'data'),
    State('generation-prompt', 'value'),
    State('max-tokens', 'value'),
    State('temperature', 'value'),
    State('device-selector', 'value'),
    prevent_initial_call=True
)
def generate_callback(n_clicks, model_data, prompt, max_tokens, temperature, device):
    """Generate text from prompt with metrics"""
    if not model_data or 'checkpoint_path' not in model_data:
        return "❌ Please load a model first", None

    if not prompt:
        return "❌ Please enter a prompt", None

    # Load model
    model, error = load_model(model_data['checkpoint_path'])
    if error:
        return f"❌ {error}", None

    # Check CUDA availability
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        prefix = "⚠️ CUDA not available, using CPU\n\n"
    else:
        prefix = ""

    # Generate
    output, metrics = generate_text(model, prompt, max_tokens, temperature, device)

    # Build metrics display
    metrics_display = []
    if metrics:
        metrics_display = [
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Prompt Analysis", className="mb-3"),
                            html.P([
                                html.Strong("Prompt Length: "),
                                f"{metrics.get('prompt_length', 0)} characters"
                            ], className="mb-1"),
                            html.P([
                                html.Strong("Prompt Perplexity: "),
                                f"{metrics.get('prompt_perplexity', 0):.2f}"
                            ], className="mb-1") if 'prompt_perplexity' in metrics else None,
                            html.P([
                                html.Strong("Prompt Accuracy: "),
                                f"{metrics.get('prompt_accuracy', 0) * 100:.2f}%"
                            ], className="mb-0") if 'prompt_accuracy' in metrics else None,
                        ])
                    ])
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Generation Stats", className="mb-3"),
                            html.P([
                                html.Strong("Tokens Generated: "),
                                f"{metrics.get('tokens_generated', 0)}"
                            ], className="mb-1"),
                            html.P([
                                html.Strong("Total Time: "),
                                f"{metrics.get('total_time_ms', 0):.0f}ms"
                            ], className="mb-1"),
                            html.P([
                                html.Strong("Speed: "),
                                f"{metrics.get('tokens_per_second', 0):.1f} tokens/sec"
                            ], className="mb-0"),
                        ])
                    ])
                ], md=4),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H6("Quality Metrics", className="mb-3"),
                            html.P([
                                html.Strong("Temperature: "),
                                f"{temperature:.1f}"
                            ], className="mb-1"),
                            html.P([
                                html.Strong("Device: "),
                                device.upper()
                            ], className="mb-1"),
                            html.P([
                                html.Strong("Output Length: "),
                                f"{len(output) - len(prompt)} chars"
                            ], className="mb-0"),
                        ])
                    ])
                ], md=4)
            ])
        ]

    return prefix + output, metrics_display
