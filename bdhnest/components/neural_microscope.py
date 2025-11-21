"""
Neural Microscope Component for BDH Dashboard v2 - Phase 2A

Minimal embedded integration with shared model state.

Phase 2A: Core functionality (inference + 3D plot)
Phase 2B: Enhanced controls (filters, position modes)
Phase 2C: Full feature parity with standalone

Key Design:
- NO experiment selector (embedded in experiment page)
- Uses experiment-model-store (shared with Inference tab)
- Minimal UI for fast implementation
- Iterative enhancement
"""

from pathlib import Path
import sys

from dash import html, dcc
import dash_bootstrap_components as dbc

# Add BDH root to path
BDH_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(BDH_ROOT))


def create_neural_microscope_interface(experiment_path: str, run_name: str) -> html.Div:
    """
    Create embedded Neural Microscope UI - Phase 2A minimal version.

    Args:
        experiment_path: Path to experiment directory (for context)
        run_name: Name of the run (for context)

    Returns:
        Minimal functional Neural Microscope interface

    Phase 2A Features:
    - Model status (from experiment-model-store)
    - Text input + inference button
    - 3D activation plot
    - Shared model state with Inference tab
    """

    # Get GPU allocation info
    try:
        from src.utils.neural_microscope_config import GPUAllocator
        allocator = GPUAllocator()
        inference_device = allocator.get_inference_device()
        rendering_device = allocator.get_rendering_device() or "CPU OpenGL"

        # Get GPU memory info if available
        gpu_info = []
        if inference_device.startswith("cuda:"):
            gpu_id = int(inference_device.split(":")[1])
            if gpu_id in allocator.available_gpus:
                props = allocator.available_gpus[gpu_id]
                gpu_info.append(f"Inference: GPU{gpu_id} ({props['name']}) - {props['memory_free']:.1f}/{props['memory_total']:.1f} GB free")

        if isinstance(rendering_device, str) and rendering_device.startswith("cuda:"):
            gpu_id = int(rendering_device.split(":")[1])
            if gpu_id in allocator.available_gpus:
                props = allocator.available_gpus[gpu_id]
                gpu_info.append(f"Rendering: GPU{gpu_id} ({props['name']}) - {props['memory_free']:.1f}/{props['memory_total']:.1f} GB free")
        elif rendering_device == "CPU OpenGL":
            gpu_info.append(f"Rendering: CPU OpenGL")

        gpu_status_card = dbc.Card([
            dbc.CardBody([
                html.H6([
                    html.I(className="bi bi-gpu-card me-2"),
                    "GPU Allocation"
                ], className="mb-2"),
                html.Ul([html.Li(info) for info in gpu_info] if gpu_info else [html.Li("No GPU info available")])
            ])
        ], className="mb-3")
    except Exception as e:
        gpu_status_card = dbc.Alert(f"Could not load GPU allocation info: {e}", color="warning", className="mb-3")

    return html.Div([
        # GPU Allocation Status
        dbc.Row([
            dbc.Col([gpu_status_card])
        ]),

        # Model Status (shows if loaded from Inference tab)
        dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.H5([
                        html.I(className="bi bi-microscope me-2"),
                        "Neural Microscope - Embedded Mode"
                    ], className="alert-heading"),
                    html.P([
                        "Load a model in the ",
                        html.Strong("Inference tab"),
                        " first, then return here to visualize neuron activations."
                    ], className="mb-0")
                ], color="info", id='microscope-initial-info'),

                html.Div(id='microscope-model-status', className="mb-3")
            ])
        ]),

        # Inference Controls
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Input Configuration"),
                    dbc.CardBody([
                        dbc.Label("Text Input:"),
                        dbc.Input(
                            id='microscope-text-input',
                            placeholder="Enter text to analyze (e.g., 'apple')...",
                            value="apple",
                            className="mb-3"
                        ),

                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Activation Threshold:"),
                                dcc.Slider(
                                    id='microscope-activation-threshold',
                                    min=0.001,
                                    max=0.1,
                                    step=0.001,
                                    value=0.01,
                                    marks={0.001: '0.001', 0.01: '0.01', 0.05: '0.05', 0.1: '0.1'},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                )
                            ], md=8),
                            dbc.Col([
                                dbc.Label("Hebbian Learning:"),
                                dbc.Switch(
                                    id='microscope-hebbian-toggle',
                                    label="Enable",
                                    value=False
                                )
                            ], md=4)
                        ], className="mb-3"),

                        dbc.Button(
                            [html.I(className="bi bi-play-fill me-2"), "Run Inference & Visualize"],
                            id='microscope-run-btn',
                            color="primary",
                            size="lg",
                            className="w-100"
                        ),

                        html.Div(id='microscope-run-status', className="mt-2")
                    ])
                ], className="mb-3")
            ])
        ]),

        # Visualization Controls
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Visualization Filters"),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Activation Type:"),
                                dcc.Dropdown(
                                    id='microscope-activation-type',
                                    options=[
                                        {'label': 'X Sparse (Post-Encoder)', 'value': 'x_sparse'},
                                        {'label': 'Y Sparse (Encoder V)', 'value': 'y_sparse'},
                                        {'label': 'XY Sparse (Hebbian Product)', 'value': 'xy_sparse'}
                                    ],
                                    value='x_sparse',
                                    clearable=False
                                )
                            ], md=4),
                            dbc.Col([
                                dbc.Label("Iteration:"),
                                dcc.Dropdown(
                                    id='microscope-iteration-selector',
                                    options=[{'label': 'All Iterations', 'value': 'all'}] +
                                            [{'label': f'Iteration {i}', 'value': str(i)} for i in range(6)],
                                    value='all',
                                    clearable=False
                                )
                            ], md=4),
                            dbc.Col([
                                dbc.Label("Character Position:"),
                                dcc.Dropdown(
                                    id='microscope-char-selector',
                                    options=[{'label': 'All Characters', 'value': 'all'}],
                                    value='all',
                                    clearable=False
                                )
                            ], md=4)
                        ], className="mb-3"),

                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Top-K Filter (show strongest activations):"),
                                dcc.Slider(
                                    id='microscope-topk-slider',
                                    min=0.1,
                                    max=100,
                                    step=0.1,
                                    value=20,
                                    marks={1: '1%', 10: '10%', 20: '20%', 50: '50%', 100: '100%'},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),
                                html.Small("Lower values = fewer, stronger activations shown", className="text-muted")
                            ], md=12)
                        ], className="mb-3"),

                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Position Mode (Manifold Hypothesis):"),
                                dcc.RadioItems(
                                    id='microscope-position-mode',
                                    options=[
                                        {'label': ' Weight Space (learned features - static)', 'value': 'weight'},
                                        {'label': ' Activation Space (representation - dynamic, validated)', 'value': 'activation'},
                                        {'label': ' Grid (legacy fallback)', 'value': 'grid'}
                                    ],
                                    value='weight',
                                    className="mb-2"
                                ),
                                html.Small([
                                    html.Strong("Weight:"), " PCA of attention weights (architecture). ",
                                    html.Strong("Activation:"), " UMAP of activation patterns (input-specific, validated). ",
                                    html.Strong("Grid:"), " Simple grid layout."
                                ], className="text-muted")
                            ], md=12)
                        ])
                    ])
                ], className="mb-3")
            ])
        ]),

        # 3D Visualization
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("3D Neuron Activation Visualization"),
                    dbc.CardBody([
                        dcc.Graph(
                            id='microscope-3d-plot',
                            config={'displayModeBar': True, 'scrollZoom': True},
                            style={'height': '700px'}
                        )
                    ])
                ])
            ])
        ]),

        # Info/Help
        dbc.Row([
            dbc.Col([
                dbc.Alert([
                    html.H6("What You're Seeing:"),
                    html.Ul([
                        html.Li([
                            html.Strong("Each dot"), " = One activated neuron during inference"
                        ]),
                        html.Li([
                            html.Strong("X/Y position"), " = Depends on Position Mode (Weight/Activation/Grid)"
                        ]),
                        html.Li([
                            html.Strong("Z position"), " = Iteration × 4 + Head (vertical stacking)"
                        ]),
                        html.Li([
                            html.Strong("Color"), " = Activation strength (brighter = stronger)"
                        ]),
                        html.Li([
                            html.Strong("Position Modes"), ":"
                        ]),
                        html.Ul([
                            html.Li([
                                html.Em("Weight Space"), " = PCA of learned weights (static architecture)"
                            ]),
                            html.Li([
                                html.Em("Activation Space"), " = UMAP of activation patterns (input-specific, validated)"
                            ]),
                            html.Li([
                                html.Em("Grid"), " = Simple grid layout (fallback)"
                            ])
                        ], className="small")
                    ], className="mb-2"),
                    html.P([
                        html.Strong("✅ Phase 2B COMPLETE: "),
                        "Filtering controls, Top-K slider, position modes with instant switching. ",
                        html.Strong("Next: "),
                        "Validation framework integration (Phase 2B+) or statistics panel (Phase 2C)."
                    ], className="mb-0 text-muted")
                ], color="light", className="mt-3")
            ])
        ]),

        # Data stores
        dcc.Store(id='microscope-activations-store'),
        dcc.Store(id='microscope-validation-store'),
    ])
