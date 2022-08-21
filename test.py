import math
from typing import Iterable
import dash
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px

from modules.dataio import load_data
from modules.asr import compute_asr_metrics, compute_global_statistics
from components.layouts import update_global_statistics

item_data, ext_vocab, metrics_available = load_data('output_test_manifest.json')
item_df = compute_asr_metrics(item_data, metrics_available, estimate_audio=False)

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

app.layout = html.Div([
    dbc.Container(id='page-content'),
    dcc.RangeSlider(
        min = 0, 
        max = math.ceil(item_df['duration'].max()), 
        step = 1, 
        value = [0, math.ceil(item_df['duration'].max())],
        id = 'duration-slider'
        ),
])

@app.callback(
    Output('page-content', 'children'),
    Input('duration-slider', 'value'))
def update_div(duration_range):
    filtered_df = item_df[(item_df.duration > duration_range[0]) & (item_df.duration < duration_range[1])]
    global_stats, vocab_data, alphabet = compute_global_statistics(filtered_df, ext_vocab, metrics_available)
    
    return update_global_statistics(global_stats, vocab_data, alphabet, metrics_available)


if __name__ == '__main__':
    app.run_server(host='0.0.0.0')
