import math
from typing import Iterable, List
import dash
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px

from modules.dataio import load_data
from modules.asr import compute_asr_metrics, compute_global_statistics
from components.layouts import update_global_statistics

item_data, ext_vocab, metrics_available, metadata_keys = load_data('dev_manifest.json')
item_df = compute_asr_metrics(item_data, metrics_available, estimate_audio=False)

app = Dash(
    __name__,
    suppress_callback_exceptions=True,
)

def add_slider(title: str, _id: str, min: int, max: int, step: int, value: List[int]):
    return dbc.Container(    
            html.Div([
                dbc.Row(dbc.Col(html.Div(title, className='text-secondary'), width=3),),
                dbc.Row(dbc.Col(dcc.RangeSlider(min=min, max=max, step=step, value=value, id=_id))),
                ],)
            )

def add_checklist(title: str, _id: str, options: List[str], value: List[str]):
    return dbc.Container(    
            html.Div([
                dbc.Row([
                    dbc.Col(html.Div(title, className='text-secondary'), width=1),
                    dbc.Col(dcc.Checklist(options=options, value=value, id=_id)),
                    ]),
                ],)
            )

app.layout = html.Div([
    add_slider(
        'Clip Duration', _id='duration-slider',
         min=0, max=math.ceil(item_df['duration'].max()), 
         step=1, value=[0, math.ceil(item_df['duration'].max())]
         ),
    add_checklist(
        'Age', _id='age-checklist',
        options=list(item_df['age'].unique()), value=list(item_df['age'].unique())
        ),
    add_checklist(
        'Gender', _id='gender-checklist',
        options=list(item_df['gender'].unique()), value=list(item_df['gender'].unique())
        ),
    add_checklist(
        'Accent', _id='accent-checklist',
        options=list(item_df['accent'].unique()), value=list(item_df['accent'].unique())
        ),
    dbc.Container(id='page-content'),
    ] )

@app.callback(
    Output('page-content', 'children'),
    Input('duration-slider', 'value'),
    Input('gender-checklist', 'value'),
    Input('age-checklist', 'value'),
    Input('accent-checklist', 'value'),
    )
def update_div(duration_range, gender_list, age_list, accent_list):

    # duration conditions
    duration_conds = (item_df.duration > duration_range[0]) & (item_df.duration < duration_range[1])
    gender_conds = (item_df.gender.isin(gender_list))
    age_conds = (item_df.age.isin(age_list))
    accent_conds = (item_df.accent.isin(accent_list))
    # all_conds = duration_conds
    all_conds = duration_conds & gender_conds & age_conds & accent_conds

    filtered_df = item_df[all_conds]
    global_stats, vocab_data, alphabet = compute_global_statistics(filtered_df, ext_vocab, metrics_available)
    
    return update_global_statistics(global_stats, filtered_df, vocab_data, alphabet, metrics_available)

if __name__ == '__main__':
    app.run_server(host='0.0.0.0')
