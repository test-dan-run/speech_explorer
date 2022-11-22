from dash import dash_table, dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
from plotly import express as px
from plotly import graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from typing import List, Tuple, Set

# plot histogram of specified field in data list
def plot_histogram(data: List, key: str, label: str):
    fig = px.histogram(
        data_frame=data[key].tolist(),
        nbins=50,
        log_y=True,
        labels={'value': label},
        opacity=0.5,
        color_discrete_sequence=['green'],
        height=200,
    )
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0, pad=0))
    return fig

def plot_word_accuracy(vocab_data: List):
    labels = ['Unrecognized', 'Sometimes recognized', 'Always recognized']
    counts = [0, 0, 0]
    for word in vocab_data:
        if word['accuracy'] == 0:
            counts[0] += 1
        elif word['accuracy'] < 100:
            counts[1] += 1
        else:
            counts[2] += 1
    colors = ['red', 'orange', 'green']

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=counts,
                marker_color=colors,
                text=['{:.2%}'.format(count / sum(counts)) for count in counts],
                textposition='auto',
            )
        ]
    )
    fig.update_layout(
        showlegend=False, margin=dict(l=0, r=0, t=0, b=0, pad=0), height=200, yaxis={'title_text': '#words'}
    )

    return fig

def plot_wordclouds(vocab_data):

    inacc_count_dict = {}
    part_count_dict = {}
    acc_count_dict = {}
    for item in vocab_data:
        if item['accuracy'] == 0:
            inacc_count_dict[item['word']] = item['count']
        elif item['accuracy'] < 100:
            part_count_dict[item['word']] = item['count']
        else:
            acc_count_dict[item['word']] = item['count']

    inacc_wc = WordCloud(background_color='white')
    inacc_wc.generate_from_frequencies(inacc_count_dict)

    part_wc = WordCloud(background_color='white')
    part_wc.generate_from_frequencies(part_count_dict)
    
    acc_wc = WordCloud(background_color='white')
    acc_wc.generate_from_frequencies(acc_count_dict)

    inacc_out_img = BytesIO()
    inacc_wc.to_image().save(inacc_out_img, format='PNG')

    part_out_img = BytesIO()
    part_wc.to_image().save(part_out_img, format='PNG')

    acc_out_img = BytesIO()
    acc_wc.to_image().save(acc_out_img, format='PNG')

    inacc_data = base64.b64encode(inacc_out_img.getbuffer()).decode('utf8') # encode to html elements
    part_data = base64.b64encode(part_out_img.getbuffer()).decode('utf8') # encode to html elements
    acc_data = base64.b64encode(acc_out_img.getbuffer()).decode('utf8') # encode to html elements

    return [f'data:image/png;base64,{inacc_data}', f'data:image/png;base64,{part_data}', f'data:image/png;base64,{acc_data}']


def draw_figures(data):
    figures_labels = {
        'duration': ['Duration', 'Duration, sec'],
        'num_words': ['Number of Words', '#words'],
        'num_chars': ['Number of Characters', '#chars'],
        'word_rate': ['Word Rate', '#words/sec'],
        'char_rate': ['Character Rate', '#chars/sec'],
        'WER': ['Word Error Rate', 'WER, %'],
        'CER': ['Character Error Rate', 'CER, %'],
        'WMR': ['Word Match Rate', 'WMR, %'],
        'I': ['# Insertions (I)', '#words'],
        'D': ['# Deletions (D)', '#words'],
        'D-I': ['# Deletions - # Insertions (D-I)', '#words'],
        # 'freq_bandwidth': ['Frequency Bandwidth', 'Bandwidth, Hz'],
        # 'level_db': ['Peak Level', 'Level, dB'],
    }
    figures_hist = {}
    for k in figures_labels:
        val = data.loc[0,k]
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            if k in figures_labels:
                ylabel = figures_labels[k][0]
                xlabel = figures_labels[k][1]
            else:
                title = k.replace('_', ' ')
                title = title[0].upper() + title[1:].lower()
                ylabel = title
                xlabel = title
            figures_hist[k] = [ylabel + ' (per utterance)', plot_histogram(data, k, xlabel)]

    graph_charts = []
    for k in figures_hist:
        graph_charts.extend([
            dbc.Row(dbc.Col(html.H5(figures_hist[k][0]), class_name='text-secondary'), class_name='mt-3'),
            dbc.Row(dbc.Col(dcc.Graph(id=f'{k.replace("_","-")}-graph', figure=figures_hist[k][1]),),),
        ])

    return graph_charts

def draw_word_acc_chart(vocab_data):
    figure_word_acc = plot_word_accuracy(vocab_data)
    return [
        dbc.Row(dbc.Col(html.H5('Word accuracy distribution'), class_name='text-secondary'), class_name='mt-3'),
        dbc.Row(dbc.Col(dcc.Graph(id='word-acc-graph', figure=figure_word_acc),),),
    ]    



def update_global_statistics(global_stats: pd.DataFrame, dataset: pd.DataFrame, vocab_data: List, alphabet: Set, metrics_available: bool = False) -> html.Div:
    title_row =  dbc.Row(dbc.Col(html.H5(children='Global Statistics'), class_name='text-secondary'), class_name='mt-3')
    header_row1 = dbc.Row(
        [
            dbc.Col(html.Div('Number of hours', className='text-secondary'), width=3, class_name='border-end'),
            dbc.Col(html.Div('Number of utterances', className='text-secondary'), width=3, class_name='border-end'),
            dbc.Col(html.Div('Vocabulary size', className='text-secondary'), width=3, class_name='border-end'),
            dbc.Col(html.Div('Alphabet size', className='text-secondary'), width=3),
        ],
        class_name='bg-light mt-2 rounded-top border-top border-start border-end',
    )
    element_row1 = dbc.Row(
        [
            # num hours
            dbc.Col(
                html.H5(
                    '{:.2f} hours'.format(global_stats['num_hours']),
                    className='text-center p-1',
                    style={'color': 'green', 'opacity': 0.7},
                ),
                width=3,
                class_name='border-end',
            ),
            # num stats
            dbc.Col(
                html.H5(len(dataset), className='text-center p-1', style={'color': 'green', 'opacity': 0.7}),
                width=3,
                class_name='border-end',
            ),
            # num of vocab
            dbc.Col(
                html.H5(
                    '{} words'.format(len(vocab_data)),
                    className='text-center p-1',
                    style={'color': 'green', 'opacity': 0.7},
                ),
                width=3,
                class_name='border-end',
            ),
            # unique characters
            dbc.Col(
                html.H5(
                    '{} chars'.format(len(alphabet)),
                    className='text-center p-1',
                    style={'color': 'green', 'opacity': 0.7},
                ),
                width=3,
            ),
        ],
        class_name='bg-light rounded-bottom border-bottom border-start border-end',
    )

    layout = [title_row, header_row1, element_row1]
    if metrics_available:
        header_row2 = dbc.Row(
            [
                dbc.Col(html.Div('Word Error Rate (WER), %', className='text-secondary'), class_name='border-end'),
                dbc.Col(html.Div('Character Error Rate (CER), %', className='text-secondary'), class_name='border-end'),
                dbc.Col(html.Div('Word Match Rate (WMR), %', className='text-secondary'), class_name='border-end'),
                dbc.Col(html.Div('SubWord Match Rate (SWMR-90%), %', className='text-secondary'), class_name='border-end'),
                dbc.Col(html.Div('SubWord Match Rate (SWMR-75%), %', className='text-secondary'), class_name='border-end'),
                dbc.Col(html.Div('Mean Vocab Accuracy, %', className='text-secondary')),
            ],
            class_name='bg-light mt-2 rounded-top border-top border-start border-end',
        )
        element_row2 = dbc.Row(
            [
                dbc.Col(
                    html.H5(
                        '{:.2f}'.format(global_stats['wer']), className='text-center p-1', style={'color': 'green', 'opacity': 0.7},
                    ),
                    # width=2,
                    class_name='border-end',
                ),
                dbc.Col(
                    html.H5(
                        '{:.2f}'.format(global_stats['cer']), className='text-center p-1', style={'color': 'green', 'opacity': 0.7}
                    ),
                    # width=2,
                    class_name='border-end',
                ),
                dbc.Col(
                    html.H5(
                        '{:.2f}'.format(global_stats['wmr']), className='text-center p-1', style={'color': 'green', 'opacity': 0.7},
                    ),
                    # width=2,
                    class_name='border-end',
                ),
                dbc.Col(
                    html.H5(
                        '{:.2f}'.format(global_stats['swmr90']), className='text-center p-1', style={'color': 'green', 'opacity': 0.7},
                    ),
                    # width=2,
                    class_name='border-end',
                ),
                dbc.Col(
                    html.H5(
                        '{:.2f}'.format(global_stats['swmr75']), className='text-center p-1', style={'color': 'green', 'opacity': 0.7},
                    ),
                    # width=2,
                    class_name='border-end',
                ),
                dbc.Col(
                    html.H5(
                        '{:.2f}'.format(global_stats['mwa']), className='text-center p-1', style={'color': 'green', 'opacity': 0.7},
                    ),
                    # width=2,
                ),
            ],
            class_name='bg-light rounded-bottom border-bottom border-start border-end',
        )

        layout.extend([header_row2, element_row2])
        layout.extend(draw_figures(dataset))
        layout.extend(draw_word_acc_chart(vocab_data))

    return layout

# def update_alphabets(alphabet)