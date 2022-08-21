from dash import dash_table, dcc, html
import dash_bootstrap_components as dbc
import pandas as pd

from typing import List, Tuple, Set

def update_global_statistics(global_stats: pd.DataFrame, vocab_data: List, alphabet: Set, metrics_available: bool = False) -> html.Div:
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
                html.H5(len(global_stats), className='text-center p-1', style={'color': 'green', 'opacity': 0.7}),
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
                dbc.Col(html.Div('Word Error Rate (WER), %', className='text-secondary'), width=3, class_name='border-end'),
                dbc.Col(html.Div('Character Error Rate (CER), %', className='text-secondary'),width=3,class_name='border-end'),
                dbc.Col(html.Div('Word Match Rate (WMR), %', className='text-secondary'), width=3, class_name='border-end'),
                dbc.Col(html.Div('Mean Word Accuracy, %', className='text-secondary'), width=3),
            ],
            class_name='bg-light mt-2 rounded-top border-top border-start border-end',
        )
        element_row2 = dbc.Row(
            [
                dbc.Col(
                    html.H5(
                        '{:.2f}'.format(global_stats['wer']), className='text-center p-1', style={'color': 'green', 'opacity': 0.7},
                    ),
                    width=3,
                    class_name='border-end',
                ),
                dbc.Col(
                    html.H5(
                        '{:.2f}'.format(global_stats['cer']), className='text-center p-1', style={'color': 'green', 'opacity': 0.7}
                    ),
                    width=3,
                    class_name='border-end',
                ),
                dbc.Col(
                    html.H5(
                        '{:.2f}'.format(global_stats['wmr']), className='text-center p-1', style={'color': 'green', 'opacity': 0.7},
                    ),
                    width=3,
                    class_name='border-end',
                ),
                dbc.Col(
                    html.H5(
                        '{:.2f}'.format(global_stats['mwa']), className='text-center p-1', style={'color': 'green', 'opacity': 0.7},
                    ),
                    width=3,
                ),
            ],
            class_name='bg-light rounded-bottom border-bottom border-start border-end',
        )

        layout.extend([header_row2, element_row2])

    return layout

# def update_alphabets(alphabet)