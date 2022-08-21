from plotly import express as px
from plotly import graph_objects as go

from typing import List, Tuple, Dict

# plot histogram of specified field in data list
def plot_histogram(data: List, key: str, label: str):
    fig = px.histogram(
        data_frame=[item[key] for item in data],
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