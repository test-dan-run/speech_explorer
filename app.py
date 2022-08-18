import os
import dash
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Speech Data Explorer')
    parser.add_argument(
        'manifest', help='path to JSON manifest file',
    )
    parser.add_argument('--vocab', help='optional vocabulary to highlight OOV words')
    parser.add_argument('--port', default='8050', help='serving port for establishing connection')
    parser.add_argument(
        '--disable-caching-metrics', action='store_true', help='disable caching metrics for errors analysis'
    )
    parser.add_argument(
        '--estimate-audio-metrics',
        '-a',
        action='store_true',
        help='estimate frequency bandwidth and signal level of audio recordings',
    )
    parser.add_argument('--debug', '-d', action='store_true', help='enable debug mode')
    args = parser.parse_args()

    print('Starting server...')
    app = dash.Dash(
        __name__,
        suppress_callback_exceptions=True,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title=os.path.basename(args.manifest),
    )

    app.run_server(host='0.0.0.0', port=args.port, debug=args.debug)