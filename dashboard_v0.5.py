import base64
import re
from collections import Counter
from datetime import datetime
from statistics import mean, median, stdev

import dash
import dash_core_components as dcc
import dash_html_components as html
import emoji
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State

days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
color_theme = [px.colors.qualitative.Plotly[i] for i in range(10)]
hhmm_list = pd.date_range('00:00', '23:59', freq='1min').time


def day_of_week(i):
    return days_of_week[i]


def extract_emojis(s):
    return ''.join(c for c in s if c in emoji.UNICODE_EMOJI)


app = dash.Dash(
    __name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css?family=Product+Sans:400,400i,700,700i",
        "https://cdn.rawgit.com/plotly/dash-app-stylesheets/2cc54b8c03f4126569a3440aae611bbef1d7a5dd/stylesheet.css",
        "https://codepen.io/bcd/pen/KQrXdb.css"
    ]
)

app.layout = html.Div([
    html.Div([
        html.Div(
            children=[
                html.Div(dcc.Upload(
                    id='upload-data',
                    children=html.Div([
                        'Drag and Drop or ',
                        html.A('Select File (.txt)')
                    ]),
                    style={
                        'width': '100%',
                        'height': '40',
                        'lineHeight': '60px',
                        'borderWidth': '1px',
                        'borderStyle': 'dashed',
                        'borderRadius': '5px',
                        'textAlign': 'center',
                        'margin-bottom': '5px',
                        'font-size': '14px'
                    }
                )),
            ]
        ),
        html.Div(id='intermediate-values', style={'display': 'none'}),
        html.Div(id='filter-selection', children=[
            html.Div(
                children=[dcc.Dropdown(
                    id='start-date'
                )],
                style={'margin-bottom': '5px', 'font-size': '14px', 'display': 'none'}
            ),
            html.Div(
                children=[dcc.Dropdown(
                    id='end-date'
                )],
                style={'margin-bottom': '5px', 'font-size': '14px', 'display': 'none'}
            ),
            html.Div(
                children=[dcc.Checklist(
                    id='user-selection'
                )],
                style={'display': 'none'}
            ),
            html.Div(
                children=[html.Button('Submit', id='submit-val', n_clicks=0)],
                style={'display': 'none'}
            ),
        ])
    ], style={'width': '14%', 'display': 'inline-block', 'vertical-align': 'top', 'margin': '5px'}),
    html.Div([
        html.Div(id='stats'),
        html.Div(id='graphs')
    ], style={'width': '84%', 'display': 'inline-block', 'margin': '5px'})
])


@app.callback(Output('intermediate-values', 'children'),
              [Input('upload-data', 'contents')])
def parse_data(contents):
    if contents is not None:
        # Decode the contents from base64 and convert it to a string
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string).decode('utf-8').split('\n')
        
        # Split into list of messages.
        lines = [x for x in decoded if x.startswith('[')]
        
    print('Messages read successfully.')
    
    parsed_lines = [
        [x[1:x.find(",")], # up to first comma, for date
         x[x.find(",")+2:x.find("]")], # from first comma to first ], for time
         x[x.find("]")+1:][:x[x.find("]"):].find(':')-1], # from first ] to next : for name
         x[x.find("]")+1:][x[x.find("]")+1:].find(':')+2:-1] # message
        ] for x in lines]

    df = pd.DataFrame(parsed_lines, columns=['Date', 'Time', 'User', 'Message'])
    df.Date = pd.to_datetime(df.Date, format='%d/%m/%y')
    df['MMYYYY'] = df.Date.apply(lambda x: x.strftime("%m/%Y"))
    df.MMYYYY = pd.to_datetime(df.MMYYYY, format='%m/%Y')
    df.Time = pd.to_datetime(df.Time, format='%H:%M:%S').dt.time
    df['Hour'] = pd.to_datetime(df.Time, format='%H:%M:%S').dt.hour
    df['Day'] = df.Date.dt.weekday
    df.Day = df.Day.apply(day_of_week)
    df.Day = df.Day.astype('category')
    df.Day.cat.reorder_categories(days_of_week, ordered=True, inplace=True)
    df['Emojis'] = df.Message.apply(extract_emojis)
    
    print('Dataframe created and data parsed.')

    # Uncomment line below to exclude all media messages
    # df = df[df.Message != '<Media omitted>']

    # Uncomment lines below to anonymize chats
    # users = sorted(df.User.unique())
    # users_map = dict(zip(users, range(1, len(users) + 1)))
    # df.User = df.User.apply(lambda x: 'User ' + str(users_map[x]))

    return df.to_json(date_format='iso', orient='split')


@app.callback(Output('filter-selection', 'children'),
              [Input('intermediate-values', 'children')])
def generate_filters(intermediate_values):
    df = pd.read_json(intermediate_values, orient='split')
    date_range = [str(x)[:10] for x in df.Date.unique()]
    users = sorted(df.User.unique())

    children = [
        html.Div(
            children=[dcc.Dropdown(
                id='start-date',
                options=[dict(label=i, value=i) for i in date_range],
                value=date_range[0]
            )],
            style={'margin-bottom': '5px', 'font-size': '14px'}
        ),
        html.Div(
            children=[dcc.Dropdown(
                id='end-date',
                options=[dict(label=i, value=i) for i in date_range],
                value=date_range[-1]
            )],
            style={'margin-bottom': '5px', 'font-size': '14px'}
        ),
        html.Div(
            children=[dcc.Checklist(
                id='user-selection',
                options=[{'label': user, 'value': user} for user in users],
                labelStyle={'display': 'block'},
                value=[user for user in users]
            )],
            style={'margin-bottom': '5px', 'font-size': '14px'}
        ),
        html.Div(children=[html.Button('Submit', id='submit-val', n_clicks=0)])
    ]
    return children


@app.callback(
    [Output('stats', 'children'),
     Output('graphs', 'children')],
    [Input('submit-val', 'n_clicks')],
    [State('intermediate-values', 'children'),
     State('start-date', 'value'),
     State('end-date', 'value'),
     State('user-selection', 'value')]
)
def update_graphs(n_clicks, intermediate_values, start, end, selected_users):
    df = pd.read_json(intermediate_values, orient='split')
    df.Date = df.Date.dt.tz_localize(None)
    df.Day = df.Day.astype('category')
    df.Day.cat.reorder_categories(days_of_week, ordered=True, inplace=True)
    df = df[
        (datetime.strptime(start, "%Y-%m-%d") <= df.Date)
        & (df.Date <= (datetime.strptime(end, "%Y-%m-%d")))
        & (df.User.isin(selected_users))
        ]

    output_stats = []

    for user in selected_users:
        df_user = df[df.User == user]
        message_lengths = df_user.Message.apply(lambda x: len(x.split(' ')))
        all_emojis = [x for y in df_user.Emojis.apply(lambda z: list(z)) for x in y]
        emoji_counts = Counter(all_emojis)
        output_stats.append(html.Div([html.P([
            f'User: {user}', html.Br(),
            f'Messages sent: {df_user.shape[0]}', html.Br(),
            f'Mean words per message: {mean(message_lengths)}', html.Br(),
            f'Median words per message: {int(median(message_lengths))}', html.Br(),
            f'Standard deviation: {stdev(message_lengths)}', html.Br(),
            f'Max message length: {max(message_lengths)}', html.Br(),
            f'Most used emojis: {" ".join(k for k, v in sorted(emoji_counts.items(), key=lambda item: item[1], reverse=True)[:5])}'
        ], style={'font-size': '14px', 'backgroundColor': 'white', 'padding': '10px'})],
            style={'display': 'inline-block', 'margin-right': '10px'}))

    df_heatmap = df \
        .groupby(['Day', 'Hour']) \
        .Message \
        .count() \
        .reset_index() \
        .sort_values(['Hour', 'Day']) \
        .reset_index(drop=True)

    dcc_graphs = [
        dcc.Graph(
            figure=px.histogram(df, x='MMYYYY', color='User',
                                color_discrete_sequence=color_theme).update_layout(bargap=0.1)
        ),
        dcc.Graph(
            figure=px.histogram(df, x='Hour', color='User', color_discrete_sequence=color_theme).update_layout(
                xaxis={'categoryorder': 'array', 'categoryarray': list(range(24))})
        ),
        dcc.Graph(
            figure=px.histogram(df, x='Day', color='User', color_discrete_sequence=color_theme).update_layout(
                bargap=0.1, xaxis={'categoryorder': 'array', 'categoryarray': days_of_week})
        ),
        dcc.Graph(
            figure=dict(data=[
                go.Heatmap(
                    x=df_heatmap.Hour,
                    y=df_heatmap.Day,
                    z=df_heatmap.Message
                )
            ])
        )
    ]

    output_graphs = [html.Div([graph], style={'width': '50%', 'display': 'inline-block', 'margin-bottom': '10px'}) for
                     graph in dcc_graphs]

    if n_clicks > 0:
        return output_stats, output_graphs
    else:
        return None


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=False)
