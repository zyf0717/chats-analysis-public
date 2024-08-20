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

df = None
date_range = None
users = None
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
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
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
        ),
        html.Div(id='filter-selection', children=[
            html.Div(
                [dcc.Dropdown(
                    id='start-date'
                )],
                style={'margin-bottom': '5px', 'font-size': '14px'}
            ),
            html.Div(
                [dcc.Dropdown(
                    id='end-date'
                )],
                style={'margin-bottom': '5px', 'font-size': '14px'}
            ),
            html.Div(
                [dcc.Checklist(
                    id='user-selection'
                )]
            ),
            html.Div([
                html.Button('Submit', id='submit-val', n_clicks=0)
            ])
        ])
    ], style={'width': '14%', 'display': 'inline-block', 'vertical-align': 'top', 'margin': '5px'}),
    html.Div([
        html.Div(id='stats'),
        html.Div(id='graphs')
    ], style={'width': '84%', 'display': 'inline-block', 'margin': '5px'})
])


@app.callback(Output('filter-selection', 'children'),
              [Input('upload-data', 'contents')])
def update_data(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string).decode('utf-8').split('\n')

    lines = []
    for line in decoded:
        if re.match(r'^[0-9]{2}\/[0-9]{2}\/[0-9]{4}', line):
            if line.find(
                    'Messages to this chat and calls are now secured with end-to-end encryption. Tap for more info.') != -1:
                continue
            if line[16:].find(':') == -1:
                continue
            lines.append(line)
        else:
            lines[-1] = lines[-1] + line

    parsed_lines = [[x[:10], x[12:17], x[20:][:x[20:].find(':')], x[20:][x[20:].find(':') + 2:].strip('\n')] for x in
                    lines]

    df_temp = pd.DataFrame(parsed_lines, columns=['Date', 'Time', 'User', 'Message'])
    df_temp['MMYYYY'] = df_temp.Date.apply(lambda x: x[3:5] + '/' + x[6:])
    df_temp.MMYYYY = pd.to_datetime(df_temp.MMYYYY, format='%m/%Y')
    df_temp.Date = pd.to_datetime(df_temp.Date, format='%d/%m/%Y')
    df_temp.Time = pd.to_datetime(df_temp.Time, format='%H:%M').dt.time
    df_temp['Hour'] = pd.to_datetime(df_temp.Time, format='%H:%M:%S').dt.hour
    df_temp['Day'] = df_temp.Date.dt.weekday
    df_temp.Day = df_temp.Day.apply(day_of_week)
    df_temp.Day = df_temp.Day.astype('category')
    df_temp.Day.cat.reorder_categories(
        days_of_week,
        ordered=True,
        inplace=True
    )
    df_temp = df_temp[df_temp.Message != '<Media omitted>']
    df_temp['Emojis'] = df_temp.Message.apply(extract_emojis)

    # Uncomment the lines below to anonymize users (to User 1, User 2, etc.)
    # users_temp = sorted(df_temp.User.unique())
    # users_map = dict(zip(users_temp, range(1, len(users_temp) + 1)))
    # df_temp.User = df_temp.User.apply(lambda x: 'User ' + str(users_map[x]))

    global df
    df = df_temp

    global date_range
    date_range = [str(x)[:10] for x in df.Date.unique()]

    global users
    users = sorted(df.User.unique())

    children = [
        html.Div(
            [dcc.Dropdown(
                id='start-date',
                options=[dict(label=i, value=i) for i in date_range],
                value=date_range[0]
            )],
            style={'margin-bottom': '5px', 'font-size': '14px'}
        ),
        html.Div(
            [dcc.Dropdown(
                id='end-date',
                options=[dict(label=i, value=i) for i in date_range],
                value=date_range[-1]
            )],
            style={'margin-bottom': '5px', 'font-size': '14px'}
        ),
        html.Div(
            [dcc.Checklist(
                id='user-selection',
                options=[{'label': user, 'value': user} for user in users],
                labelStyle={'display': 'block'},
                value=[user for user in users]
            )],
            style={'margin-bottom': '5px', 'font-size': '14px'}
        ),
        html.Div([
            html.Button('Submit', id='submit-val', n_clicks=0)
        ])
    ]

    return children


@app.callback(
    [Output('stats', 'children'),
     Output('graphs', 'children')],
    [Input('submit-val', 'n_clicks')],
    [State('start-date', 'value'),
     State('end-date', 'value'),
     State('user-selection', 'value')]
)
def update_graphs(n_clicks, start, end, selected_users):
    df_temp = df[
        (datetime.strptime(start, "%Y-%m-%d") <= df.Date)
        & (df.Date <= (datetime.strptime(end, "%Y-%m-%d")))
        & (df.User.isin(selected_users))
        ]

    output_stats = []

    for user in selected_users:
        df_user = df[df.User == user]
        message_lengths = df_user.Message.apply(lambda x: len(x.split(' ')))
        all_emojis = [x for y in df_user.Emojis.apply(lambda z: list(z)) for x in y if x != []]
        emoji_counts = Counter(all_emojis)
        output_stats.append(html.Div([html.P([
            f'User: {user}', html.Br(),
            f'Messages sent: {df_user.shape[0]}', html.Br(),
            f'Mean words per message: {mean(message_lengths)}', html.Br(),
            f'Median words per message: {int(median(message_lengths))}', html.Br(),
            f'Standard deviation: {stdev(message_lengths)}', html.Br(),
            f'Max message length: {max(message_lengths)}', html.Br(),
            f'Most used emojis: {" ".join(k for k, v in emoji_counts.items() if v >= 10)}'
        ], style={'font-size': '14px', 'backgroundColor': 'white', 'padding': '10px'})],
            style={'display': 'inline-block', 'margin-right': '15px'}))

    df_heatmap = df_temp \
        .groupby(['Day', 'Hour']) \
        .Message \
        .count() \
        .reset_index() \
        .sort_values(['Hour', 'Day']) \
        .reset_index(drop=True)

    dcc_graphs = [
        dcc.Graph(
            figure=px.histogram(df_temp, x='MMYYYY', color='User',
                                color_discrete_sequence=color_theme).update_layout(bargap=0.1)
        ),
        dcc.Graph(
            figure=px.histogram(df_temp, x='Time', color='User', color_discrete_sequence=color_theme).update_layout(
                xaxis={'categoryorder': 'array', 'categoryarray': hhmm_list})
        ),
        dcc.Graph(
            figure=px.histogram(df_temp, x='Day', color='User', color_discrete_sequence=color_theme).update_layout(
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
    app.run_server()
