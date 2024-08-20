import base64
import re
from datetime import datetime

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
days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
color_theme = [px.colors.qualitative.Plotly[i] for i in range(6)]
hhmm_list = pd.date_range('00:00', '23:59', freq='1min').time


def day_of_week(i):
    return days_of_week[i]


def extract_emojis(s):
    return ''.join(c for c in s if c in emoji.UNICODE_EMOJI)


app = dash.Dash()

app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '50%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin-bottom': '5px'
        }
    ),
    html.Div(id='date-selection', children=[
        html.Div(
            [dcc.Dropdown(
                id='start-date'
            )],
            style=dict(width='25%', display='inline-block')
        ),
        html.Div(
            [dcc.Dropdown(
                id='end-date'
            )],
            style=dict(width='25%', display='inline-block')
        ),
        html.Div([
            html.Button('Submit', id='submit-val', n_clicks=0)
        ])
    ]),
    html.Div(id='graphs')
])


@app.callback(Output('date-selection', 'children'),
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

    global df
    df = df_temp

    global date_range
    date_range = [str(x)[:10] for x in df.Date.unique()]

    children = [
        html.Div(
            [dcc.Dropdown(
                id='start-date',
                options=[dict(label=i, value=i) for i in date_range],
                value=date_range[0]
            )],
            style=dict(width='25%', display='inline-block')
        ),
        html.Div(
            [dcc.Dropdown(
                id='end-date',
                options=[dict(label=i, value=i) for i in date_range],
                value=date_range[-1]
            )],
            style=dict(width='25%', display='inline-block')
        ),
        html.Div([
            html.Button('Submit', id='submit-val', n_clicks=0)
        ])
    ]

    return children


@app.callback(
    Output('graphs', 'children'),
    [Input('submit-val', 'n_clicks')],
    [State('start-date', 'value'), State('end-date', 'value')])
def update_graphs(n_clicks, start, end):
    df_temp = df[(datetime.strptime(start, "%Y-%m-%d") <= df.Date) & (df.Date <= (datetime.strptime(end, "%Y-%m-%d")))]

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

    children = [html.Div([graph], style=dict(width='48%', display='inline-block')) for graph in dcc_graphs]

    if n_clicks > 0:
        return children
    else:
        return None


if __name__ == '__main__':
    app.run_server()
