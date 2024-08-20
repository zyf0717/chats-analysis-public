import os
import re
from datetime import datetime

import dash
import dash_core_components as dcc
import dash_html_components as html
import emoji
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash.dependencies import Input, Output

files = sorted(os.listdir('./chats'))
chat_path = f'./chats/{files[4]}'

lines = []

with open(chat_path, encoding='utf-8') as file:
    while True:
        line = file.readline()
        if not line:
            break
        elif re.match(r'^[0-9]{2}\/[0-9]{2}\/[0-9]{4}', line):
            if line.find(
                    'Messages to this chat and calls are now secured with end-to-end encryption. Tap for more info.') != -1:
                continue
            if line[16:].find(':') == -1:
                continue
            lines.append(line)
        else:
            lines[-1] = lines[-1] + line

parsed_lines = [[x[:10], x[12:17], x[20:][:x[20:].find(':')], x[20:][x[20:].find(':') + 2:].strip('\n')] for x in lines]

days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def day_of_week(i):
    return days_of_week[i]


def extract_emojis(s):
    return ''.join(c for c in s if c in emoji.UNICODE_EMOJI)


df = pd.DataFrame(parsed_lines, columns=['Date', 'Time', 'User', 'Message'])
df['MMYYYY'] = df.Date.apply(lambda x: x[3:5] + '/' + x[6:])
df.MMYYYY = pd.to_datetime(df.MMYYYY, format='%m/%Y')
df.Date = pd.to_datetime(df.Date, format='%d/%m/%Y')
df.Time = pd.to_datetime(df.Time, format='%H:%M').dt.time
df['Hour'] = pd.to_datetime(df.Time, format='%H:%M:%S').dt.hour
df['Day'] = df.Date.dt.weekday
df.Day = df.Day.apply(day_of_week)
df.Day = df.Day.astype('category')
df.Day.cat.reorder_categories(
    days_of_week,
    ordered=True,
    inplace=True
)
df = df[df.Message != '<Media omitted>']
df['Emojis'] = df.Message.apply(extract_emojis)

# Anonymize users
# users = list(set(df.User))
# df.User = df.User.apply(lambda x: 'User 2' if x == 'Zheng Yifei' else 'User 1')

# Need to update list of users
users = sorted(list(set(df.User)))

app = dash.Dash()

date_range = [str(x)[:10] for x in df.Date.unique()]

color_theme = [px.colors.qualitative.Plotly[i] for i in [0, 5]]

hhmm_list = pd.date_range('00:00', '23:59', freq='1min').time

app.layout = html.Div([
    html.Div(
        [dcc.Dropdown(
            id='start-date',
            options=[dict(label=i, value=i) for i in date_range],
            value=date_range[0]
        )],
        style=dict(width='48%', display='inline-block')
    ),
    html.Div(
        [dcc.Dropdown(
            id='end-date',
            options=[dict(label=i, value=i) for i in date_range],
            value=date_range[-1]
        )],
        style=dict(width='48%', display='inline-block')
    ),
    html.Div([dcc.Graph(id='histogram-1')], style=dict(width='48%', display='inline-block')),
    html.Div([dcc.Graph(id='histogram-2')], style=dict(width='48%', display='inline-block')),
    html.Div([dcc.Graph(id='histogram-3')], style=dict(width='48%', display='inline-block')),
    html.Div([dcc.Graph(id='heatmap')], style=dict(width='48%', display='inline-block'))
])


@app.callback(Output('histogram-1', 'figure'),
              [Input('start-date', 'value'),
               Input('end-date', 'value')])
def update_histogram_1(start, end):
    df_hist_1 = df[
        (datetime.strptime(start, "%Y-%m-%d") <= df.Date) & (df.Date <= (datetime.strptime(end, "%Y-%m-%d")))]
    fig = px.histogram(df_hist_1, x='MMYYYY', color='User', color_discrete_sequence=color_theme)
    fig.update_layout(bargap=0.1)
    return fig


@app.callback(Output('histogram-2', 'figure'),
              [Input('start-date', 'value'),
               Input('end-date', 'value')])
def update_histogram_1(start, end):
    df_hist_2 = df[
        (datetime.strptime(start, "%Y-%m-%d") <= df.Date) & (df.Date <= (datetime.strptime(end, "%Y-%m-%d")))]
    fig = px.histogram(df_hist_2, x='Time', color='User', color_discrete_sequence=color_theme)
    fig.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': hhmm_list})
    return fig


@app.callback(Output('histogram-3', 'figure'),
              [Input('start-date', 'value'),
               Input('end-date', 'value')])
def update_histogram_1(start, end):
    df_hist_3 = df[
        (datetime.strptime(start, "%Y-%m-%d") <= df.Date) & (df.Date <= (datetime.strptime(end, "%Y-%m-%d")))]
    fig = px.histogram(df_hist_3, x='Day', color='User', color_discrete_sequence=color_theme)
    fig.update_layout(bargap=0.1, xaxis={'categoryorder': 'array', 'categoryarray': days_of_week})
    return fig


@app.callback(Output('heatmap', 'figure'),
              [Input('start-date', 'value'),
               Input('end-date', 'value')])
def update_heatmap(start, end):
    df_heatmap = df[
        (datetime.strptime(start, "%Y-%m-%d") <= df.Date) & (df.Date <= (datetime.strptime(end, "%Y-%m-%d")))] \
        .groupby(['Day', 'Hour']) \
        .Message \
        .count() \
        .reset_index() \
        .sort_values(['Hour', 'Day']) \
        .reset_index(drop=True)
    return dict(
        data=[
            go.Heatmap(
                x=df_heatmap.Hour,
                y=df_heatmap.Day,
                z=df_heatmap.Message
            )
        ],
        layout=go.Layout(title='Message Count')
    )


if __name__ == '__main__':
    app.run_server()
