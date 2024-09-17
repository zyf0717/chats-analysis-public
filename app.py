"""
This module defines a Shiny application for uploading and analyzing text and JSON files. 

The application uses various libraries for data manipulation, statistical analysis, and visualization, including:
- collections.Counter for counting elements
- statistics for basic statistical functions
- numpy and pandas for data manipulation
- plotly for data visualization
- shiny for creating the web application interface and server logic

The module imports several custom functions from the `funcs` module for data parsing and manipulation, and icons from the `icons` module.

The main components of the application are:
- `app_ui`: Defines the user interface of the application, including file upload input and output display areas.
- `server`: Defines the server logic of the application, including reactive calculations and data processing.

Functions:
- `parseFile`: A reactive function that processes the uploaded file and returns a DataFrame along with additional data.

Usage:
- Run the application by creating an instance of the `App` class with `app_ui` and `server` as arguments.
"""

from collections import Counter
from statistics import StatisticsError, mean, median, stdev

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo
from shinywidgets import output_widget, render_widget

from funcs import (
    parse_data, add_message_group, add_convo_id, add_response_time, 
    parse_day_column, transpose_stats_df, timedelta_to_str, days_of_week, hhmm_list, hh_list
)
from icons import question_circle_fill


##### APP SECTION #####

app_ui = ui.page_fluid(
    ui.page_sidebar(
        ui.sidebar(
            ui.input_file(
                "file1", 
                "Upload file (.txt, .json):", 
                accept=[".txt", ".json"], 
                multiple=False
            ),
            ui.output_ui("addFilters"),
        ),
        ui.output_ui("addCards"),
    )
)


def server(input: Inputs, output: Outputs, session: Session):
    @reactive.calc
    def parseFile():
        file: list[FileInfo] | None = input.file1()
        if not file:
            return pd.DataFrame(), [], []

        else:
            file_path = file[0]['datapath']
            filename = file[0]['name']
            df = parse_data(file_path, filename)
            date_range = [date.isoformat() for date in df['Date'].dt.date.unique()]
            users = sorted(df['User'].unique())
            return df, date_range, users


    @render.ui
    @reactive.event(input.file1)
    def addFilters():
        _, date_range, users = parseFile()
        users = {user: ui.span(user) for user in users}
        return ui.TagList(
            ui.hr(),
            ui.input_date_range(
                "daterange",
                "Select date-range:",
                start=min(date_range),
                end=max(date_range),
                min=min(date_range),
                max=max(date_range),
                format="dd/mm/yy",
                startview="decade",
                separator="to",
            ),
            ui.input_checkbox_group(
                "users",
                "Select user(s):",
                users,
                selected=list(users.keys()),
            ),
            ui.input_select(
                id="plot_theme", 
                label="Select theme:", 
                choices=["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "none"],
                selected="seaborn"
            ),
            ui.hr(),
            ui.input_action_button("generate", "Generate")
        )


    @reactive.calc
    @reactive.event(input.generate)
    def parseDataFrame():
        df, date_range, _ = parseFile()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df = df[
            (input.daterange()[0] <= df['Date'].dt.date)
            & (df['Date'].dt.date <= input.daterange()[1])
            & (df['User'].isin(input.users()))
            ].sort_values('Date').reset_index(drop=True)
        df = add_message_group(df)
        df = add_convo_id(df)
        df = add_response_time(df)
        df['Day'] = parse_day_column(df['Day'])
        df['Message_Length'] = df['Message'].str.split().str.len()
        return df, date_range, sorted(df['User'].unique())  # New list of users after filtering


    @render.ui
    @reactive.event(input.generate)
    def addCards():
        df, _, _ = parseDataFrame()
        month_list = sorted(df['MMYYYY'].unique(), reverse=True)
        month_list = list(map(lambda x: x.strftime("%b %Y"), month_list))
        return ui.layout_columns(
            ui.card(
                ui.card_header(            
                    ui.tooltip(
                        ui.span("User Stats ", question_circle_fill),
                        "Descriptive statistics of text messages per user.",
                        placement="right",
                        id="user_stats_tooltip",
                    ),
                ),
                ui.layout_columns(
                    ui.output_data_frame("userStats"),
                    output_widget("userStatsPlot"),
                    col_widths=[12, 12]
                ),
            ),
            ui.card(
                ui.card_header(            
                    ui.tooltip(
                        ui.span("Responsiveness ", question_circle_fill),
                        """
                        Non-convo: responsiveness to another user not as part of a convo.
                        In-convo: responsiveness to preceding message as part of a convo.
                        """,
                        placement="right",
                        id="response_stats_tooltip",
                    ),
                ),
                ui.layout_columns(
                    ui.input_select(
                        id="response_metric", 
                        label=None, 
                        choices=["All", "Non-convo", "In-convo"], 
                        selected="All",
                    ),
                    ui.output_data_frame("responseStats"),
                    output_widget("responsePlot"),
                    col_widths=[6, 12, 12],
                ),
            ),
            ui.card(
                ui.card_header(            
                    ui.tooltip(
                        ui.span("Convo Stats ", question_circle_fill),
                        "Convo: a consecutive series of messages between at least 2 users, with no consecutive timedelta exceeding 10 min.",
                        placement="right",
                        id="convo_stats_tooltip",
                    ),
                ),
                ui.layout_columns(
                    ui.input_select(
                        id="convo_metric", 
                        label=None, 
                        choices=["Duration", "Message count", "Word count"], 
                        selected="Duration",
                    ),
                    ui.output_data_frame("ConvoStats"),
                    output_widget("convoPlot"),
                    col_widths=[6, 12, 12]
                ),
            ),
            ui.card(
                ui.card_header("Message count"),
                ui.layout_columns(
                    ui.input_select(
                        id="plot1_metric", 
                        label=None, 
                        choices={
                            "MMYYYY": "Overall", 
                            "Day": "Day-of-week", 
                            "Month": {m: m for m in month_list},
                        },  
                        selected="MMYYYY"
                    ),
                    col_widths=[4]
                ),
                output_widget("plot1"),
                full_screen=True
            ),
            ui.card(
                ui.card_header("24-Hour Distribution"),
                ui.layout_columns(
                    ui.input_select(
                        id="plot2_metric", 
                        label=None, 
                        choices=["Hourly", "15-min"],
                        selected="15-min"
                    ),
                    col_widths=[4]
                ),
                output_widget("plot2"),
                full_screen=True
            ),
            ui.card(
                ui.card_header("Responsiveness"),
                ui.layout_columns(
                    ui.input_select(
                        id="plot3_metric1", 
                        label=None, 
                        choices=["All", "Non-convo", "In-convo"],
                        selected="All"
                    ), 
                    ui.input_select(
                        id="plot3_metric2", 
                        label=None, 
                        choices=["Mean", "Median"],  # To-do: Max? What is max? Max inactivity?
                        selected="Mean"
                    ),
                    col_widths=[4, 4,]
                ),

                output_widget("plot3"),
                full_screen=True
            ),
            ui.card(
                output_widget("plot4"),
                full_screen=True
            ),
            col_widths=[4, 4, 4, 6, 6, 6, 6]
        )


# To-do: media/images sent(?), reactions(?), message replies(?)


    @render.data_frame
    @reactive.event(input.generate)
    def userStats():
        df, _, users = parseDataFrame()
        user_stats = []
        for user in users:
            df_user = df[df['User'] == user]

            message_count = df_user.shape[0]
            message_lengths = df_user['Message'].str.split().str.len()
            total_words = sum(message_lengths)
            all_emojis = list(df_user['Emojis'].str.cat())
            emoji_counts = Counter(all_emojis)
            mean_words = round(mean(message_lengths), 2)
            median_words = int(median(message_lengths))
            try:
                standard_dev = round(stdev(message_lengths), 2)
            except StatisticsError:
                standard_dev = None
            max_length = max(message_lengths)
            top_emojis = " ".join(k for k, v in sorted(emoji_counts.items(), key=lambda item: item[1], reverse=True)[:5])

            user_stats.append([
                user,
                message_count,
                total_words,
                mean_words,
                median_words,
                max_length,
                standard_dev,
                top_emojis
            ])

        user_stats_df = pd.DataFrame(
            user_stats, 
            columns=[
                'User', 
                'Messages', 
                'Total words', 
                'Mean words', 
                'Median words', 
                'Max length', 
                'Standard deviation', 
                'Top emojis'
            ]
        )

        user_stats_df = transpose_stats_df(user_stats_df)

        return render.DataGrid(
            user_stats_df,
            width = "100%"
        )


    @render_widget
    @reactive.event(input.generate)
    def userStatsPlot():
        df, _, _ = parseDataFrame()
        df = df.sort_values('User').reset_index(drop=True)
        df['Message_Length'] = df['Message'].str.split().str.len()  # Compute earlier?

        figure = px.histogram(df, x='Message_Length', color='User', template=input.plot_theme())

        figure.update_layout(
            barmode="group",
            bargap=0.1,
            xaxis=dict(range=[0.5, df['Message_Length'].quantile(0.95)]),
            xaxis_title="Message length",
            yaxis_title=None,
            legend_title_text=None,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1,
                xanchor="right",
                x=1
            ),
        )
        return figure


    @render.data_frame
    @reactive.event(input.generate, input.response_metric)
    def responseStats():
        df, _, users = parseDataFrame()
        df = df.copy()

        response_stats = []
        for user in users:
            df_user = df[df['User'] == user].reset_index(drop=True)

            if input.response_metric() == "All":
                df_response = df_user
            elif input.response_metric() == "Non-convo":
                df_response = df_user[
                    (df_user['Is_Response'] == True) & 
                    (df_user['Is_Convo_Response'] == False)
                    ]
            elif input.response_metric() == "In-convo":
                df_response = df_user[
                    (df_user['Is_Convo_Response'] == True)
                    ]

            if input.response_metric() in ["All", "Non-convo"]:
                rt_max_gap = df_user['Date'].diff().max()
                rt_mean = df_response['Response_Time'].mean()
                rt_median = df_response['Response_Time'].median()
                rt_sd = df_response['Response_Time'].std()
                # rt_min = df_response['Response_Time'].min()
                # rt_max = df_response['Response_Time'].max()
            elif input.response_metric() == "In-convo":
                rt_max_gap = df_response['Time_Delta'].max()
                rt_mean = df_response['Time_Delta'].mean()
                rt_median = df_response['Time_Delta'].median()
                rt_sd = df_response['Time_Delta'].std()
                # rt_min = df_response['Time_Delta'].min()
                # rt_max = df_response['Time_Delta'].max()


            response_stats.append([
                user,
                # convos_started,
                timedelta_to_str(rt_max_gap),
                timedelta_to_str(rt_mean),
                timedelta_to_str(rt_median),
                timedelta_to_str(rt_sd),
                # timedelta_to_str(rt_min),
                # timedelta_to_str(rt_max),
            ])

            response_stats_df = pd.DataFrame(
                response_stats,
                columns=[
                    'User',
                    'Max inactivity',
                    'Mean responsiveness', 
                    'Median responsiveness', 
                    'Standard deviation',
                    # 'Quickest response',
                    # 'Slowest response',
                ]
            )

        response_stats_df = transpose_stats_df(response_stats_df)

        return render.DataGrid(
            response_stats_df,
            width = "100%"
        )


    @render_widget  
    @reactive.event(input.response_metric, input.generate)
    def responsePlot():
        df, _, _ = parseDataFrame()
        df = df.sort_values('User').reset_index(drop=True)

        if input.response_metric() == "All":
            pass
        elif input.response_metric() == "Non-convo":
            df = df[
                (df['Is_Response'] == True) & 
                (df['Is_Convo_Response'] == False)
                ]
        elif input.response_metric() is "In-convo":
            df = df[
                (df['Is_Convo_Response'] == True)
                ]

        if input.response_metric() in ["All", "Non-convo"]:
            df = df[df['Response_Time'].notnull()]
            df['Response_Time'] = df['Response_Time'].dt.total_seconds()
            df = df[df['Response_Time'] >= 0]
            figure = px.histogram(df, x='Response_Time', color='User',
                template=input.plot_theme())
            figure.update_layout(
                xaxis=dict(range=[0, np.percentile(df['Response_Time'], 95)]),
                xaxis_title="Seconds"
            )
        elif input.response_metric() in ["In-convo"]:
            df = df[df['Time_Delta'].notnull()]
            df['Time_Delta'] = df['Time_Delta'].dt.total_seconds()
            df = df[df['Time_Delta'] >= 0]
            figure = px.histogram(df, x='Time_Delta', color='User',
                template=input.plot_theme())
            figure.update_layout(
                xaxis=dict(range=[0, np.percentile(df['Time_Delta'], 95)]),
                xaxis_title="Seconds"
            )

        figure.update_layout(
            barmode="group",
            bargap=0.1,
            yaxis_title=None,
            legend_title_text=None,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1,
                xanchor="right",
                x=1
            ),
        )
        return figure


    @render.data_frame
    @reactive.event(input.generate, input.convo_metric)
    def ConvoStats():
        df, _, _ = parseDataFrame()
        df = df[df['Convo_ID'] != 0].reset_index(drop=True)
        num_convos = df['Convo_ID'].nunique()

        if input.convo_metric() == "Duration":
            df_first = df.groupby(by='Convo_ID').head(1).reset_index()
            df_last = df.groupby(by='Convo_ID').tail(1).reset_index()
            df_duration = df_first
            df_duration['Duration'] = df_last['Date'] - df_first['Date']

            total_duration = timedelta_to_str(df_duration['Duration'].sum())
            mean_duration = timedelta_to_str(df_duration['Duration'].mean())
            median_duration = timedelta_to_str(df_duration['Duration'].median())
            min_duration = timedelta_to_str(df_duration['Duration'].min())
            max_duration = timedelta_to_str(df_duration['Duration'].max())
            sd_duration = timedelta_to_str(df_duration['Duration'].std())

            convo_stats = [[  # Note: list of list(s) (i.e. row(s)) for dataframe
                num_convos,
                total_duration,
                mean_duration,
                median_duration,
                min_duration,
                max_duration,
                sd_duration
            ]]

            convo_stats_df = pd.DataFrame(
                convo_stats, 
                columns=[
                    "Number of convos",
                    "Total duration", 
                    "Mean duration", 
                    "Median duration",
                    "Min duration",
                    "Max duration",
                    "Standard deviation",
                ]
            )

        elif input.convo_metric() == "Message count":
            df_grouped = df.groupby(by='Convo_ID')['Message'].count().reset_index()

            total_messages = df.shape[0]
            mean_messages = round(df_grouped['Message'].mean(), 2)
            median_messages = df_grouped['Message'].median()
            min_messages = df_grouped['Message'].min()
            max_messages = df_grouped['Message'].max()
            sd_messages = round(df_grouped['Message'].std(), 2)

            convo_stats = [[  # Note: list of list(s) (i.e. row(s)) for dataframe
                num_convos,
                total_messages,
                mean_messages,
                median_messages,
                min_messages,
                max_messages,
                sd_messages,
            ]]

            convo_stats_df = pd.DataFrame(
                convo_stats, 
                columns=[
                    "Number of convos",
                    "Total messages", 
                    "Mean (per convo)", 
                    "Median (per convo)",
                    "Min convo length",
                    "Max convo length",
                    "Standard deviation",
                ]
            )

        elif input.convo_metric() == "Word count":
            df['Message_Length'] = df['Message'].str.len()
            df_grouped = df.groupby(by='Convo_ID')['Message_Length'].sum().reset_index()

            total_messages = df_grouped['Message_Length'].sum()
            mean_length = round(df_grouped['Message_Length'].mean(), 2)
            median_length = df_grouped['Message_Length'].median()
            min_length = df_grouped['Message_Length'].min()
            max_length = df_grouped['Message_Length'].max()
            sd_length = round(df_grouped['Message_Length'].std(), 2)

            convo_stats = [[  # Note: list of list(s) (i.e. row(s)) for dataframe
                num_convos,
                total_messages,
                mean_length,
                median_length,
                min_length,
                max_length,
                sd_length,
            ]]

            convo_stats_df = pd.DataFrame(
                convo_stats, 
                columns=[
                    "Number of convos",
                    "Total word count", 
                    "Mean word count", 
                    "Median word count",
                    "Min word count",
                    "Max word count",
                    "Standard deviation",
                ]
            )

        convo_stats_df = convo_stats_df.T.reset_index()
        convo_stats_df.columns = ["Metric", "Value"]

        return render.DataGrid(
            convo_stats_df,
            width = "100%"
        )
    
    
    @render_widget  
    @reactive.event(input.convo_metric, input.generate)
    def convoPlot():
        df, _, _ = parseDataFrame()
        df = df[df['Convo_ID'] != 0].reset_index(drop=True)

        if input.convo_metric() == "Duration":
            df_first = df.groupby(by='Convo_ID').head(1).reset_index()
            df_last = df.groupby(by='Convo_ID').tail(1).reset_index()
            df_duration = df_first
            df_duration['Duration'] = df_last['Date'] - df_first['Date']
            df_duration['Duration'] = df_duration['Duration'].dt.total_seconds()
            figure = px.histogram(df_duration, x='Duration',
                template=input.plot_theme())
            figure.update_layout(
                xaxis_title="Seconds",
                xaxis=dict(range=[0, np.percentile(df_duration['Duration'], 95)]),
            )
        elif input.convo_metric() == "Message count":
            df_grouped = df.groupby(by='Convo_ID')['Message'].count().reset_index()
            figure = px.histogram(df_grouped, x='Message',
                template=input.plot_theme())
            figure.update_layout(
                xaxis_title="Number of messages",
                xaxis=dict(range=[0, np.percentile(df_grouped['Message'], 95)]),
            )
        elif input.convo_metric() == "Word count":
            df['Message_Length'] = df['Message'].str.len()
            df_grouped = df.groupby(by='Convo_ID')['Message_Length'].sum().reset_index()
            figure = px.histogram(df_grouped, x='Message_Length',
                template=input.plot_theme())
            figure.update_layout(
                xaxis_title="Word count",
                xaxis=dict(range=[0, np.percentile(df_grouped['Message_Length'], 95)]),
            )

        figure.update_layout(
            bargap=0.1,
            yaxis_title=None,
            legend_title_text=None,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1,
                xanchor="right",
                x=1
            ),
        )

        return figure


    @render_widget  
    @reactive.event(input.plot1_metric, input.generate)
    def plot1():
        df, _, _ = parseDataFrame()
        df = df.sort_values('User').reset_index(drop=True)

        if input.plot1_metric() in ['Day', 'MMYYYY']:
            figure = px.histogram(df, x=input.plot1_metric(), color='User',
                template=input.plot_theme())

            if input.plot1_metric() == 'Day':
                figure.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': days_of_week})

        else:
            df = df[df['MMYYYY'] == pd.to_datetime(input.plot1_metric(), format='%b %Y')]
            total_days = (df['Date'].max() - df['Date'].min()).days + 1
            figure = px.histogram(df, x='Date', color='User', nbins=total_days,
                template=input.plot_theme())

        figure.update_layout(
            bargap=0.1,
            xaxis_title=None,
            yaxis_title=None,
            legend_title_text=None,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1,
                xanchor="right",
                x=1
            ),
        )

        return figure


    @render_widget  
    @reactive.event(input.plot2_metric, input.generate)
    def plot2():
        df, _, _ = parseDataFrame()
        df = df.sort_values('User').reset_index(drop=True)

        if input.plot2_metric() == '15-min':
            df['Time'] = df['Time'].dt.strftime('%H:%M')
            figure = px.histogram(df, x='Time', color='User',
                template=input.plot_theme(),
                ).update_layout(xaxis={'categoryorder': 'array', 'categoryarray': hhmm_list})
        elif input.plot2_metric() == 'Hourly':
            df['Time'] = df['Time'].dt.strftime('%H')
            figure = px.histogram(df, x='Time', color='User',
                template=input.plot_theme(),
                ).update_layout(xaxis={'categoryorder': 'array', 'categoryarray': hh_list})
            
        figure.update_layout(
            bargap=0.1,
            xaxis_title=None,
            yaxis_title=None,
            legend_title_text=None,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1,
                xanchor="right",
                x=1
            ),
        )
        return figure


    @render_widget  
    @reactive.event(input.plot3_metric1, input.plot3_metric2, input.generate)
    def plot3():
        df, _, _ = parseDataFrame()
        df = df.sort_values('User').reset_index(drop=True)

        if input.plot3_metric1() == 'Non-convo':
            df_response_time = df[
                (df['Is_Response'] == True) &
                (df['Is_Convo_Response'] == False)
                ].groupby(['User', 'MMYYYY'])['Response_Time']
        elif input.plot3_metric1() == 'In-convo':
            df_response_time = df[
                df['Is_Convo_Response'] == True
                ].groupby(['User', 'MMYYYY'])['Time_Delta']
        elif input.plot3_metric1() == 'All':
            df_response_time = df[
                (df['Is_Response'] == True)
                ].groupby(['User', 'MMYYYY'])['Response_Time']

        if input.plot3_metric2() == 'Mean':
            df_response_time = df_response_time.mean().reset_index()
        elif input.plot3_metric2() == 'Median':
            df_response_time = df_response_time.median().reset_index()

        if input.plot3_metric1() in ['Non-convo', 'All']:
            df_response_time['Response_Time'] = df_response_time['Response_Time'].dt.total_seconds()/60/60
            figure = px.scatter(
                df_response_time,
                x="MMYYYY",
                y="Response_Time",
                trendline="lowess",
                color="User",
                template=input.plot_theme()
            ).update_layout(
                yaxis_title="Hours",
            )
        elif input.plot3_metric1() in ['In-convo']:
            df_response_time['Time_Delta'] = df_response_time['Time_Delta'].dt.total_seconds()

            figure = px.scatter(
                df_response_time, 
                x="MMYYYY", 
                y="Time_Delta",
                trendline="lowess",
                color="User",
                template=input.plot_theme()
            ).update_layout(
                yaxis_title="Seconds",
            )

        figure.update_layout(
            xaxis_title=None,
            legend_title_text=None,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1,
                    xanchor="right",
                    x=1
            ),
        )
        return figure


    @render_widget  
    @reactive.event(input.generate)
    def plot4():
        df, _, _ = parseDataFrame()

        df_heatmap = df.groupby(['Day', 'Hour'], observed=False)['Message'].count().reset_index().sort_values(['Hour', 'Day']).reset_index(drop=True)

        # Creating the figure
        figure = go.Figure(data=go.Heatmap(
            x=df_heatmap['Hour'],
            y=df_heatmap['Day'],
            z=df_heatmap['Message'],
        ))
        
        return figure


app = App(app_ui, server)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
