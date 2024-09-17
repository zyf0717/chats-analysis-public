import numpy as np
import pandas as pd
import re
import json
import plotly.express as px
import plotly.graph_objs as go

from collections import Counter
from statistics import mean, median, stdev
from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shiny.types import FileInfo
from shinywidgets import output_widget, render_widget

from icons import question_circle_fill

days_of_week = np.array(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
hhmm_range = pd.date_range('00:00', '23:59', freq='15min').time
hhmm_list = [t.strftime('%H:%M') for t in hhmm_range]
hh_range = pd.date_range('00:00', '23:59', freq='h').time
hh_list = [t.strftime('%H') for t in hh_range]

emoji_pattern = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)


##### ETL FUNCTIONS #####

def extract_whatsapp_row(string: str) -> list:
    """
    Extracts a list of 4 elements from a given string representing a WhatsApp chat message.

    The elements are:
    - Date (str)
    - Time (str)
    - User (str)
    - Message (str)

    Args:
        string (str): The string to parse, representing a WhatsApp message.

    Returns:
        list: A list of 4 elements containing the extracted information.
    """
    string = string.replace("\u202f", " ")
    output = [None, None, None, None]

    # Extract date and time string
    match = re.search(r'.*?(?=[\]\-])', string)
    if match:
        date_time_string = match.group().strip("[]- ,")
        date = re.search(r'\d{1,2}/\d{1,2}/\d{2,4}', date_time_string)
        if date:
            output[0] = date.group().strip("[]- ,")

        time = re.search(r',\s.*', date_time_string)
        if time:
            output[1] = time.group().strip("[]- ,")

    user_and_message = re.search(r'([-\]]).*', string)
    if user_and_message:
    
        # Extract user
        match = re.search(r'^(.*?):',user_and_message.group())
        if match:
            user = match.group().strip("[]-: ")
            user = re.sub(r" */ *", " ", user)
            output[2] = user
        
        # Extract message
        match = re.search(r':.*', user_and_message.group())
        if match:
            output[3]= match.group().lstrip("[]-: ,").strip()
    
    return output


def get_date_format(dates):
    # Potential date formats
    formats = [
        "%d/%m/%y",  # 21/08/23
        "%d/%m/%Y",  # 21/08/2023
        "%m/%d/%y",  # 08/21/23
        "%m/%d/%Y",  # 08/21/2023
        "%Y/%m/%d",  # 2023/08/21
        "%Y-%m-%d",  # 2023-08-21
        "%d-%m-%Y",  # 21-08-2023
        "%m-%d-%Y",  # 08-21-2023
        "%d.%m.%Y",  # 21.08.2023
        "%m.%d.%Y",  # 08.21.2023
        "%Y.%m.%d",  # 2023.08.21
    ]

    matched_format = None
    for fmt in formats:
        parsed_dates = pd.to_datetime(dates, format=fmt, errors='coerce')
        if parsed_dates.notnull().all():
            matched_format = fmt
            print(f"Date format determined to be {fmt}")
            break
    return matched_format


def get_time_format(times):
    # Potential time formats
    formats = [
        "%H:%M:%S",     # 14:30:00
        "%H:%M",        # 14:30
        "%I:%M %p",     # 02:30 PM
        "%I:%M:%S %p",  # 02:30:00 PM
        "%H%M",         # 1430
        "%I%M %p"       # 0230 PM
    ]

    matched_format = None
    for fmt in formats:
        parsed_times = pd.to_datetime(times, format=fmt, errors='coerce')
        if parsed_times.notnull().all():
            matched_format = fmt
            print(f"Time format determined to be {fmt}")
            break
    return matched_format


def get_day_of_week(i: int) -> str:
    return days_of_week[i]


def parse_day_column(df_col):
    df_col = df_col.astype('category')
    existing_categories = [day for day in days_of_week if day in df_col.cat.categories]
    df_col = df_col.cat.reorder_categories(existing_categories, ordered=True)
    return df_col


def parse_whatsapp_data(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse WhatsApp data

    Args:
        input_df (pd.DataFrame): Input DataFrame with columns ['Date', 'Time', 'User', 'Message']

    Returns:
        pd.DataFrame: DataFrame with the following columns:
            ['Date', 'Time', 'MMYYYY', 'Week', 'Hour', 'Day', 'User', 'Message', 'Emojis']
    """
    df = input_df.copy()

    date_format = get_date_format(df['Date'])
    time_format = get_time_format(df['Time'])
    if not all([date_format, time_format]):
        print("Could not determine date and/or time formats. Using 'mixed' format...")
        df['Date'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True, format='mixed', errors='coerce')
    else:
        print("Parsing date and time formats with detected formats...")
        df['Date'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format=f"{date_format} {time_format}")
    df = df.dropna().reset_index(drop=True)

    df['MMYYYY'] = df['Date'].apply(lambda x: x.strftime("%m/%Y"))
    df['MMYYYY'] = pd.to_datetime(df['MMYYYY'], format='%m/%Y')
    df['Week'] = df['Date'].dt.isocalendar().week
    df['Hour'] = df['Date'].dt.hour
    df['Time'] = df['Date'].dt.floor('15min')
    df['Day'] = df['Date'].dt.weekday
    df['Day'] = df['Day'].apply(get_day_of_week)
    df['Emojis'] = df['Message'].str.findall(emoji_pattern).str.join('')
    print('Dataframe created and WhatsApp data parsed.')
    return df


def parse_telegram_data(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse Telegram data

    Args:
        input_df (pd.DataFrame): Input DataFrame with columns ['date_unixtime', 'text', 'from']

    Returns:
        pd.DataFrame: DataFrame with the following columns:
            ['Date', 'Time', 'MMYYYY', 'Week', 'Hour', 'Day', 'User', 'Message', 'Emojis']
    """
    df = input_df.copy()
    df['date_unixtime'] = df['date_unixtime'].apply(lambda x: int(x) + 8 * 60 * 60)  # For GMT+8
    df['date_unixtime'] = pd.to_datetime(df['date_unixtime'], unit='s')
    df['Date'] = df['date_unixtime']
    df['Time'] = df['date_unixtime'].dt.floor('15min')
    df['MMYYYY'] = df['Date'].dt.strftime("%m/%Y")
    df['Week'] = df['date_unixtime'].dt.isocalendar().week
    df['MMYYYY'] = pd.to_datetime(df['MMYYYY'], format='%m/%Y')
    df['Hour'] = df['date_unixtime'].dt.hour
    df['Day'] = df['date_unixtime'].dt.weekday
    df['Day'] = df['Day'].apply(get_day_of_week)
    df['Message'] = df['text'].astype(str)
    df = df[df['Message'].str.len() > 0]
    df['Emojis'] = df['Message'].str.findall(emoji_pattern).str.join('')
    df['User'] = df['from']
    df = df[['Date', 'Time', 'MMYYYY', 'Week', 'Hour', 'Day', 'User', 'Message', 'Emojis']]
    print('Dataframe created and Telegram data parsed.')
    return df


def parse_data(file_type: str, file_path: str) -> pd.DataFrame:
    """
    Parse a WhatsApp or Telegram chat file and return a DataFrame.

    Args:
        file_type (str): The type of file to parse. Must be either 'text' or 'json'.
        file_path (str): The path to the file to parse.

    Returns:
        pd.DataFrame: The parsed DataFrame with the following columns:
            ['Date', 'Time', 'MMYYYY', 'Week', 'Hour', 'Day', 'User', 'Message', 'Emojis']
    """
    df = None
    
    with open(file_path, encoding='utf-8') as file:
        decoded = file.readlines()
            
    if 'text' in file_type:
        print("WhatsApp chat detected.")
        
        # Split into list of messages and keep only relevant ones.
        parsed_lines = [extract_whatsapp_row(x) for x in decoded]
        parsed_lines = [row for row in parsed_lines if all(row)]  # Only filled rows
        df = pd.DataFrame(parsed_lines, columns=['Date', 'Time', 'User', 'Message'])
        df = parse_whatsapp_data(df)
        
    if 'json' in file_type:
        print("Telegram chat detected.")

        try:
            decoded = ''.join(decoded)
            json_data = json.loads(decoded)
            print("JSON is valid.")

        except json.JSONDecodeError as e:
            print("Error: Invalid JSON string.")
            print(e)
            return df
            
        if json_data:
            df = pd.DataFrame(json_data)
            if 'messages' in df:
                df = pd.json_normalize(df.messages)
                df = df[['date_unixtime', 'from', 'text']]
                df = df[  # Keep only rows with non-empty strings
                    (df['date_unixtime'].str.strip() != '') &
                    (df['from'].str.strip() != '') &
                    (df['text'].str.strip() != '')
                ].reset_index(drop=True)
                df = parse_telegram_data(df)  
            else:
                print("Error: 'messages' field not found in JSON data.")
        else:
            print("Error: Empty JSON file.")

    # Anonymize users with initials.
    df['User'] = df.User.apply(lambda x: ''.join([name[0].upper() for name in x.split()]))

    return df


def add_message_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'Message_Group' column to a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to add the 'Message_Group' column to.

    Returns:
        pd.DataFrame: The DataFrame with the added 'Message_Group' column.
    """
    df = df.copy()
    user_change = df['User'] != df['User'].shift()
    df['Message_Group'] = user_change.cumsum()
    
    return df


def add_convo_id(df: pd.DataFrame, minutes: int = 10) -> pd.DataFrame:
    """
    Add a 'Convo_ID' column to a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to add the 'Convo_ID' column to.
        minutes (int, optional): The number of minutes to consider a message as part of the same conversation. Defaults to 10.

    Returns:
        pd.DataFrame: The same DataFrame with the added 'Convo_ID' column.
    """        
    df = df.copy()
    df['Time_Delta'] = df['Date'].diff()
    cutoff_td = pd.Timedelta(minutes=minutes)
    mask = df['Time_Delta'] > cutoff_td

    # Initialize
    current_id = 0
    id_list = np.zeros(df.shape[0])
    
    # Assign conversation IDs based on the time delta
    for item in list(enumerate(mask))[1:]:
        if item[1]:
            current_id += 1
        id_list[item[0]] = current_id

    df['Convo_ID'] = id_list

    # Get the conversational rows where the number of unique users is greater than 1
    df_convo = df[df.Convo_ID != 0]
    df_convo = df_convo[df_convo.groupby('Convo_ID')['User'].transform('nunique') > 1]

    # Re-factorize the conversation IDs
    df_convo['Convo_ID'] = pd.factorize(df_convo.Convo_ID)[0] + 1

    # Get the non-conversational rows
    df_non_convo = df[~df.index.isin(df_convo.index)].copy()
    df_non_convo['Convo_ID'] = 0

    # Concatenate the conversational and non-conversational rows and sort by index
    df = pd.concat([df_convo, df_non_convo]).sort_index()

    return df


def add_response_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds two new columns to the input DataFrame: 'Is_Response' and 'Response_Time'.
    'Is_Response' indicates whether a message is a response by a different user.
    'Response_Time' is the time between the current message and the previous message.
    If the message is not a response, it is NaN.
    If the message is the first in a conversation, it is also NaN.

    Args:
        df (pd.DataFrame): The DataFrame to add the 'Is_Response' and 'Response_Time' columns to.

    Returns:
        pd.DataFrame: The same DataFrame with the added 'Is_Response' and 'Response_Time' columns.
    """
    df = df.copy()

    # Whether a message is a response by a different user
    df['Is_Response'] = df['User'] != df['User'].shift()

    # Add corresponding response time
    df_is_response = df.drop_duplicates(subset=['Message_Group'], keep='first').copy()
    df_is_not_response = df[~df.index.isin(df_is_response.index)].copy()
    df_is_response['Response_Time'] = df_is_response.Date.diff()
    df = pd.concat([df_is_response, df_is_not_response]).sort_index()

    # Whether a response is within a conversation (excl. first response)
    df_is_convo_response = df.copy()[df.Convo_ID != 0].groupby("Convo_ID").head(1)
    df_is_convo_response['Is_Convo_Response'] = False  # first message in any conversation to false
    df_else = df.copy()[~df.index.isin(df_is_convo_response.index)]
    # df_else['Is_Convo_Response'] = list(map(lambda x: False if x[0] == 0 else x[1], zip(df_else. Convo_ID, df_else['Is_Response'])))
    df_else['Is_Convo_Response'] = np.where(df_else['Convo_ID'] == 0, False, df_else['Is_Response'])
    df = pd.concat([df_is_convo_response, df_else]).sort_index()

    return df


def timedelta_to_str(td: pd.Timedelta) -> str:
    """
    Convert timedelta to days, hours, minutes, seconds.

    Args:
        td: A pandas Timedelta object.

    Returns:
        str: A string representation of the timedelta in days, hours, minutes, and seconds.
    """
    td_components = td.components
    days = td_components.days
    hours = td_components.hours
    minutes = td_components.minutes
    seconds = td_components.seconds

    components = []
    if days:
        components.append(f"{days} days")
    if hours:
        components.append(f"{hours} hrs")
    if minutes:
        components.append(f"{minutes} min")
    if seconds:
        components.append(f"{seconds} sec")

    return " ".join(components)


def transpose_stats_df(stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transposes a Pandas DataFrame that contains statistics about chat messages.

    Args:
        param stats_df: A Pandas DataFrame with a 'User' column and other columns
            representing message statistics.

    Returns:
        pd.DataFrame: The same DataFrame, but with the columns and rows transposed.
    """
    stats_df = stats_df.copy()
    stats_df = stats_df.set_index('User')
    stats_df = stats_df.T
    stats_df = stats_df.reset_index()
    stats_df.columns.name = None
    stats_df = stats_df.rename(columns={'index': ' '})
    return stats_df


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
        file_type = str(file[0]["type"]).lower()
        file_path = file[0]['datapath']

        if file is None:
            return pd.DataFrame(), [], []
        else:
            df = parse_data(file_type, file_path)
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
        df, date_range, users = parseFile()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df = df[
            (input.daterange()[0] <= df['Date'].dt.date)
            & (df['Date'].dt.date <= input.daterange()[1])
            & (df['User'].isin(input.users()))
            ].reset_index(drop=True)
        df = add_message_group(df)
        df = add_convo_id(df)
        df = add_response_time(df)
        df['Day'] = parse_day_column(df['Day'])
        df['Message_Length'] = df['Message'].str.split().str.len()
        return df, date_range, users
   

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
                        choices=["Mean", "Median"],
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
            except:
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
                timedelta_to_str(rt_max_gap),
                timedelta_to_str(rt_mean),
                timedelta_to_str(rt_median),
                timedelta_to_str(rt_sd),
            ])

            response_stats_df = pd.DataFrame(
                response_stats,
                columns=[
                    'User',
                    'Max inactivity',
                    'Mean responsiveness', 
                    'Median responsiveness', 
                    'Standard deviation',
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
        elif input.response_metric() == "In-convo":
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
            df['Message_Length'] = df['Message'].apply(lambda x: len(x))
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
                xaxis=dict(range=[0, np.percentile(df_grouped['Date'], 95)]),
            )
        elif input.convo_metric() == "Word count":
            df['Message_Length'] = df['Message'].apply(lambda x: len(x))
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
