"""
This module provides various utility functions and constants for data extraction, transformation, and loading (ETL) processes, particularly for handling WhatsApp chat messages and AWS services.

Imports:
- datetime: For date and time manipulation.
- re: For regular expressions.
- json: For JSON parsing.
- boto3: For AWS services interaction.
- numpy and pandas: For data manipulation.

Day- and hour-related constants:
- `days_of_week`: Array of days of the week.
- `hhmm_range`: Time range in 15-minute intervals.
- `hhmm_list`: List of time strings in 'HH:MM' format.
- `hh_range`: Time range in hourly intervals.
- `hh_list`: List of hour strings in 'HH' format.

Emoji pattern:
- `emoji_pattern`: Regular expression pattern to parse emojis from messages.

Functions:
- `extract_whatsapp_row`: Extracts date, time, user, and message from a WhatsApp chat message string.

Usage:
- Import the required functions and constants from this module to perform ETL operations on WhatsApp chat data and interact with AWS services.
"""

from datetime import datetime, timedelta

import re
import json

import numpy as np
import pandas as pd


# Day- and hour-related
days_of_week = np.array(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)
hhmm_range = [t.time() for t in pd.date_range('00:00', '23:59', freq='15min')]
hhmm_list = [t.strftime('%H:%M') for t in hhmm_range]
hh_range = [t.time() for t in pd.date_range('00:00', '23:59', freq='h')]
hh_list = [t.strftime('%H') for t in hh_range]

# Emoji pattern to parse emojis from messages
emoji_pattern = re.compile("[\U00010000-\U0010ffff]", flags=re.UNICODE)


##### ETL FUNCTIONS #####

def extract_whatsapp_row(string: str) -> list:
    """
    Extracts date, time, user, and message from a WhatsApp chat message string.

    Args:
        string (str): The WhatsApp message string to parse.

    Returns:
        list: A list containing the extracted date, time, user, and message.
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
        user_match = re.search(r'^(.*?):', user_and_message.group())
        if user_match:
            user = user_match.group().strip("[]-: ")
            user = re.sub(r" */ *", " ", user)
            output[2] = user

        # Extract message
        message_match = re.search(r':.*', user_and_message.group())
        if message_match:
            output[3]= message_match.group().lstrip("[]-: ,").strip()

    return output


def get_date_format(dates: pd.Series) -> str:
    """
    Determine the date format of a given pandas Series of date strings.

    Args:
        dates (pd.Series): A pandas Series containing date strings.

    Returns:
        str: The date format string that matches all dates in the Series, or None if no format matches.

    The function tries to parse the dates using a list of potential date formats. If all dates in the Series
    can be parsed with a particular format, that format is returned. If no format matches all dates, the function
    returns None.
    """
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

    for fmt in formats:
        parsed_dates = pd.to_datetime(dates, format=fmt, errors='coerce')
        if parsed_dates.notnull().all():
            print(f"Date format determined to be {fmt}")
            return fmt
    return None


def get_time_format(times: pd.Series) -> str:
    """
    Determines the time format of a given list of time strings.

    Args:
        times (list of str): A list of time strings to be analyzed.

    Returns:
        str or None: The matched time format string if a format is determined, 
                     otherwise None.

    Example:
        >>> get_time_format(["14:30:00", "15:45:00"])
        '%H:%M:%S'
    """
    # Potential time formats
    formats = [
        "%H:%M:%S",     # 14:30:00
        "%H:%M",        # 14:30
        "%I:%M %p",     # 02:30 PM
        "%I:%M:%S %p",  # 02:30:00 PM
        "%H%M",         # 1430
        "%I%M %p"       # 0230 PM
    ]

    for fmt in formats:
        parsed_times = pd.to_datetime(times, format=fmt, errors='coerce')
        if parsed_times.notnull().all():
            print(f"Time format determined to be {fmt}")
            return fmt
    return None


def get_day_of_week(i: int) -> str:
    """
    Returns the day of the week corresponding to the given index.

    Args:
        i (int): Index of the day in the week (0 for Monday, 1 for Tuesday, ..., 6 for Sunday).

    Returns:
        str: Name of the day of the week.

    Raises:
        IndexError: If the index is out of range (not between 0 and 6 inclusive).
    """
    return days_of_week[i]


def parse_day_column(df_col: pd.Series) -> pd.Series:
    """
    Converts a DataFrame column to a categorical type and reorders its categories 
    based on the days of the week.

    Parameters:
    df_col (pd.Series): The DataFrame column to be converted and reordered.

    Returns:
    pd.Series: The DataFrame column with reordered categories.
    """
    df_col = df_col.astype('category')
    existing_categories = [day for day in days_of_week if day in df_col.cat.categories]
    df_col = df_col.cat.reorder_categories(existing_categories, ordered=True)
    return df_col


def parse_whatsapp_data(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parses WhatsApp chat data from a DataFrame and returns a new DataFrame with additional columns.

    Args:
        input_df (pd.DataFrame): The input DataFrame containing WhatsApp chat data. 
                                 Expected columns are 'Date', 'Time', and 'Message'.

    Returns:
        pd.DataFrame: A new DataFrame with the following additional columns:
                      - 'Date': Combined and parsed date and time.
                      - 'MMYYYY': Month and year in '%m/%Y' format.
                      - 'Week': ISO calendar week number.
                      - 'Hour': Hour of the day.
                      - 'Time': Time rounded to the nearest 15 minutes.
                      - 'Day': Day of the week.
                      - 'Emojis': Extracted emojis from the 'Message' column.
    """
    df = input_df.copy()

    # Handling date and time formats because unknown
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


def parse_data(file_path: str, filename: str) -> pd.DataFrame:
    """
    Parse a WhatsApp or Telegram chat file and return a DataFrame.

    Args:
        file_path (str): The path to the file to parse.
        filename (str): The name of the file to parse.

    Returns:
        pd.DataFrame: The parsed DataFrame with the following columns:
            ['Date', 'Time', 'MMYYYY', 'Week', 'Hour', 'Day', 'User', 'Message', 'Emojis']
    """
    df = None

    with open(file_path, encoding='utf-8') as file:
        decoded = file.readlines()

    if 'txt' in filename.lower():
        print("Text file (i.e WhatsApp) detected.")

        # Split into list of messages and keep only relevant ones.
        parsed_lines = [extract_whatsapp_row(x) for x in decoded]
        parsed_lines = [row for row in parsed_lines if all(row)]  # Only filled rows
        df = pd.DataFrame(parsed_lines, columns=['Date', 'Time', 'User', 'Message'])
        df = parse_whatsapp_data(df)

    if 'json' in filename.lower():
        print("JSON file (i.e. Telegram) detected.")

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

    # Partially anonymize users using initials.  # To-do: handle clashing initials
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

    # # To-do: figure out why the following code doesn't work like the above
    # mask = df['Time_Delta'] > cutoff_td
    # shifted_mask = mask.shift(fill_value=False)
    # df['Convo_ID'] = shifted_mask.cumsum()

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
