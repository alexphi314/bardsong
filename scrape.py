#!/Users/Alex/.virtualenvs/bardsong/bin/python
from typing import Tuple, List, Dict
import re
import argparse
import pickle
import logging
import datetime as dt
import smtplib
import ssl
import os
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from copy import deepcopy

import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import numpy as np

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

BASE_URL = 'https://www.webtoons.com/en/challenge/bardsong/list?title_no=305507'
BASELINE_COLUMN_NAMES = ['Times', 'Subscribers', 'Views', 'Stars', 'Likes']

def convert_time(time) -> dt.datetime:
    """
    Convert pandas time to datetime format

    :param time: input time
    :return: time in datetime format
    """
    return dt.datetime.strptime(str(time)[:26], '%Y-%m-%d %H:%M:%S.%f')

def get_stats() -> Tuple[float, float, float]:
    """
    Fetch the stats for the comic. Return number of subs, views, and star rating

    :return: subs, views, star rating
    """
    resp = requests.get(BASE_URL)

    soup = BeautifulSoup(resp.text, 'html.parser')
    stats = soup.findAll('ul', {'class': 'grade_area'})[0].findAll('em')

    subs = float(stats[0].contents[0])
    views = stats[1].contents[0]
    stars = float(stats[2].contents[0])

    # Convert views from string to float if K or M modifier included
    views_re = re.compile('(?P<start>.+)(?P<modifier>[km])', re.IGNORECASE)
    match = views_re.search(views.lower())
    if match and match.group('modifier') == 'k':
        views = float(match.group('start')) * 1000
    elif match and match.group('modifier') == 'm':
        views = float(match.group('start')) * 1e6
    else:
        views = float(views)

    return subs, views, stars

def get_likes() -> Tuple[List[str], List[float]]:
    """
    :return: number of likes per episode
    """
    resp = requests.get(BASE_URL)

    soup = BeautifulSoup(resp.text, 'html.parser')
    likes = soup.findAll('ul', {'id': '_listUl'})[0].findAll('span', {'class': 'like_area _likeitArea'})
    names = soup.findAll('ul', {'id': '_listUl'})[0].findAll('span', {'class': 'subj'})

    names = [name.contents[0].contents[0] for name in names]
    likes = [float(like.contents[1]) for like in likes]
    report_str = ' | '.join(['{}: {} likes'.format(episode.split()[0], like) for like, episode in zip(likes, names)])

    logger.info('Found %s', report_str.encode('utf-8'))

    return names, likes

def plot_stats(data: pd.DataFrame, names: List[str]) -> None:
    """
    Plot the metrics and save a figure.

    :param data: dataframe containing times, subs, views, and stars
    :param names: episode names
    """
    times = [convert_time(time) for time in data['Times']]

    # Plot overall metrics
    plt.subplot(311)
    plt.plot(times, data['Subscribers'])
    plt.title('Subscriber Count ({:.0f})'.format(data.iloc[0]['Subscribers']))

    plt.subplot(312)
    plt.plot(times, data['Views'], 'r')
    plt.title('Number of Views ({:.0f})'.format(data.iloc[0]['Views']))

    plt.subplot(313)
    plt.plot(times, data['Stars'], 'g')
    plt.title('Rating ({:.2f})'.format(data.iloc[0]['Stars']))

    # Format plot
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig('stats.png')

    plt.subplot(211)
    episode_columns = [column for column in data.columns if column not in BASELINE_COLUMN_NAMES]
    for episode in episode_columns:
        times_clean, data_clean = zip(*filter(lambda x: not np.isnan(x[1]), zip(times, data[episode])))
        plt.plot(times_clean, data_clean, label=episode.split()[0])
    plt.title('Number of Episode Likes')
    plt.legend(loc='center left')

    plt.subplot(212)
    plt.plot(times, data['Likes'])
    plt.title('Number of Total Likes ({:.0f})'.format(data.iloc[0]['Likes']))

    # Format plot
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.savefig('likes.png')

    # Plot daily change
    if (times[-1] + dt.timedelta(days=-1)) in times:
        data = data.loc[(convert_time(data['Times']) == times[-1]) or
                        (convert_time(data['Times']) == times[-1] + dt.timedelta(days=-1))]

def send_email() -> None:
    """
    Send metrics plot out via email
    """
    # Create a secure SSL context
    context = ssl.create_default_context()

    # Load recipients
    with open('config.yaml', 'rb') as fin:
        data = yaml.load(fin, Loader=yaml.SafeLoader)

    with smtplib.SMTP_SSL("smtp.gmail.com", port=465, context=context) as server:
        server.login(os.environ['USERNAME'], os.environ['PASSWORD'])

        message = MIMEMultipart()
        message['From'] = os.environ['USERNAME']
        message['To'] = ', '.join(data['recipients'])
        message['Subject'] = 'bardsong stats for {}'.format(dt.datetime.now().strftime('%b %d %y'))

        files = ['stats.png', 'likes.png']
        for file in files:
            with open(file, 'rb') as fin:
                img = MIMEImage(fin.read(), name=file)
                message.attach(img)

        server.sendmail(
            os.environ['USERNAME'], data['recipients'], message.as_string()
        )

if __name__ == '__main__':
    # Define input arguments
    parser = argparse.ArgumentParser()
    OPTIONAL = parser.add_argument_group('optional arguments')
    OPTIONAL.add_argument('--no_save', '-n', help='Do not save results from this run',
                          action='store_true', default=False)
    OPTIONAL.add_argument('--send', '-s', help='Send email reporting stats, even if already sent today',
                          action='store_true', default=False)
    args = parser.parse_args()

    NO_SAVE = args.no_save
    SEND = args.send

    # Change to current directory
    os.chdir('/Users/Alex/Documents/Projects/bardsong')

    # Set up logging
    LOG_FILE = 'log.info'
    logger = logging.getLogger('scrape')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(LOG_FILE, mode='a')
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

    # Get dates of sent emails from log file
    with open(LOG_FILE, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()

    lines = [line for line in lines if 'email' in line] # Only save log lines saying an email was sent
    dates = {dt.datetime.strptime(line[0:10], '%Y-%m-%d').date() for line in lines}

    logger.info('Launching scraping script')

    # Fetch stats
    subs, views, stars = get_stats()
    names, likes = get_likes()

    col_names = deepcopy(BASELINE_COLUMN_NAMES)
    col_names += names
    input_data = [dt.datetime.now(), subs, views, stars, sum(likes)]
    input_data += likes
    data = pd.DataFrame(data=[input_data], columns=col_names)

    logger.info('Found stats: %s subscribers, %s views, %s rating', subs, views, stars)

    # Load historical stats
    try:
        old_data = pd.read_csv('Data/trend.csv', index_col=0)

        # Add filler data columns to old data if they don't already exist
        if not all([col1 == col2 for col1, col2 in zip(data.columns, old_data.columns)]):
            add_columns = [column for column in data.columns if column not in old_data.columns]
            for column in add_columns:
                old_data[column] = 0

        data = pd.concat([data, old_data], ignore_index=True, axis=0)
    except Exception as e:
        logger.warning('No historical data found')

    # Plot stats
    plot_stats(data, names)
    logger.debug('Plotted stats')

    # Save stats
    if not NO_SAVE:
        data.to_csv('Data/trend.csv')

    # Email stats
    if not dt.datetime.now().date() in dates or SEND:
        send_email()
        logger.info('Sent email with stats')
