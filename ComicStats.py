from typing import Tuple, List, Dict, Union
import re
import smtplib
import ssl
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import datetime as dt
import logging
import os
from copy import deepcopy

import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import yaml
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def convert_time(time) -> Union[dt.datetime, float]:
    """
    Convert pandas time to datetime format

    :param time: input time
    :return: time in datetime format
    """
    if isinstance(time, float):
        return time

    return dt.datetime.strptime(str(time)[:26], '%Y-%m-%d %H:%M:%S.%f')


def convert_string(raw_str: str) -> float:
    """
    Convert strings with 'K' or 'M' in them to actual float values

    :param raw_str: input string, i.e. 1.1k
    :return: float representation of string
    """
    # Convert views from string to float if K or M modifier included
    views_re = re.compile('(?P<start>.+)(?P<modifier>[km])', re.IGNORECASE)
    match = views_re.search(raw_str.lower())
    if match and match.group('modifier') == 'k':
        val = float(match.group('start')) * 1000
    elif match and match.group('modifier') == 'm':
        val = float(match.group('start')) * 1e6
    else:
        val = float(raw_str)

    return val


class ComicStats:
    """Class containing statistics scraped from webtoon site"""

    def __init__(self, url: str, no_save: bool, log_file: str) -> None:
        """
        Initialize variables

        :param url: url of webtoon canvas comic
        :param no_save: True if the data object created should not be saved for later
        :param log_file: log filepath
        """
        self.baseline_comic_names = ['Times', 'Subscribers', 'Views', 'Stars', 'Likes']
        self.no_save = no_save
        self.resp = requests.get(url)

        # Set up logging
        self.logger = logging.getLogger('ComicStats')
        self.logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(log_file, mode='a')
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.addHandler(fh)
        self.logger.info('Initializing class')

        # Fetch stats
        self.subs, self.views, self.stars = self._get_stats()
        self.names, self.likes = self._get_likes()

        # Create data representation
        col_names = deepcopy(self.baseline_comic_names)
        col_names += self.names
        input_data = [dt.datetime.now(), self.subs, self.views, self.stars, sum(self.likes)]
        input_data += self.likes
        self.data = pd.DataFrame(data=[input_data], columns=col_names)

        self.logger.info('Found stats: %s subscribers, %s views, %s rating', self.subs, self.views, self.stars)

        # Load historical stats & combine with current stats
        try:
            old_data = pd.read_csv('Data/trend.csv', index_col=0)

            # Add filler data columns to old data if they don't already exist
            if not all([col1 == col2 for col1, col2 in zip(self.data.columns, old_data.columns)]):
                add_columns = [column for column in self.data.columns if column not in old_data.columns]
                for column in add_columns:
                    old_data[column] = np.nan

            self.data = pd.concat([self.data, old_data], ignore_index=True, axis=0)
        except FileNotFoundError:
            self.logger.warning('No historical data found')

        # Save stats
        if not self.no_save:
            file_dest = 'Data/trend.csv'
            self.data.to_csv(file_dest)
            self.logger.debug('Saved data to {}'.format(file_dest))

        # Plot stats
        self._plot_stats(self.data, '')

        # Plot daily change
        times = [convert_time(time) for time in self.data['Times']]
        hour_diff = [(times[0] - time).total_seconds() / 3600 for time in times]
        day_diff = [hour // 24 for hour in hour_diff]

        indices = [day_diff.index(day) for day in list(set(day_diff))]
        daily_change_data = deepcopy(self.data.iloc[indices])

        # Format daily change data
        rows = daily_change_data.index
        ref_time = daily_change_data.iloc[0]['Times']
        for (rindx, row), indx in zip(daily_change_data.iterrows(), range(0, len(rows))):
            if indx == len(rows) - 1:
                break

            next_row = daily_change_data.iloc[indx+1]
            for column in row.index:
                if column == 'Times':
                    diff = convert_time(ref_time) - convert_time(next_row[column])
                    diff = diff.total_seconds()/3600/24

                else:
                    diff = row[column] - next_row[column]

                daily_change_data.at[rindx, column] = diff

        daily_change_data = daily_change_data.drop(daily_change_data.tail(1).index)

        self._plot_stats(daily_change_data, 'daily_change_')

    def _get_stats(self) -> Tuple[float, float, float]:
        """
        Fetch the stats for the comic. Return number of subs, views, and star rating

        :return: subs, views, star rating
        """
        soup = BeautifulSoup(self.resp.text, 'html.parser')
        stats = soup.findAll('ul', {'class': 'grade_area'})[0].findAll('em')

        subs = stats[0].contents[0]
        views = stats[1].contents[0]
        stars = float(stats[2].contents[0])

        subs = convert_string(subs)
        views = convert_string(views)

        return subs, views, stars

    def _get_likes(self) -> Tuple[List[str], List[float]]:
        """
        :return: number of likes per episode
        """
        soup = BeautifulSoup(self.resp.text, 'html.parser')
        likes = soup.findAll('ul', {'id': '_listUl'})[0].findAll('span', {'class': 'like_area _likeitArea'})
        names = soup.findAll('ul', {'id': '_listUl'})[0].findAll('span', {'class': 'subj'})

        names = [name.contents[0].contents[0] for name in names]
        likes = [float(like.contents[1]) for like in likes]
        report_str = ' | '.join(['{}: {} likes'.format(episode.split()[0], like) for like, episode in zip(likes, names)])

        self.logger.info('Found %s', report_str.encode('utf-8'))

        return names, likes

    def _plot_stats(self, data: pd.DataFrame, name: str) -> None:
        """
        Plot the metrics and save a figure.

        :param data: data to plot
        :param name: append this name to start of files
        """
        # Daily delta mode handles times as float numbers of days rather than datetimes
        if isinstance(data['Times'][0], float):
            daily_delta_mode = True
            times = data['Times']
        else:
            daily_delta_mode = False
            times = [convert_time(time) for time in data['Times']]

        # Plot overall metrics
        plt.subplot(311)
        plt.plot(times, data['Subscribers'])

        if daily_delta_mode:
            plt.xlabel('Elapsed Days')
            plt.title('Daily Delta Subscriber Count ({:.0f})'.format(data.iloc[0]['Subscribers']))
        else:
            plt.title('Subscriber Count ({:.0f})'.format(data.iloc[0]['Subscribers']))

        plt.subplot(312)
        plt.plot(times, data['Views'], 'r')

        if daily_delta_mode:
            plt.xlabel('Elapsed Days')
            plt.title('Daily Delta Number of Views ({:.0f})'.format(data.iloc[0]['Views']))
        else:
            plt.title('Number of Views ({:.0f})'.format(data.iloc[0]['Views']))

        plt.subplot(313)
        plt.plot(times, data['Stars'], 'g')

        if daily_delta_mode:
            plt.xlabel('Elapsed Days')
            plt.title('Daily Delta Rating ({:.2f})'.format(data.iloc[0]['Stars']))
        else:
            plt.title('Rating ({:.2f})'.format(data.iloc[0]['Stars']))

        # Format plot
        if not daily_delta_mode:
            plt.gcf().autofmt_xdate()

        plt.tight_layout()
        plt.savefig('Plots/{}stats.png'.format(name))

        plt.figure()
        plt.subplot(211)
        episode_columns = [column for column in data.columns if column not in self.baseline_comic_names]
        for episode in episode_columns:
            times_clean, data_clean = zip(*filter(lambda x: not np.isnan(x[1]), zip(times, data[episode])))
            plt.plot(times_clean, data_clean, label=episode.split()[0])
        plt.legend(loc='center left')

        if daily_delta_mode:
            plt.xlabel('Elapsed Days')
            plt.title('Daily Delta Number of Episode Likes')
        else:
            plt.title('Number of Episode Likes')

        plt.subplot(212)
        plt.plot(times, data['Likes'])

        if daily_delta_mode:
            plt.xlabel('Elapsed Days')
            plt.title('Daily Delta Number of Total Likes ({:.0f})'.format(data.iloc[0]['Likes']))
        else:
            plt.title('Number of Total Likes ({:.0f})'.format(data.iloc[0]['Likes']))

        # Format plot
        if not daily_delta_mode:
            plt.gcf().autofmt_xdate()

        plt.tight_layout()
        plt.savefig('Plots/{}likes.png'.format(name))

    def send_email(self) -> None:
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

            files = os.listdir('Plots')
            files.sort(key=lambda x: 'daily' in x)
            for file in files:
                with open('Plots/'+file, 'rb') as fin:
                    img = MIMEImage(fin.read(), name=file)
                    message.attach(img)

            server.sendmail(
                os.environ['USERNAME'], data['recipients'], message.as_string()
            )

        self.logger.info('Sent email with stats')
