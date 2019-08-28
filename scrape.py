#!/Users/Alex/.virtualenvs/bardsong/bin/python
import argparse
import datetime as dt
import os

from ComicStats import ComicStats

BASE_URL = 'https://www.webtoons.com/en/challenge/bardsong/list?title_no=305507'

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

    # Get dates of sent emails from log file
    LOG_FILE = 'log.info'
    with open(LOG_FILE, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()

    lines = [line for line in lines if 'email' in line] # Only save log lines saying an email was sent
    dates = {dt.datetime.strptime(line[0:10], '%Y-%m-%d').date() for line in lines}

    # Create stat class
    stats = ComicStats(BASE_URL, NO_SAVE, LOG_FILE)

    # Email stats
    if not dt.datetime.now().date() in dates or SEND:
        stats.send_email()
