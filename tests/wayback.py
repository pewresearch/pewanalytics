from __future__ import print_function
from builtins import range
from . import *
import requests, datetime
from bs4 import BeautifulSoup


def get_wayback_url(url, start_date, end_date, day_steps=10):
    urls = []
    current_date = start_date
    while current_date < end_date:
        try:
            response = requests.get("http://archive.org/wayback/available", params={
                "url": url,
                "timestamp" : current_date.strftime("%Y%m%d"),
            }).json()
            if 'archived_snapshots' in list(response.keys()) and 'closest' in list(response['archived_snapshots'].keys()) \
                    and response['archived_snapshots']['closest']['available']:
                new_url = response['archived_snapshots']['closest']['url']
                if new_url not in urls:
                    urls.append(new_url)
            else:
                print(response)
                import pdb
                pdb.set_trace()
        except Exception as e:
            print(e)
        current_date += datetime.timedelta(days=day_steps)
    return urls


base_url = 'https://meta.wikimedia.org/wiki/List_of_Wikipedias'
wayback_urls = get_wayback_url(base_url,
    start_date = datetime.datetime(2005, 1, 1),
    end_date = datetime.datetime(2007, 1, 1))

for og in wayback_urls:
    req = requests.get(pg).text
    soup = BeautifulSoup(req, 'lxml')

    t = soup.findAll('ol')
    stuff = []
    for i in range(len(t)):
        stuff.append(t[0].text)
    break

stuff = flatten_list(stuff)
for s in stuff:
    s = s.split('\n')
    if is_not_null(s):
        s = s.split('-')
