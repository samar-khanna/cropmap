import argparse
from collections import defaultdict
import sys

from db import connect, get_acquisition_dates, get_thumbnail_urls

parser = argparse.ArgumentParser('Query Metadata DB')
parser.add_argument('db_file', help='sqlite file to query')
parser.add_argument('h', help='ARD h value to query')
parser.add_argument('v', help='ARD v value to query')
parser.add_argument('year', help='year to inspect')
parser.add_argument('--print-stats', action='store_true',
    help='Print yearly stats for tile')
parser.add_argument('--output-images', action='store_true',
        help='output image urls in text file for display by `feh`')

def _strip_year(date_str):
    return date_str.split('-')[0]

if __name__ == '__main__':
    args = parser.parse_args()
    db = connect(args.db_file)
    if args.print_stats: # query against h,v and then find unique years
        dates = get_acquisition_dates(db, args.h, args.v)
        partition = defaultdict(lambda : 0)
        for date in dates:
            partition[_strip_year(date[0])] += 1
        for k in sorted(partition, key=lambda k: partition[k], reverse=True):
            print(k, partition[k])

    if args.output_images: # query against h,v, and year
        tups = get_thumbnail_urls(db, args.h, args.v, args.year)
        with open('output_images.txt', 'w') as fp:
            for l in map(lambda t: t[0]+'\n', tups):
                fp.write(l)
