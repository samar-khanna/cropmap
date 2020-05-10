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

def _month_str(month):
    t = {
        5: 'May',
        6: 'Jun',
        7: 'Jul',
        8: 'Aug',
        9: 'Sep'
    }
    return t[month]


if __name__ == '__main__':
    args = parser.parse_args()
    db = connect(args.db_file)
    if args.print_stats: # query against h,v and then find unique years
        dates = get_acquisition_dates(db, args.h, args.v)
        partition = defaultdict(lambda : defaultdict(lambda : 0))
        recent = filter(lambda d: d['year'] >= 2013, dates)
        for date in recent:
            partition[date['year']][date['month']] += 1

        for year, months in partition.items():
            print(year)
            for month in months:
                print('\t', _month_str(month), partition[year][month])
            
    if args.output_images: # query against h,v, and year
        urls = get_thumbnail_urls(db, args.h, args.v, args.year)
        with open('output_images.txt', 'w') as fp:
            for l in map(lambda u: u+'\n', urls):
                fp.write(l)
