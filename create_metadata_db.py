import logging
import sys

from db import connect, add_records, close
from metadata import *

MAX_RESULTS_PER_QUERY = 50000


def make_grid(hmin, hmax, vmin, vmax):
    return [(h,v) for h in range(hmin, hmax+1) for v in range(vmin, vmax+1)]

exclude = [
    (2,2), (2,3), (2,12), (2,13), (3,13), (14,19), (16,19), (16,18), (17,18),
    (25,17), (25,18), (25,19), (26,19), (27,19), (20,6), (31,6), (24,7),
    (25,7)
]
exclude.extend(make_grid(2,10,14,19))
exclude.extend(make_grid(11,13,17,19))
exclude.extend(make_grid(18,25,17,19))
exclude.extend(make_grid(21,24,16,16))
exclude.extend(make_grid(28,31,8,19))
exclude.extend(make_grid(27,27,13,19))
exclude.extend(make_grid(26,26,14,16))
exclude.extend(make_grid(29,31,7,7))
exclude.extend(make_grid(30,31,2,6))
exclude.extend(make_grid(8,29,2,2))
exclude.extend(make_grid(18,29,3,3))
exclude.extend(make_grid(19,29,4,4))
exclude.extend(make_grid(21,22,5,7))
exclude.extend(make_grid(23,26,5,6))

idxs = set(make_grid(2,31,2,19)).difference(set(exclude))


def main():
    try:
        api_key = login(sys.argv[1], sys.argv[2])
    except:
        logging.info('login failed!')
        sys.exit(-1)

    # don't do concurrent requests since they dont allow them anyway
    db = connect('test.db')
    for idx, i in zip(idxs, range(len(idxs))):
        try:
            h,v = idx
            starting_number=1
            response = do_query(api_key, h, v, MAX_RESULTS_PER_QUERY)
            total_hits = response['totalHits']
            logging.info('h=%s,v=%s,results=%s (%s/%s)', h, v, total_hits, i+1, len(idxs))
            next_record = response['nextRecord']
            results = response['results']
            while next_record != total_hits:
                starting_number += MAX_RESULTS_PER_QUERY 
                
                response = do_query(api_key, h, v, MAX_RESULTS_PER_QUERY, starting_number=starting_number)
                next_record = response['nextRecord']
                results.extend(response['results'])
            
            add_records(db, h, v, results)
            
        except Exception as e:
            logging.error(e)
            sys.exit(-1)

    close(db)


if __name__ == '__main__':
    logging.basicConfig(filename='test.log',level=logging.INFO,
            format='%(asctime)s %(message)s')
    main()
