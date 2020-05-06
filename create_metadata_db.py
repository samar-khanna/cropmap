import sys

from db import connect, add_records, close
from metadata import *

MAX_RESULTS_PER_QUERY = 50000


def make_grid(hmin, hmax, vmin, vmax):
    return [(h,v) for h in range(hmin, hmax+1) for v in range(vmin, vmax+1)]


exclude = [
    (2,2), (2,3), (2,12), (2,13), (3,13), (14,19)
]
exclude.extend(make_grid(2,10,14,19))
exclude.extend(make_grid(11,13,17,19))

idxs = set(make_grid(2,32,2,20)).difference(set(exclude))

try:
    api_key = login(sys.argv[1], sys.argv[2])
except:
    print('login failed!')
    sys.exit(-1)

# don't do concurrent requests since they dont allow them anyway
db = connect('test.db')
for idx in idxs:
    try:
        h,v = idx
        starting_number=1
        response = do_query(api_key, h, v, MAX_RESULTS_PER_QUERY)
        total_hits = response['totalHits']
        next_record = response['nextRecord']
        results = response['results']
        while next_record != total_hits:
            starting_number += MAX_RESULTS_PER_QUERY 
            
            response = do_query(api_key, h, v, MAX_RESULTS_PER_QUERY, starting_number=starting_number)
            next_record = response['nextRecord']
            results.extend(response['results'])
        
        add_records(db, results)
        break
        
    except Exception as e:
        print(e)
        sys.exit(-1)

close(db)

