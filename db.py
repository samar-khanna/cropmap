from collections import OrderedDict
import logging
import os
import sqlite3

from concurrent.futures import ThreadPoolExecutor
import requests

SQLITE_MEMDB_MAGIC_STR = ':memory:'

COLS = OrderedDict({ # can iterate over keys in order to produce valid row
    'acquisitionDate': 'text', # sqlite natively supports ISO8601 strs as dates
    'browseUrl': 'text',
    'dataAccessUrl': 'text',
    'downloadUrl': 'text',
    'entityId': 'text',
    'displayId': 'text',
    'cloudCover': 'real',
    'metadataUrl': 'text',
    'orderUrl': 'text',
    'browseUrl': 'text', 
    'h': 'integer',# not in api search results, requires bookkeeping
    'v': 'integer'# not in api search results, requires bookkeeping
})

CREATE_TABLE = 'CREATE TABLE search_results ({})'.format(
        ', '.join(k+' '+v for k,v in COLS.items())
)

INSERT = 'INSERT INTO search_results VALUES (' + ('?,'*(len(COLS)-1)) + '?)'

UPDATE = 'UPDATE search_results SET thumbnail = ? where entityId = ?'

CONNECTION, CURSOR, THREADPOOL, THREADS = range(4)


def connect(db_file=SQLITE_MEMDB_MAGIC_STR):
    exists = os.path.isfile(db_file)
    conn = sqlite3.connect(db_file) # file gets created if it doesn't exist
    cur = conn.cursor()
    if db_file == SQLITE_MEMDB_MAGIC_STR or not exists:
        # if we reach here, we are creating a new db
        cur.execute(CREATE_TABLE)
        conn.commit()

    return (conn, cur, ThreadPoolExecutor(), {})


def _cols_in_response():
    for k in COLS.keys():
        if k == 'h' or k == 'v':
            continue
        else:
            yield k


def add_records(db, h, v, query_results):
    rows = list(range(len(query_results)))
    for i, r in zip(rows, query_results):
        # TODO make this nicer
        rows[i] = tuple([r[k] for k in _cols_in_response()] + [h,v])
        #eid = r['entityId']
        #db[THREADS][eid] = db[THREADPOOL].submit(
        #        _update_thumbnail(db, eid,r['browseUrl']))

    logging.info('INSERTing %s records', len(rows))
    db[CURSOR].executemany(INSERT, rows)
    db[CONNECTION].commit()


def get_acquisition_dates(db, h,v):
    db[CURSOR].execute(
        'SELECT acquisitionDate FROM search_results WHERE h=? AND v=?', (h,v))
    return db[CURSOR].fetchall()


def get_thumbnail_urls(db, h, v, year):
    query = '''
    SELECT browseUrl 
    FROM search_results 
    WHERE h=? AND v=? AND acquisitionDate >= ? and acquisitionDate <= ?
    '''
    db[CURSOR].execute(query, (h, v, year+'-01-01', year+('-12-31')))
    return db[CURSOR].fetchall()

def close(db):
    db[CONNECTION].close()
    print('shutting down threadpool')
    '''
    for entity_id, fut in db[THREADS].items():
        res = fut.result()
        if res:
            print('{} - error:\n{}\n'.format(entity_id, res))
    db[THREADPOOL].shutdown()
    '''

