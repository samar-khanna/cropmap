from collections import OrderedDict
import os
import sqlite3

from concurrent.futures import ThreadPoolExecutor
import requests

SQLITE_MEMDB_MAGIC_STR = ':memory:'

COLS = OrderedDict({ # can iterate over keys in order to produce valid row
    'acquisitionDate': 'text', # sqlite natively supports ISO8601 strs as dates
    'acquisitionDate': 'text',
    'browseUrl': 'text',
    'dataAccessUrl': 'text',
    'downloadUrl': 'text',
    'entityId': 'text',
    'displayId': 'text',
    'cloudCover': 'real',
    'metadataUrl': 'text',
    'orderUrl': 'text',
    'thumbnail': 'blob' # not in api search results, requires bookkeeping
})

CREATE_TABLE = 'CREATE TABLE search_results ({})'.format(
        ', '.join(k+' '+v for k,v in COLS.items())
)

INSERT = 'INSERT INTO search_results VALUES (' + ('?,'*len(COLS)-1) + '?)'

UPDATE = 'UPDATE search_results SET thumbnail = ? where entityId = ?'

CONNECTION, CURSOR, THREADPOOL = range(3)


def connect(db_file=SQLITE_MEMDB_MAGIC_STR):
    exists = os.path.isfile(db_file)
    conn = sqlite3.connect(db_file) # file gets created if it doesn't exist
    cur = conn.cursor()
    if db_file == SQLITE_MEMDB_MAGIC_STR or not exists:
        # if we reach here, we are creating a new db
        cur.execute(CREATE_TABLE)
        conn.commit()

    return conn, cur, ThreadPoolExecutor()


def _update_thumbnail(db, entity_id, thumbnail_url):
    try:
        response = requests.get(thumbnail_url)
        db[CURSOR].execute(UPDATE, response.content, entity_id)
        db[CONNECTION].commit()
    except Exception as ex:
        return ex
    # otherwise return nothing


def add_records(db, query_results):
    rows = map(lambda r: tuple([r[k] for k, _ in _custom_iteritems()] + [None]),
                query_results)
    db[CURSOR].executemany(INSERT, rows)
    db[CONNECTION].commit()


def close(db):
    db[CONNECTION].close()
    # TODO add threadpoolexecutor cleanup code


def _custom_iteritems():
    for k,v in COLS.items():
        if k == 'thumbnail':
            continue
        else:
            yield k,v
