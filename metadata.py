from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
import sys
import time
from math import inf
from traceback import print_exc
import xml.etree.ElementTree as ET

from numpy.random import randint
import requests

# TODO have random h & v
SEARCH_PARAMS = {
    'filterType': 'and',
    'childFilters': [
        {
            'filterType': 'value',
            'fieldId': 21787,
            'value': '3',
            'operand': '='
        },
        {
            'filterType': 'value',
            'fieldId': 21788,
            'value': '11',
            'operand': '='
        },
        {
            'filterType': 'value',
            'fieldId': 21789,
            'value': 'CU',
            'operand': '=',
        },
    ]
}


API_ROOT = 'https://earthexplorer.usgs.gov/inventory/json/v/1.4.1/'
XML_QUERY = '{http://earthexplorer.usgs.gov/eemetadata.xsd}metadataFields'


def to_json(**kwargs):
    return {'jsonRequest': json.dumps(kwargs)}


def login(u,p):
    data = to_json(username=u, password=p, catalogID='EE')
    response = requests.post(API_ROOT+'login?', data=data).json()
    if response['error']:
        raise Exception('EE: {}'.format(response['error']))
    return response['data']


def api_call(code, api_key, **kwargs):
    url =  API_ROOT + code
    kwargs.update(apiKey=api_key)
    params = to_json(**kwargs)
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print('Received status code {}'.format(response.status_code))
    print(response.status_code)
    data = response.json()
    if data['error']:
        raise Exception('EE: {}'.format(data['error']))
    else:
        return data['data']
 

def get_fill(root):
    return float(root.find(XML_QUERY)[9][0].text)


def get_valid_images_sorted(results, fills, preds, topk=.5):
    def get_dates(result):
        return datetime.strptime(result['acquisitionDate'], '%Y-%m-%d')

    def run_preds(date):
        for f in preds:
            if not f(date):
                return False
        return True

    # first, get indexes for which preds are true
    idxs = [
            (i,f,d) for i,f,d in zip(range(len(fills)), fills, map(get_dates, results))
                if run_preds(d)]
    idxs.sort(key=lambda t: t[2]) 
    ret = defaultdict(list)
    for i, f, d in idxs:
        results[i]['fill'] = f
        ret[d.year].append(results[i])

    return ret


def return_out_dict(metadata, search_resp, idx):
    return {
            'metadata': list(metadata)[idx], 
            'data': search_resp['results'][idx]
            }


def get_random_ard():
    return (str(randint(low=0, high=32)), str(randint(low=0,high=21)))


def print_stats(results):
    for r in results:
        print(r['acquisitionDate'], r['fill'])


if __name__ == '__main__':
    api_key = None
    h,v = sys.argv[3], sys.argv[4]
    print('H: {}, V: {}'.format(h,v))
    SEARCH_PARAMS['childFilters'][0]['value'] = h
    SEARCH_PARAMS['childFilters'][1]['value'] = v
    try:
        api_key = login(sys.argv[1], sys.argv[2])
        response = api_call('search',
                api_key,
                datasetName='ARD_TILE',
                includeUnknownCloudCover=False,
                maxCloudCover=10,
                sortOrder='DESC',
                maxResults=1000,
                additionalCriteria=SEARCH_PARAMS)
        print('Query returned {} results'.format(len(response['results'])))
        if len(response['results']) == 0:
            sys.exit(0)
        with open('{}_{}_results.json'.format(h,v), 'w') as fp:
            fp.write(json.dumps(response))

        results = response['results']

        metadata = list(map(lambda r: r['metadataUrl'], results))
        exc = ThreadPoolExecutor()
        xml_futs = exc.map(lambda m: requests.get(m).content, metadata)
        tick = time.time()
        xmls = [fut for fut in xml_futs]
        exc.shutdown(wait=False)
        print('{} s to get all metadata'.format(time.time()-tick))

        roots = map(lambda x: ET.fromstring(x), xmls)
        fills = list(map(get_fill, roots))
        preds = [
            lambda d: d.year >= 2012,
            lambda d: d.month == 8,
        ]
        to_mosaic = get_valid_images_sorted(results, fills, preds)
        with open('{}_{}_mosaic.json'.format(h,v), 'w') as fp:
            fp.write(json.dumps(to_mosaic))


        for year in to_mosaic.keys():
            print()
            print(year)
            print('------------------------')
            print_stats(to_mosaic[year])

                
    except Exception as ex:
        print_exc()
