from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
import time
from traceback import print_exc
import xml.etree.ElementTree as ET

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
    response = requests.get(url, params=params).json()
    if response['error']:
        raise Exception('EE: {}'.format(response['error']))
    else:
        return response['data']
 

def get_fill(root):
    return float(root.find(XML_QUERY)[9][0].text)


def return_out_dict(metadata, search_resp, idx):
    return {
            'metadata': list(metadata)[idx], 
            'data': search_resp['results'][idx]
            }


if __name__ == '__main__':
    try:
        if False: #os.path.exists('results.json'):
            with open('results.json', 'r') as fp:
                results = json.load(fp)
        else:
            api_key = login(sys.argv[1], sys.argv[2])
            response = api_call('search',
                    api_key,
                    datasetName='ARD_TILE',
                    includeUnknownCloudCover=False,
                    maxCloudCover=1,
                    sortOrder='DESC',
                    maxResults=1000,
                    additionalCriteria=SEARCH_PARAMS)
            with open('results.json', 'w') as fp:
                fp.write(json.dumps(response))

        print('Query returned {} results'.format(len(response['results'])))
        metadata = list(map(lambda r: r['metadataUrl'], response['results']))
        exc = ThreadPoolExecutor()
        xml_futs = exc.map(lambda m: requests.get(m).content, metadata)
        tick = time.time()
        xmls = [fut for fut in xml_futs]
        exc.shutdown(wait=False)
        print('{} s to get all metadata'.format(time.time()-tick))

        roots = map(lambda x: ET.fromstring(x), xmls)
        fills = list(map(get_fill, roots))
        min_idx = fills.index(min(fills))
        to_mosaic = []
        if min_idx > 0 :
            to_mosaic.append(return_out_dict(metadata, response, min_idx-1))
        to_mosaic.append(return_out_dict(metadata, response, min_idx))
        if min_idx+1 < len(metadata):
            to_mosaic.append(return_out_dict(metadata, response, min_idx+1))

        with open('mosaic.json', 'w') as fp:
            fp.write(json.dumps({'mosaic': to_mosaic}))
            
    except Exception as ex:
        print_exc()
