import pandas as pd
import requests
import json

import elasticsearch
import elasticsearch.helpers

from utilities import excel_reader

# See https://discuss.elastic.co/t/request-must-contain-a-kbn-xsrf-header/96736
_DEFAULT_KIBANA_HEADERS = {'kbn-xsrf': 'commons.elastic'}

_DEFAULT_TYPE_NAME = 'default'

# At index time, if no analyzer has been specified, it looks for an analyzer in the index settings
# called default. Failing that, it defaults to using the standard analyzer.
# See https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis.html
_DEFAULT_ANALYZER_NAME = 'default'

_DEFAULT_INDEX_SETTINGS = {
    'number_of_replicas': 0,
    'number_of_shards': 1,
    'analysis': {
        'filter': {
            'english_possessive_stemmer': {
                'type': 'stemmer',
                'language': 'possessive_english'
            },
            'english_stop': {
                'type': 'stop',
                'stopwords': '_english_'
            },
            'english_stemmer': {
                'type': 'stemmer',
                'language': 'english'
            },
        },
        'analyzer': {
            _DEFAULT_ANALYZER_NAME: {
                'tokenizer': 'standard',
                'filter': [
                    'english_possessive_stemmer',
                    'lowercase',
                    'english_stop',
                    'porter_stem'
                ]
            }
        }
    }
}

class ElasticClient:

    def __init__(self, es_host, kibana_host, es_user="", es_password=""):
        self.es_host = es_host
        self.kibana_host = kibana_host
        self.es_user = es_user
        self.es_password = es_password
        if self.es_user.strip():
            self.es = elasticsearch.Elasticsearch(
                self.es_host, http_auth=(self.es_user,
                    self.es_password))
        else:
            self.es = elasticsearch.Elasticsearch(self.es_host)

    def _create_index(self, index_name, properties, vers_7=False):
        print("Creating '%s' Elasticsearch index" % index_name)
        self.es.indices.delete(index=index_name, ignore=[404],
            timeout="30s", master_timeout="30s")
        if vers_7:
            self.es.indices.create(index=index_name, body={
                'settings': _DEFAULT_INDEX_SETTINGS,
                'mappings': {
                    'properties': properties
                }
            })
        else:
            self.es.indices.create(index=index_name, body={
                'settings': _DEFAULT_INDEX_SETTINGS,
                'mappings': {
                    _DEFAULT_TYPE_NAME: {
                        'properties': properties
                    }
                }
            })

    def _index_exists(self, index_name):
        try:
            index = self.es.indices.get(index=index_name)
            return True
        except:
            return False

    def _create_pattern(self, pattern_id, time_field, field_for_url_with_template="edit_url",
            fields_for_url = [],
            webservice_edit_host="localhost:8503"):
        print("Creating '%s' Kibana index pattern object" % pattern_id)
        attribs = {'title': pattern_id}
        if time_field is not None:
            attribs['timeFieldName'] = time_field
        attrib_mapping = {}
        if field_for_url_with_template is not None:
            attrib_mapping[field_for_url_with_template] = {
                "id":"url","params":{
                "urlTemplate":"http://%s/?_id={{value}}&_index=%s&_es_host=%s&_kibana_host=%s"%(
                    webservice_edit_host, pattern_id, self.es_host, self.kibana_host),
                "labelTemplate":"Edit row"}}
        for field in fields_for_url:
            attrib_mapping[field] = {"id":"url"}
        
        attribs['fieldFormatMap'] = json.dumps(attrib_mapping)
        resp = requests.post(
            'http://%s%s/api/saved_objects/index-pattern/%s?overwrite=true' % ("%s:%s@"%(
                    self.es_user, self.es_password) if self.es_user.strip() else "",
                self.kibana_host,
                pattern_id,
            ),
            headers=_DEFAULT_KIBANA_HEADERS,
            data=json.dumps({
                'attributes': attribs
            })
        ).raise_for_status()


    def _create_search(self, search_id, pattern_id, columns, security_enabled):
        print("Creating '%s' Kibana saved search object (%d columns)" %
              (search_id, len(columns)))
        requests.post(
            'http://%s%s/api/saved_objects/search/%s?overwrite=true' % ("%s:%s@"%(
                    self.es_user, self.es_password) if self.es_user.strip() else "",
                self.kibana_host,
                search_id,
            ),
            headers=_DEFAULT_KIBANA_HEADERS,
            data=json.dumps({
                'attributes': {'title': search_id,
                               'columns': columns,
                               'sort': ['_score', 'desc'],
                               'kibanaSavedObjectMeta': {
                                   'searchSourceJSON': "{\"index\": \"%s\", \"query\":{"
                                                       "\"language\":\"lucene\",\"query\":\"*\"}}" % pattern_id
                               }}
            })
        ).raise_for_status()

    def prepare_bulk_actions(self, documents, index_name, properties, vers_7=False, use_id=""):
        if not use_id.strip():
            return [{
                        '_index': index_name,
                        '_type': _DEFAULT_TYPE_NAME,
                        '_source': {
                            prop: value
                            for (prop, value)
                            in row.to_dict().items()
                            if prop in list(properties.keys())
                        }} if not vers_7 else {
                        '_index': index_name,
                        '_source': {
                            prop: value
                            for (prop, value)
                            in row.to_dict().items()
                            if prop in list(properties.keys())
                        }
                    } for index, row in documents.iterrows()]
        else:
            return [{
                        '_id': row[use_id],
                        '_index': index_name,
                        '_type': _DEFAULT_TYPE_NAME,
                        '_source': {
                            prop: value
                            for (prop, value)
                            in row.to_dict().items()
                            if prop in list(properties.keys())
                        }} if not vers_7 else {
                        '_id': row[use_id],
                        '_index': index_name,
                        '_source': {
                            prop: value
                            for (prop, value)
                            in row.to_dict().items()
                            if prop in list(properties.keys())
                        }
                    } for index, row in documents.iterrows()]


    def bulk_index(self, df, properties, index_name, time_field=None,
                   create_pattern=True, create_search=True,
                   field_for_url_with_template=None, webservice_edit_host=None,
                   vers_7=False, fields_for_url=[]):
        assert isinstance(df, pd.DataFrame)    
        documents = df.fillna('')  # TODO refactor
        self._create_index(index_name, properties)

        print('Preparing bulk operation')
        bulk_actions = self.prepare_bulk_actions(
            documents, index_name, properties, vers_7=vers_7)

        print('Indexing %d docs...' % len(bulk_actions))
        elasticsearch.helpers.bulk(self.es, bulk_actions,
                                   stats_only=True, chunk_size=1000, request_timeout=400)

        if create_pattern:
            self._create_pattern(pattern_id=index_name, time_field=time_field,
                field_for_url_with_template=field_for_url_with_template,
                webservice_edit_host=webservice_edit_host,
                fields_for_url=fields_for_url)

        if create_search:
            self._create_search(search_id=index_name, pattern_id=index_name,
                columns=sorted(properties.keys()))

        print('Done')

    def update_docs(self, df, properties, index_name, time_field=None,
                   create_pattern=True, create_search=False,
                   drop_index=False, vers_7=False, field_for_url_with_template=None,
                   webservice_edit_host=None, skip_uploading=False,
                   use_id="id", fields_for_url=[]):
        if not skip_uploading:
            assert isinstance(df, pd.DataFrame)    
            documents = df.fillna('')
            if not self._index_exists(index_name) or drop_index:
                self._create_index(index_name, properties, vers_7=vers_7)

            print('Preparing bulk operation')

            bulk_actions = self.prepare_bulk_actions(
            documents, index_name, properties, vers_7=vers_7, use_id=use_id)

            print('Indexing %d docs...' % len(bulk_actions))
            elasticsearch.helpers.bulk(self.es, bulk_actions,
                                       stats_only=True, chunk_size=1000, request_timeout=300)

        if create_pattern:
            self._create_pattern(pattern_id=index_name, time_field=time_field,
                field_for_url_with_template=field_for_url_with_template,
                webservice_edit_host=webservice_edit_host,
                fields_for_url=fields_for_url)

        if create_search:
            self._create_search(search_id=index_name, pattern_id=index_name,
                           columns=sorted(properties.keys()))

        print('Done')

    def bulk_index_big_data(self, folder_with_data, properties, index_name, time_field=None,
                   create_pattern=True, create_search=True, vers_7=False,
                   field_for_url_with_template=None, webservice_edit_host=None,
                   fields_for_url=[]):
        self._create_index(index_name, properties, vers_7=vers_7)
        print('Preparing bulk operation')
        for documents in excel_reader.ExcelReader().read_distributed_df_sequantially(folder_with_data):
                print("Read %d articles"%len(documents))
                bulk_actions = self.prepare_bulk_actions(
                    documents, index_name, properties, vers_7=vers_7)

                print('Indexing %d docs...' % len(bulk_actions))
                elasticsearch.helpers.bulk(self.es, bulk_actions,
                                           stats_only=True, chunk_size=1000, request_timeout=300)

        if create_pattern:
            _create_pattern(pattern_id=index_name, time_field=time_field, security_enabled=security_enabled,
                field_for_url_with_template=field_for_url_with_template,
                webservice_edit_host=webservice_edit_host,
                fields_for_url=fields_for_url)

        if create_search:
            _create_search(search_id=index_name, pattern_id=index_name,
                           columns=sorted(properties.keys()),
                           security_enabled = security_enabled)

        print('Done')

    def export_dashboard(dashboard_id):
        # https://www.elastic.co/guide/en/kibana/master/dashboard-import-api-export.html

        response = requests.get(
            'http://{}/api/kibana/dashboards/export?dashboard={}'.format(
                self.kibana_host,
                dashboard_id
            ),
            headers=_DEFAULT_KIBANA_HEADERS
        )
        response.raise_for_status()
        return response.json()


    def import_dashboard(self, dashboard_json, dashboard_id):
        # https://www.elastic.co/guide/en/kibana/master/dashboard-import-api-import.html

        try:
            requests.delete("http://{}/api/saved_objects/dashboard/{}".format(
                self.kibana_host, dashboard_id),
                headers=_DEFAULT_KIBANA_HEADERS).raise_for_status()
        except Exception:
            pass
        r = requests.post(
            "http://{}/api/kibana/dashboards/import?force=true".format(
                self.kibana_host
            ),
            headers=_DEFAULT_KIBANA_HEADERS,
            data=json.dumps(dashboard_json)
        )
        r.raise_for_status()
