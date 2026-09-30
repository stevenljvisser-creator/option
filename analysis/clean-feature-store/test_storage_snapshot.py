"""Dependency-free isolation tests of the actual source via AST extraction.
No application imports, database connections, credentials, or network access.
"""
import ast
import gzip
import io
import json
import unittest
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from types import SimpleNamespace
import gc
import time

ROOT=Path(__file__).resolve().parents[2]

def extract(path,names,namespace):
    tree=ast.parse((ROOT/path).read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)

class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.lookups=[];self.requests=[];self.loads=0
        self.values={'HETZNER_S3_ENDPOINT':'endpoint','HETZNER_S3_BUCKET':'bucket',
                     'HETZNER_S3_ACCESS_KEY':'access','HETZNER_S3_SECRET_KEY':'secret'}
        def get_setting(key,default=None):
            self.lookups.append(key);return self.values.get(key,default)
        def get_storage_settings():
            self.loads+=1;return self.values.copy()
        class Client:
            def __getattr__(client,operation):
                def request(**kwargs):
                    self.requests.append((operation,kwargs))
                    return {'Body':io.BytesIO(b'payload'),'Contents':[{'Key':'key'}],
                            'IsTruncated':False}
                return request
        self.ns={'contextmanager':contextmanager,'ContextVar':ContextVar,
                 'dataclass':dataclass,'field':field,'get_setting':get_setting,
                 'get_storage_settings':get_storage_settings,'gzip':gzip,'io':io,'json':json,
                 '_cached_client':lru_cache(maxsize=8)(lambda *args:Client()),
                 '_job_settings':ContextVar('test_job_settings',default=None)}
        extract(Path('core/storage.py'),{'StorageSettings','load_storage_settings',
            'storage_settings_scope','client','bucket_name','get_bytes','put_bytes',
            'exists','list_keys'},self.ns)

    def operations(self):
        n=self.ns
        self.assertEqual(n['get_bytes']('input'),b'payload')
        n['put_bytes']('output',b'bytes','text/csv','gzip')
        self.assertTrue(n['exists']('input'))
        self.assertEqual(list(n['list_keys']('prefix')),['key'])

    def test_same_requests_and_no_per_object_settings_reads(self):
        self.operations();old=self.requests.copy()
        self.assertEqual(len(self.lookups),24)
        self.requests.clear();self.lookups.clear()
        settings=self.ns['load_storage_settings']()
        with self.ns['storage_settings_scope'](settings):self.operations()
        self.assertEqual(self.requests,old)
        self.assertEqual(self.lookups,[]);self.assertEqual(self.loads,1)

    def test_thread_scope_isolation_rotation_and_exception_cleanup(self):
        settings=self.ns['load_storage_settings']()
        self.assertNotIn('secret',repr(settings));self.assertNotIn('access',repr(settings))
        with self.assertRaises(RuntimeError):
            with self.ns['storage_settings_scope'](settings):
                self.values['HETZNER_S3_BUCKET']='new'
                def work():
                    with self.ns['storage_settings_scope'](settings):
                        return self.ns['bucket_name']()
                with ThreadPoolExecutor(max_workers=4) as pool:
                    self.assertEqual(list(pool.map(lambda _:work(),range(20))),['bucket']*20)
                    self.assertEqual(pool.submit(self.ns['bucket_name']).result(),'new')
                self.assertEqual(self.ns['bucket_name'](),'bucket')
                raise RuntimeError('test')
        self.assertEqual(self.ns['bucket_name'](),'new')
        with self.ns['storage_settings_scope'](self.ns['load_storage_settings']()):
            self.assertEqual(self.ns['bucket_name'](),'new')

    def test_nested_scopes_and_missing_configuration(self):
        a=self.ns['load_storage_settings']();self.values['HETZNER_S3_BUCKET']='other'
        b=self.ns['load_storage_settings']()
        with self.ns['storage_settings_scope'](a):
            with self.ns['storage_settings_scope'](b):self.assertEqual(self.ns['bucket_name'](),'other')
            self.assertEqual(self.ns['bucket_name'](),'bucket')
        self.values.clear()
        with self.ns['storage_settings_scope'](self.ns['load_storage_settings']()):
            with self.assertRaisesRegex(RuntimeError,'bucket ontbreekt'):self.ns['bucket_name']()
            with self.assertRaisesRegex(RuntimeError,'nog niet ingesteld'):self.ns['client']()
            self.assertFalse(self.ns['exists']('input'))

    def test_worker_threads_and_receipts_share_one_snapshot(self):
        n=self.ns;observed=[]
        def save(ticker,day,h):
            observed.append(('save',n['bucket_name']()))
            self.values['HETZNER_S3_BUCKET']='rotated'
            return 2
        n.update(time=time,gc=gc,ThreadPoolExecutor=ThreadPoolExecutor,as_completed=as_completed,
            FEATURE_ROOT='features',initial_counts=lambda *a:(0,2),
            planned_dates_for_ticker=lambda job,ticker,days:days,
            save_feature_day=save,check_stop=lambda *a:False,
            write_receipt=lambda *a,**kw:observed.append(('receipt',n['bucket_name']())),
            update_company=lambda *a:None,finalize_company_rows=lambda *a:None,
            finish_by_progress=lambda *a:None)
        extract(Path('worker/tasks.py'),{'run_features','_run_features'},n)
        job=SimpleNamespace(payload={'parallel_workers':2},tickers=['A'],id=1,
                            completed_units=0,end_date=date(2025,1,3))
        n['run_features'](SimpleNamespace(commit=lambda:None),job,[date(2025,1,2),date(2025,1,3)])
        self.assertEqual(self.loads,1);self.assertEqual(self.lookups,[])
        self.assertEqual(observed.count(('save','bucket')),2)
        self.assertEqual(observed.count(('receipt','bucket')),2)
        self.assertEqual(job.completed_units,2)
        self.assertEqual(n['bucket_name'](),'rotated')

    def test_settings_query_is_selective_single_query_and_closes(self):
        calls=[]
        class Query:
            def where(self,predicate):calls.append(('filter',predicate));return self
        class Key:
            def in_(self,keys):return keys
        class DB:
            def scalars(self,query):calls.append('query');return self
            def all(self):return [SimpleNamespace(key='HETZNER_S3_SECRET_KEY',value='encrypted',encrypted=True)]
            def close(self):calls.append('close')
        ns={'SessionLocal':DB,'select':lambda model:Query(),
            'AppSetting':SimpleNamespace(key=Key()),'decrypt_value':lambda value:'decrypted'}
        extract(Path('core/runtime_settings.py'),{'get_storage_settings'},ns)
        self.assertEqual(ns['get_storage_settings'](),{'HETZNER_S3_SECRET_KEY':'decrypted'})
        self.assertEqual(calls.count('query'),1);self.assertEqual(calls[-1],'close')
        self.assertEqual(len(calls[0][1]),5)
        ns['decrypt_value']=lambda value:(_ for _ in ()).throw(ValueError('decode'))
        with self.assertRaises(ValueError):ns['get_storage_settings']()
        self.assertEqual(calls[-1],'close')

    def test_worker_scope_restored_on_failure_and_stop(self):
        n=self.ns
        extract(Path('worker/tasks.py'),{'run_features'},n)
        def fail(*args):
            self.assertEqual(n['bucket_name'](),'bucket')
            raise RuntimeError('build failure')
        n['_run_features']=fail
        with self.assertRaisesRegex(RuntimeError,'build failure'):
            n['run_features'](None,None,[])
        self.assertIsNone(n['_job_settings'].get())
        n['_run_features']=lambda *args:'stopped'
        self.assertEqual(n['run_features'](None,None,[]),'stopped')
        self.assertIsNone(n['_job_settings'].get())

if __name__=='__main__':unittest.main()
