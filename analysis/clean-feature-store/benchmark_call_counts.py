"""Offline measurement of actual storage wrappers; no imports of application runtime.
Run: python3 analysis/clean-feature-store/benchmark_call_counts.py
Mocks count setting lookups and requests. Timings are mock overhead, not I/O.
"""
import ast, io, json, time, resource, hashlib
from pathlib import Path
from functools import lru_cache
root=Path(__file__).resolve().parents[2]
source=(root/'core/storage.py').read_text()
names={'client','bucket_name','get_bytes','put_bytes','exists','list_keys'}
module=ast.Module(body=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
counts={}
settings={'HETZNER_S3_ENDPOINT':'offline','HETZNER_S3_REGION':'offline','HETZNER_S3_BUCKET':'offline','HETZNER_S3_ACCESS_KEY':'offline','HETZNER_S3_SECRET_KEY':'offline'}
def setting(key,default=None):
    counts['settings_reads']=counts.get('settings_reads',0)+1
    return settings.get(key,default)
class Client:
    def __getattr__(self,name):
        def call(**kwargs):
            counts[name]=counts.get(name,0)+1
            return {'Body':io.BytesIO(b''),'Contents':[],'IsTruncated':False}
        return call
namespace={'get_setting':setting,'_cached_client':lru_cache(maxsize=8)(lambda *args:Client())}
exec(compile(module,str(root/'core/storage.py'),'exec'),namespace)
results=[]
for name,call in [('HEAD',lambda:namespace['exists']('offline')),('GET',lambda:namespace['get_bytes']('offline')),('PUT',lambda:namespace['put_bytes']('offline',b'')),('LIST',lambda:list(namespace['list_keys']('offline')))]:
    for n in (1000,10000,100000):
        counts.clear();cpu=time.process_time();start=time.perf_counter()
        for _ in range(n):call()
        elapsed=time.perf_counter()-start
        results.append(dict(operation=name,operations=n,wall_seconds=elapsed,cpu_seconds=time.process_time()-cpu,**counts))
print(json.dumps({'scope':'offline mocks; no market records, real DB queries or network requests','source_sha256':hashlib.sha256(source.encode()).hexdigest(),'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'results':results},indent=2))
