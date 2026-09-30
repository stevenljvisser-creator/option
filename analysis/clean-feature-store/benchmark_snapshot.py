"""Compare live-settings and job-snapshot branches with identical offline I/O."""
import hashlib
import json
import time
from test_storage_snapshot import SnapshotTests

results=[]
for scoped in (False,True):
    test=SnapshotTests();test.setUp();n=test.ns
    def run():
        for _ in range(10000):
            n['get_bytes']('input')
            n['put_bytes']('output',b'bytes')
            n['exists']('input')
            list(n['list_keys']('prefix'))
    cpu=time.process_time();start=time.perf_counter()
    if scoped:
        with n['storage_settings_scope'](n['load_storage_settings']()):run()
    else:run()
    wall=time.perf_counter()-start
    trace=repr(test.requests).encode()
    results.append(dict(mode='job_snapshot' if scoped else 'live_settings',
        mock_storage_requests=len(test.requests),individual_settings_reads=len(test.lookups),
        snapshot_reads=test.loads,wall_seconds=wall,cpu_seconds=time.process_time()-cpu,
        request_trace_sha256=hashlib.sha256(trace).hexdigest()))
assert results[0]['request_trace_sha256']==results[1]['request_trace_sha256']
print(json.dumps({'scope':'offline mocks; no marketdata throughput or real I/O measured','results':results},indent=2))
