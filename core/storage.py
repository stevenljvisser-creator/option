import gzip,json,io
from functools import lru_cache
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass,field
import boto3
import pandas as pd
from botocore.config import Config
from core.runtime_settings import get_setting,get_storage_settings

@dataclass(frozen=True)
class StorageSettings:
    endpoint:str|None
    region:str|None
    bucket:str|None
    access:str|None=field(repr=False)
    secret:str|None=field(repr=False)

_job_settings=ContextVar("storage_job_settings",default=None)

def load_storage_settings():
    values=get_storage_settings()
    return StorageSettings(
        values.get("HETZNER_S3_ENDPOINT"),values.get("HETZNER_S3_REGION","hel1"),
        values.get("HETZNER_S3_BUCKET"),values.get("HETZNER_S3_ACCESS_KEY"),
        values.get("HETZNER_S3_SECRET_KEY")
    )

@contextmanager
def storage_settings_scope(settings):
    """Bind a job snapshot; executor threads must enter this scope explicitly."""
    token=_job_settings.set(settings)
    try:
        yield
    finally:
        _job_settings.reset(token)

@lru_cache(maxsize=8)
def _cached_client(endpoint,region,access,secret):
    """Reuse the boto3 connection pool instead of rebuilding it per object.

    Clean Feature Store jobs perform many LIST/HEAD/GET/PUT operations.  A new
    boto3 client for every request throws away keep-alive connections and was a
    measurable source of avoidable latency.  The cache key contains every
    credential/endpoint setting, so a settings change automatically creates a
    fresh client without mutating old clients in flight.
    """
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts":8,"mode":"adaptive"},
            max_pool_connections=64,
            connect_timeout=10,
            read_timeout=120,
        )
    )

def client():
    settings=_job_settings.get()
    if settings is not None:
        if not all([settings.endpoint,settings.bucket,settings.access,settings.secret]):
            raise RuntimeError("Hetzner Object Storage is nog niet ingesteld via de website.")
        return _cached_client(settings.endpoint,settings.region,settings.access,settings.secret)
    endpoint=get_setting("HETZNER_S3_ENDPOINT")
    region=get_setting("HETZNER_S3_REGION","hel1")
    bucket=get_setting("HETZNER_S3_BUCKET")
    access=get_setting("HETZNER_S3_ACCESS_KEY")
    secret=get_setting("HETZNER_S3_SECRET_KEY")
    if not all([endpoint,bucket,access,secret]):
        raise RuntimeError("Hetzner Object Storage is nog niet ingesteld via de website.")
    return _cached_client(endpoint,region,access,secret)

def bucket_name():
    settings=_job_settings.get()
    b=settings.bucket if settings is not None else get_setting("HETZNER_S3_BUCKET")
    if not b:
        raise RuntimeError("Object Storage bucket ontbreekt.")
    return b

def put_bytes(key,data,content_type="application/octet-stream",content_encoding=None):
    kw={"Bucket":bucket_name(),"Key":key,"Body":data,"ContentType":content_type}
    if content_encoding:
        kw["ContentEncoding"]=content_encoding
    client().put_object(**kw)

def get_bytes(key):
    return client().get_object(Bucket=bucket_name(),Key=key)["Body"].read()

def put_df(key,df):
    payload=gzip.compress(df.to_csv(index=False).encode("utf-8"),compresslevel=1)
    put_bytes(key,payload,"text/csv","gzip")

def get_df(key,usecols=None):
    raw=get_bytes(key)
    try:
        raw=gzip.decompress(raw)
    except Exception:
        pass
    return pd.read_csv(io.BytesIO(raw),low_memory=False,usecols=usecols)

def put_json(key,obj):
    put_bytes(
        key,
        json.dumps(obj,ensure_ascii=False,default=str).encode("utf-8"),
        "application/json"
    )

def get_json(key):
    raw=get_bytes(key)
    try:
        raw=gzip.decompress(raw)
    except Exception:
        pass
    return json.loads(raw.decode("utf-8"))

def exists(key):
    try:
        client().head_object(Bucket=bucket_name(),Key=key)
        return True
    except Exception:
        return False

def list_keys(prefix):
    c=client()
    token=None
    while True:
        kw={"Bucket":bucket_name(),"Prefix":prefix,"MaxKeys":1000}
        if token:
            kw["ContinuationToken"]=token
        r=c.list_objects_v2(**kw)
        for x in r.get("Contents",[]) or []:
            yield x["Key"]
        if not r.get("IsTruncated"):
            break
        token=r.get("NextContinuationToken")

def head_meta(key):
    r=client().head_object(Bucket=bucket_name(),Key=key)
    return {
        "key":key,
        "size":int(r.get("ContentLength",0) or 0),
        "last_modified":r.get("LastModified"),
        "etag":str(r.get("ETag","")).strip('"'),
        "content_type":r.get("ContentType"),
    }

def list_objects_meta(prefix="",callback=None):
    c=client()
    token=None
    scanned=0
    pages=0
    while True:
        kw={"Bucket":bucket_name(),"Prefix":prefix,"MaxKeys":1000}
        if token:
            kw["ContinuationToken"]=token
        r=c.list_objects_v2(**kw)
        pages+=1
        items=r.get("Contents",[]) or []
        scanned+=len(items)
        if callback:
            callback(scanned,pages,r.get("IsTruncated",False))
        for x in items:
            yield {
                "key":x["Key"],
                "size":int(x.get("Size",0) or 0),
                "last_modified":x.get("LastModified"),
                "etag":str(x.get("ETag","")).strip('"'),
            }
        if not r.get("IsTruncated"):
            break
        token=r.get("NextContinuationToken")

def test_connection():
    client().head_bucket(Bucket=bucket_name())
    return True
