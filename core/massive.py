import re,requests,boto3,gzip,gc,io
from botocore.config import Config
import pandas as pd
from core.runtime_settings import get_setting,import_chunk_rows

class CountingBody(io.RawIOBase):
    """Wrap boto3 StreamingBody and keep an exact compressed-byte counter."""
    def __init__(self,body):
        self.body=body
        self.bytes_read=0
    def readable(self):
        return True
    def read(self,size=-1):
        data=self.body.read(size)
        if data:
            self.bytes_read+=len(data)
        return data
    def readinto(self,b):
        data=self.body.read(len(b))
        if not data:
            return 0
        n=len(data)
        b[:n]=data
        self.bytes_read+=n
        return n
    def close(self):
        try:self.body.close()
        finally:super().close()

def option_underlying_series(series):
    return series.astype("string").str.extract(
        r"^O:([A-Z0-9.\-]+?)\d{6}[CP]\d{8}$",expand=False
    )

def flat_client():
    access=get_setting("MASSIVE_FLAT_ACCESS_KEY")
    secret=get_setting("MASSIVE_FLAT_SECRET_KEY")
    endpoint=get_setting("MASSIVE_FLAT_ENDPOINT","https://files.massive.com")
    if not access or not secret:
        raise RuntimeError("Massive Flat Files is nog niet ingesteld via de website.")
    session=boto3.Session(
        aws_access_key_id=access,
        aws_secret_access_key=secret
    )
    return session.client(
        "s3",
        endpoint_url=endpoint,
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts":8,"mode":"adaptive"},
            max_pool_connections=8
        )
    )

def flat_bucket():
    return get_setting("MASSIVE_FLAT_BUCKET","flatfiles")

def flat_key(dataset,day):
    d=pd.Timestamp(day).date()
    return f"{dataset}/{d:%Y}/{d:%m}/{d:%Y-%m-%d}.csv.gz"

def flat_chunks(dataset,day,chunksize=None):
    """
    Massive -> gzip stream -> pandas chunks.
    Progress is based on actual compressed bytes read, so the progress bar
    always advances while the file is being consumed.
    """
    chunksize=int(chunksize or import_chunk_rows())
    resp=flat_client().get_object(
        Bucket=flat_bucket(),
        Key=flat_key(dataset,day)
    )
    total=max(1,int(resp.get("ContentLength") or 1))
    counter=CountingBody(resp["Body"])
    buffered=io.BufferedReader(counter,buffer_size=1024*1024)
    gz=gzip.GzipFile(fileobj=buffered,mode="rb")
    try:
        reader=pd.read_csv(gz,chunksize=chunksize,low_memory=False)
        for n,chunk in enumerate(reader,1):
            fraction=min(0.999,counter.bytes_read/total)
            yield chunk,float(fraction),n
    finally:
        try:gz.close()
        except Exception:pass
        try:buffered.close()
        except Exception:pass
        try:counter.close()
        except Exception:pass

class MassiveREST:
    def __init__(self):
        self.base=get_setting("MASSIVE_REST_ENDPOINT","https://api.massive.com")
        self.key=get_setting("MASSIVE_API_KEY")
        if not self.key:
            raise RuntimeError("Massive REST API key is nog niet ingesteld via de website.")

    def _get_url(self,url,params=None):
        p=dict(params or {})
        if "apiKey=" not in url:
            p["apiKey"]=self.key
        r=requests.get(url,params=p,timeout=90)
        r.raise_for_status()
        return r.json()

    def get(self,path,params=None):
        return self._get_url(self.base+path,params)

    def paginate(self,path,params=None):
        url=self.base+path
        first=True
        while url:
            j=self._get_url(url,params if first else None)
            first=False
            yield j.get("results",[]) or []
            url=j.get("next_url")

    def test_connection(self):
        self.get("/v3/reference/tickers",{"limit":1})
        return True

    def aggregates(self,ticker,start,end):
        j=self.get(
            f"/v2/aggs/ticker/{ticker}/range/1/minute/{start}/{end}",
            {"adjusted":"true","sort":"asc","limit":50000}
        )
        rows=j.get("results",[]) or []
        if not rows:
            return pd.DataFrame()
        d=pd.DataFrame(rows).rename(columns={
            "o":"open","h":"high","l":"low","c":"close","v":"volume",
            "vw":"vwap","n":"transactions","t":"timestamp"
        })
        d["datetime"]=pd.to_datetime(d["timestamp"],unit="ms",utc=True)
        return d

    def news(self,ticker,start_iso,end_iso):
        rows=[]
        params={
            "ticker":ticker,
            "published_utc.gte":start_iso,
            "published_utc.lt":end_iso,
            "sort":"published_utc","order":"asc","limit":1000
        }
        for page in self.paginate("/v2/reference/news",params):
            rows.extend(page)
        out=[]
        for x in rows:
            ins=x.get("insights") or []
            out.append({
                "published_utc":x.get("published_utc"),
                "title":x.get("title"),
                "description":x.get("description"),
                "publisher":(x.get("publisher") or {}).get("name"),
                "tickers":",".join(x.get("tickers") or []),
                "sentiment":",".join(sorted(set(
                    i.get("sentiment","") for i in ins if i.get("sentiment")
                ))),
                "article_url":x.get("article_url")
            })
        return pd.DataFrame(out)


    def earnings(self,ticker,start_date,end_date):
        """
        Massive/Benzinga earnings. If the account has no Benzinga Earnings
        entitlement the caller gets a normal HTTP error which MarketScope
        reports clearly instead of fabricating data.
        """
        rows=[]
        params={
            "ticker":ticker,
            "date.gte":str(start_date),
            "date.lte":str(end_date),
            "limit":50000,
            "sort":"date.asc"
        }
        for page in self.paginate("/benzinga/v1/earnings",params):
            rows.extend(page)
        return pd.DataFrame(rows)

    def option_chain(self,underlying):
        rows=[]
        for page in self.paginate(
            f"/v3/snapshot/options/{underlying}",{"limit":250}
        ):
            rows.extend(page)
        out=[]
        for x in rows:
            d=x.get("details",{}) or {}
            q=x.get("last_quote",{}) or {}
            tr=x.get("last_trade",{}) or {}
            g=x.get("greeks",{}) or {}
            u=x.get("underlying_asset",{}) or {}
            day=x.get("day",{}) or {}
            bid=q.get("bid_price");ask=q.get("ask_price")
            mid=(bid+ask)/2 if bid is not None and ask is not None else tr.get("price",day.get("close"))
            out.append({
                "contract":d.get("ticker"),
                "type":d.get("contract_type"),
                "expiry":d.get("expiration_date"),
                "strike":d.get("strike_price"),
                "bid":bid,"ask":ask,"mid":mid,
                "iv":x.get("implied_volatility"),
                "open_interest":x.get("open_interest"),
                "delta":g.get("delta"),"theta":g.get("theta"),
                "day_volume":day.get("volume"),
                "underlying_price":u.get("price")
            })
        return pd.DataFrame(out)

def test_flat_files():
    c=flat_client()
    c.list_objects_v2(Bucket=flat_bucket(),MaxKeys=1)
    return True
