from io import StringIO
import requests,pandas as pd
FRED_3M_URL="https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS3MO"
def fetch_risk_free(start,end):
    r=requests.get(FRED_3M_URL,params={"cosd":str(start),"coed":str(end)},timeout=30,headers={"User-Agent":"OptionEdge/15.0"})
    r.raise_for_status()
    d=pd.read_csv(StringIO(r.text))
    dc="DATE" if "DATE" in d.columns else d.columns[0];vc="DGS3MO" if "DGS3MO" in d.columns else d.columns[-1]
    d=d.rename(columns={dc:"date",vc:"rate_percent"})
    d["date"]=pd.to_datetime(d["date"],errors="coerce")
    d["rate_percent"]=pd.to_numeric(d["rate_percent"],errors="coerce")
    d=d.dropna(subset=["date"]).sort_values("date");d["rate_percent"]=d["rate_percent"].ffill()
    d["risk_free_rate"]=d["rate_percent"]/100.0
    return d[["date","rate_percent","risk_free_rate"]]
