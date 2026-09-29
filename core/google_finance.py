from __future__ import annotations
from datetime import datetime,timezone,timedelta
from urllib.parse import urljoin,urlparse
import json,re,hashlib
import requests
from bs4 import BeautifulSoup
import pandas as pd

USER_AGENT=(
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

# Explicit mapping for the MarketScope training universe.
EXCHANGES={
    "NVDA":"NASDAQ","AMD":"NASDAQ","AVGO":"NASDAQ","AAPL":"NASDAQ",
    "MSFT":"NASDAQ","GOOGL":"NASDAQ","GOOG":"NASDAQ","META":"NASDAQ",
    "AMZN":"NASDAQ","TSLA":"NASDAQ","COST":"NASDAQ","QQQ":"NASDAQ",
    "JPM":"NYSE","BAC":"NYSE","GS":"NYSE","LLY":"NYSE","UNH":"NYSE",
    "XOM":"NYSE","CVX":"NYSE","CAT":"NYSE","BA":"NYSE","WMT":"NYSE",
    "SPY":"NYSEARCA","PLTR":"NASDAQ","COIN":"NASDAQ",
}

SKIP_DOMAINS={
    "google.com","www.google.com","accounts.google.com","support.google.com",
    "policies.google.com","encrypted-tbn0.gstatic.com","encrypted-tbn1.gstatic.com",
    "encrypted-tbn2.gstatic.com","encrypted-tbn3.gstatic.com"
}

def utcnow():
    return datetime.now(timezone.utc)

def exchange_for(ticker):
    return EXCHANGES.get(str(ticker).upper(),"NASDAQ")

def finance_url(ticker):
    t=str(ticker).upper()
    return f"https://www.google.com/finance/quote/{t}:{exchange_for(t)}?hl=en"

def _clean(s):
    return re.sub(r"\s+"," ",str(s or "")).strip()

def _domain(url):
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""

def _parse_relative_time(text,now=None):
    now=now or utcnow()
    x=_clean(text).lower()
    m=re.search(r"(\d+)\s*(minute|min|minutes|mins)\s+ago",x)
    if m:return now-timedelta(minutes=int(m.group(1)))
    m=re.search(r"(\d+)\s*(hour|hours|hr|hrs)\s+ago",x)
    if m:return now-timedelta(hours=int(m.group(1)))
    m=re.search(r"(\d+)\s*(day|days)\s+ago",x)
    if m:return now-timedelta(days=int(m.group(1)))
    if "yesterday" in x:return now-timedelta(days=1)
    return None

def _next_value(lines,labels):
    norm=[_clean(x) for x in lines]
    low=[x.lower() for x in norm]
    labels=[x.lower() for x in labels]
    for i,x in enumerate(low):
        if x in labels:
            for j in range(i+1,min(i+4,len(norm))):
                v=norm[j]
                if v and v.lower() not in labels:
                    return v
    return None

class GoogleFinanceClient:
    def __init__(self,timeout=25):
        self.timeout=timeout
        self.session=requests.Session()
        self.session.headers.update({
            "User-Agent":USER_AGENT,
            "Accept-Language":"en-US,en;q=0.9",
        })

    def get_page(self,ticker):
        url=finance_url(ticker)
        r=self.session.get(url,timeout=self.timeout)
        r.raise_for_status()
        if len(r.text)<5000:
            raise RuntimeError("Google Finance returned an unexpectedly small page.")
        return r.text,url

    def parse(self,ticker,html,url=None):
        now=utcnow()
        soup=BeautifulSoup(html,"html.parser")
        lines=[_clean(x) for x in soup.stripped_strings if _clean(x)]
        text="\n".join(lines)

        snapshot={
            "ticker":str(ticker).upper(),
            "exchange":exchange_for(ticker),
            "captured_utc":now.isoformat(),
            "google_finance_url":url or finance_url(ticker),
            "price":_next_value(lines,[str(ticker).upper(),"Price"]),
            "open":_next_value(lines,["Open"]),
            "high":_next_value(lines,["High"]),
            "low":_next_value(lines,["Low"]),
            "market_cap":_next_value(lines,["Market cap"]),
            "avg_volume":_next_value(lines,["Avg Volume","Average volume"]),
            "volume":_next_value(lines,["Volume"]),
            "pe_ratio":_next_value(lines,["P/E ratio"]),
            "dividend_yield":_next_value(lines,["Dividend yield"]),
            "primary_exchange":_next_value(lines,["Primary exchange"]),
            "52_week_high":_next_value(lines,["52-week high"]),
            "52_week_low":_next_value(lines,["52-week low"]),
            "eps":_next_value(lines,["EPS"]),
            "beta":_next_value(lines,["Beta"]),
            "shares_outstanding":_next_value(lines,["Shares outstanding"]),
            "employees":_next_value(lines,["Employees"]),
            "latest_report":_next_value(lines,["Latest report"]),
            "fiscal_period":_next_value(lines,["Fiscal period"]),
            "eps_vs_estimate":_next_value(lines,["EPS vs estimate"]),
            "revenue_vs_estimate":_next_value(lines,["Revenue vs estimate"]),
        }

        # Google Finance page anchors point at publisher stories. Avoid brittle CSS classes.
        stories=[]
        seen=set()
        for a in soup.find_all("a",href=True):
            title=_clean(a.get_text(" ",strip=True))
            href=urljoin("https://www.google.com",a.get("href"))
            dom=_domain(href)
            if not title or len(title)<18 or len(title)>350:
                continue
            if not href.startswith("http"):
                continue
            if dom in SKIP_DOMAINS or dom.endswith("gstatic.com"):
                continue
            if href in seen:
                continue
            # Find nearby source/time in a small parent container.
            parent=a
            context=""
            for _ in range(4):
                if parent.parent is None:break
                parent=parent.parent
                context=_clean(parent.get_text(" ",strip=True))
                if len(context)>len(title)+10:
                    break
            published=_parse_relative_time(context,now)
            source=dom
            # Search nearby text for a likely source label.
            bits=[_clean(x) for x in parent.stripped_strings] if parent else []
            for bit in bits:
                if bit==title:continue
                if 2<=len(bit)<=80 and not re.search(r"\d+\s+(minute|hour|day)",bit.lower()):
                    if "." in bit or bit.lower() in ["reuters","benzinga","barron's","yahoo finance","seeking alpha","cnbc"]:
                        source=bit
                        break
            stories.append({
                "ticker":str(ticker).upper(),
                "captured_utc":now.isoformat(),
                "published_utc":published.isoformat() if published else None,
                "source":source,
                "headline":title,
                "article_url":href,
                "google_finance_url":url or finance_url(ticker),
                "google_context":context[:800],
            })
            seen.add(href)

        # Keep likely finance-news links and avoid huge unrelated link lists.
        return snapshot,pd.DataFrame(stories[:60])

    def fetch(self,ticker):
        html,url=self.get_page(ticker)
        return self.parse(ticker,html,url)

class ArticleEnricher:
    def __init__(self,timeout=20):
        self.timeout=timeout
        self.session=requests.Session()
        self.session.headers.update({
            "User-Agent":USER_AGENT,
            "Accept-Language":"en-US,en;q=0.9",
        })

    def _jsonld(self,soup):
        items=[]
        for s in soup.find_all("script",type="application/ld+json"):
            try:
                x=json.loads(s.string or s.get_text() or "{}")
                items.extend(x if isinstance(x,list) else [x])
            except Exception:
                continue
        flat=[]
        def walk(x):
            if isinstance(x,dict):
                flat.append(x)
                for v in x.values():walk(v)
            elif isinstance(x,list):
                for v in x:walk(v)
        for x in items:walk(x)
        return flat

    def enrich_one(self,row):
        url=str(row.get("article_url") or "")
        base=dict(row)
        base.update({
            "article_title":None,"article_description":None,
            "article_text":None,"article_word_count":0,
            "article_published_utc":None,"author":None,
            "source_domain":_domain(url),"enrichment_status":"headline_only"
        })
        if not url.startswith("http"):
            return base
        try:
            r=self.session.get(url,timeout=self.timeout,allow_redirects=True)
            r.raise_for_status()
            if "text/html" not in (r.headers.get("content-type") or "").lower():
                return base
            soup=BeautifulSoup(r.text,"html.parser")

            def meta(*keys):
                for key in keys:
                    m=soup.find("meta",attrs={"property":key}) or soup.find("meta",attrs={"name":key})
                    if m and m.get("content"):
                        return _clean(m.get("content"))
                return None

            title=meta("og:title","twitter:title")
            if not title and soup.title:
                title=_clean(soup.title.get_text(" ",strip=True))
            desc=meta("og:description","description","twitter:description")
            published=meta("article:published_time","datePublished")
            author=meta("author","article:author")

            article_body=None
            for item in self._jsonld(soup):
                typ=str(item.get("@type","")).lower()
                if "article" in typ or "newsarticle" in typ:
                    title=title or _clean(item.get("headline"))
                    desc=desc or _clean(item.get("description"))
                    published=published or _clean(item.get("datePublished"))
                    av=item.get("author")
                    if not author:
                        if isinstance(av,dict):author=_clean(av.get("name"))
                        elif isinstance(av,list):
                            author=", ".join(_clean(x.get("name")) for x in av if isinstance(x,dict) and x.get("name"))
                    body=item.get("articleBody")
                    if body and len(_clean(body))>200:
                        article_body=_clean(body)
                        break

            if not article_body:
                paras=[]
                for p in soup.find_all("p"):
                    txt=_clean(p.get_text(" ",strip=True))
                    if len(txt)>=50:
                        paras.append(txt)
                    if sum(len(x) for x in paras)>=12000:
                        break
                article_body="\n".join(paras)[:12000] if paras else None

            base.update({
                "article_title":title,
                "article_description":desc,
                "article_text":article_body,
                "article_word_count":len((article_body or "").split()),
                "article_published_utc":published,
                "author":author,
                "source_domain":_domain(r.url) or _domain(url),
                "resolved_url":r.url,
                "enrichment_status":"enriched" if (article_body or desc) else "headline_only"
            })
        except Exception as e:
            base["enrichment_status"]=f"failed:{type(e).__name__}"
        return base

    def enrich_frame(self,df,max_articles=40):
        if df is None or df.empty:
            return pd.DataFrame()
        rows=[]
        for _,r in df.head(max_articles).iterrows():
            rows.append(self.enrich_one(r.to_dict()))
        return pd.DataFrame(rows)
