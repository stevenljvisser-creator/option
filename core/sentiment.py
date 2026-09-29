import re
import pandas as pd

POS={"beat","beats","growth","record","strong","surge","surges","upgrade","upgraded","raises","raised","outperform","profit","profits","approval","approved","win","wins","positive","higher","expands","expansion","bullish","rebound","exceeds"}
NEG={"miss","misses","weak","decline","declines","downgrade","downgraded","cuts","cut","lawsuit","probe","investigation","recall","loss","losses","lower","warning","negative","bearish","fraud","halt","layoff","layoffs","risk","falls"}
SURPRISE={"unexpected","surprise","surprises","beat","beats","miss","misses","raises","cuts","halt","plunge","surge","shock","record","warning","exceeds"}
EVENTS={
    "earnings":["earnings","eps","quarter","quarterly results","revenue","profit"],
    "guidance":["guidance","outlook","forecast","expects","raises forecast","cuts forecast"],
    "analyst":["analyst","upgrade","downgrade","price target","outperform","underperform"],
    "product":["launch","product","chip","drug","trial","approval","contract","order"],
    "legal_regulatory":["lawsuit","court","regulator","regulatory","sec ","doj","ftc","probe","investigation"],
    "mna":["acquisition","acquire","merger","takeover","buyout"],
    "management":["ceo","cfo","management","resigns","resignation","appoints","appointed"],
    "macro":["fed","inflation","interest rate","jobs report","gdp","tariff","treasury","recession"],
}

def score_text(text):
    words=re.findall(r"[A-Za-z]+",str(text).lower())
    if not words:return 0.0,0.0,0.0
    p=sum(w in POS for w in words);n=sum(w in NEG for w in words)
    raw=(p-n)/max(1,p+n)
    emotion=min(1.0,(p+n)/6.0+str(text).count("!")*.08)
    surprise=min(1.0,sum(w in SURPRISE for w in words)/3.0)
    return float(raw),float(emotion),float(surprise)

def classify_event(text):
    x=str(text or "").lower()
    for cat,terms in EVENTS.items():
        if any(term in x for term in terms):return cat
    return "other"

def enrich_news(df):
    if df.empty:return df
    d=df.copy();scores=[];emotions=[];surprises=[];cats=[]
    for _,r in d.iterrows():
        text=" ".join(str(r.get(c,"") or "") for c in [
            "headline","title","article_title","description","article_description",
            "article_text","google_context"
        ])
        s,e,u=score_text(text)
        provider=str(r.get("sentiment","")).lower()
        if "positive" in provider:s=max(s,.65)
        elif "negative" in provider:s=min(s,-.65)
        elif "neutral" in provider and abs(s)<.25:s=0.0
        scores.append(s);emotions.append(e);surprises.append(u);cats.append(classify_event(text))
    d["sentiment_score"]=scores;d["emotion_intensity"]=emotions;d["surprise_score"]=surprises
    d["event_category"]=cats;d["relevance_score"]=1.0
    return d
