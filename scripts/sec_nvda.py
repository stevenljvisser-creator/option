"""SEC-only NVDA acceptance test. No Hetzner, database, FMP or API key needed."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, logging, os, time
from pathlib import Path
import urllib.error, urllib.request
from zoneinfo import ZoneInfo

CIK='0001045810'
DEFAULT_AGENT='OptionEdge SEC Research/1.0 stevenljvisser-creator https://github.com/stevenljvisser-creator/option'
CONCEPTS={
    'revenue':('RevenueFromContractWithCustomerExcludingAssessedTax','Revenues','SalesRevenueNet'),
    'net_income':('NetIncomeLoss','ProfitLoss'),
    'eps_diluted':('EarningsPerShareDiluted',),
    'eps_basic':('EarningsPerShareBasic',),
    'operating_cash_flow':('NetCashProvidedByUsedInOperatingActivities',),
    'assets':('Assets',), 'liabilities':('Liabilities',),
    'equity':('StockholdersEquity','StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'),
}
INSTANT={'assets','liabilities','equity'}
LOG=logging.getLogger('sec-nvda')

class SecClient:
    def __init__(self,agent,interval=1.,attempts=5):
        if not agent.strip():raise ValueError('A declared SEC User-Agent is required')
        self.agent=agent;self.interval=interval;self.attempts=attempts;self.last=0.
    def fetch(self,url):
        if not url.startswith('https://data.sec.gov/'):raise ValueError('Only official SEC data endpoints are allowed')
        for attempt in range(self.attempts):
            time.sleep(max(0.,self.interval-(time.monotonic()-self.last)))
            self.last=time.monotonic()
            req=urllib.request.Request(url,headers={'User-Agent':self.agent,'Accept':'application/json'})
            try:
                with urllib.request.urlopen(req,timeout=90) as response:
                    raw=response.read();payload=json.loads(raw)
                    LOG.info('SEC HTTP %s: %s (%s bytes)',response.status,url,len(raw))
                    return payload,raw
            except urllib.error.HTTPError as exc:
                if exc.code not in (403,429,500,502,503,504) or attempt==self.attempts-1:raise
                try:delay=min(120.,max(2.**(attempt+1),float(exc.headers.get('Retry-After','0'))))
                except ValueError:delay=2.**(attempt+1)
                LOG.warning('SEC HTTP %s; retry %s/%s after %.1fs',exc.code,attempt+1,self.attempts,delay)
                time.sleep(delay)
            except (urllib.error.URLError,TimeoutError):
                if attempt==self.attempts-1:raise
                time.sleep(2.**(attempt+1))
        raise RuntimeError('SEC retry limit reached')


def acceptance_map(submissions):
    recent=(submissions.get('filings') or {}).get('recent') or {}
    return {acc:stamp for acc,stamp in zip(recent.get('accessionNumber',[]),recent.get('acceptanceDateTime',[])) if stamp}


def records(facts,submissions,start,observed):
    if str(facts.get('cik')).lstrip('0')!=CIK.lstrip('0'):raise ValueError('Company Facts CIK mismatch')
    if 'NVDA' not in submissions.get('tickers',[]):raise ValueError('Submissions ticker mismatch')
    accepted=acceptance_map(submissions);gaap=(facts.get('facts') or {}).get('us-gaap') or {}
    rows=[];seen=set()
    for metric,names in CONCEPTS.items():
        for concept in names:
            for unit,observations in (gaap.get(concept,{}).get('units') or {}).items():
                for fact in observations:
                    if fact.get('form') not in ('10-Q','10-Q/A','10-K','10-K/A'):continue
                    end=dt.date.fromisoformat(fact['end'])
                    if end<start or end>dt.date.fromisoformat(observed[:10]):continue
                    begin=dt.date.fromisoformat(fact['start']) if fact.get('start') else None
                    duration=(end-begin).days if begin else None
                    if metric in INSTANT:
                        if begin is not None:continue
                        period_type='instant'
                    else:
                        # Exact quarter durations only. Never call YTD/annual EPS quarterly.
                        if duration is None or not 60<=duration<=120:continue
                        period_type='quarter'
                    filed=fact.get('filed');accession=fact.get('accn')
                    if not filed or not accession:continue
                    available=accepted.get(accession)
                    basis='sec_acceptance_datetime'
                    if not available:
                        conservative=dt.datetime.combine(dt.date.fromisoformat(filed)+dt.timedelta(days=1),dt.time(),tzinfo=ZoneInfo('America/New_York'))
                        available=conservative.astimezone(dt.timezone.utc).isoformat();basis='after_filing_day_conservative'
                    identity=(concept,unit,fact.get('start'),fact['end'],accession,filed,fact['val'])
                    if identity in seen:continue
                    seen.add(identity)
                    rows.append({'ticker':'NVDA','cik':CIK,'metric':metric,'taxonomy':'us-gaap','concept':concept,'unit':unit,
                        'period_type':period_type,'period_start':fact.get('start'),'period_end':fact['end'],
                        'duration_days':duration,'value':float(fact['val']),
                        # A comparative fact can be reported in a later fiscal-year filing.
                        # Keep official filing FY/FP, do not mislabel the underlying period.
                        'reported_fiscal_year':fact.get('fy'),'reported_fiscal_period':fact.get('fp'),
                        'form':fact['form'],'filing_date':filed,'accession_number':accession,'frame':fact.get('frame'),
                        'available_utc':available,'availability_basis':basis,'observed_at_utc':observed})
    return sorted(rows,key=lambda r:(r['period_end'],r['metric'],r['filing_date'],r['accession_number']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('sec-nvda-output'))
    p.add_argument('--start',type=dt.date.fromisoformat,default=dt.date(2022,9,16))
    p.add_argument('--user-agent',default=os.environ.get('SEC_USER_AGENT',DEFAULT_AGENT));a=p.parse_args()
    if os.environ.get('GITHUB_ACTIONS')!='true' or os.environ.get('RUNNER_ENVIRONMENT')!='github-hosted':
        raise RuntimeError('Real SEC requests are restricted to a GitHub-hosted Actions runner, outside Hetzner')
    import pyarrow as pa
    import pyarrow.parquet as pq
    a.output.mkdir(parents=True,exist_ok=True)
    client=SecClient(a.user_agent)
    facts_url=f'https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json'
    submissions_url=f'https://data.sec.gov/submissions/CIK{CIK}.json'
    facts,facts_raw=client.fetch(facts_url);submissions,submissions_raw=client.fetch(submissions_url)
    observed=dt.datetime.now(dt.timezone.utc).isoformat()
    (a.output/'NVDA-companyfacts.json').write_bytes(facts_raw)
    (a.output/'NVDA-submissions.json').write_bytes(submissions_raw)
    rows=records(facts,submissions,a.start,observed)
    quarters=[r for r in rows if r['period_type']=='quarter']
    metrics={r['metric'] for r in quarters}
    if not quarters or not {'revenue','net_income','eps_diluted'}.issubset(metrics):
        raise RuntimeError('NVDA quarter validation failed: no revenue/net income/diluted EPS')
    table=pa.Table.from_pylist(rows);parquet=a.output/'NVDA-quarterly.parquet'
    pq.write_table(table,parquet,compression='zstd')
    loaded=pq.read_table(parquet)
    if not table.equals(loaded):raise RuntimeError('Parquet roundtrip changed source data')
    if any(r['ticker']!='NVDA' for r in loaded.to_pylist()):raise RuntimeError('Parquet ticker validation failed')
    summary={'status':'success','ticker':'NVDA','cik':CIK,'source':facts_url,'runner':'GitHub-hosted Actions',
        'github_run_id':os.environ.get('GITHUB_RUN_ID'),'github_sha':os.environ.get('GITHUB_SHA'),
        'observed_at_utc':observed,'start_date':str(a.start),'row_count':len(rows),
        'quarter_fact_rows':len(quarters),'unique_quarter_periods':len({(r['period_start'],r['period_end']) for r in quarters}),
        'metrics':sorted({r['metric'] for r in rows}),
        'first_period_end':min(r['period_end'] for r in quarters),'last_period_end':max(r['period_end'] for r in quarters),
        'parquet_file':parquet.name,'parquet_sha256':hashlib.sha256(parquet.read_bytes()).hexdigest(),
        'companyfacts_sha256':hashlib.sha256(facts_raw).hexdigest(),'parquet_roundtrip_verified':True,
        'data_notes':['Source revisions and accession IDs are retained.',
            'Reported FY/FP refer to the filing, including comparative facts; use period_start/end for the observation.',
            'Only directly reported 60-120 day duration facts are called quarterly. YTD/annual values are excluded; annual EPS is never subtracted into Q4.',
            'SEC-only test: no FMP, database, S3 or OptionEdge deployment changes.']}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    LOG.info('NVDA quarter data fetched successfully: %s facts, %s quarter periods; Parquet readback passed',len(rows),summary['unique_quarter_periods'])
    print(json.dumps(summary,indent=2))
    step_summary=os.environ.get('GITHUB_STEP_SUMMARY')
    if step_summary:
        with open(step_summary,'a') as f:f.write(f"## SEC NVDA acceptance test\n\nSuccess: **{len(rows)} facts**, **{summary['unique_quarter_periods']} quarter periods**.\n\nParquet SHA256: `{summary['parquet_sha256']}`. Readback verified.\n")

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s');main()
