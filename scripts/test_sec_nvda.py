import unittest
from sec_nvda import records,CIK,make_user_agent

class QuarterTests(unittest.TestCase):
    def test_contact_is_configurable_and_placeholder_is_rejected(self):
        self.assertEqual(make_user_agent('research@example.org'),'OptionEdge stevenljvisser-creator/option research@example.org')
        for invalid in (None,'','<MIJN_EMAILADRES>','x@example.org\r\nInjected: x'):
            with self.assertRaises(ValueError):make_user_agent(invalid)
    def payload(self):
        base={'end':'2025-04-27','filed':'2025-05-28','form':'10-Q','accn':'abc','fy':2026,'fp':'Q1'}
        facts={'cik':int(CIK),'facts':{'us-gaap':{'EarningsPerShareDiluted':{'units':{'USD/shares':[
            {**base,'start':'2025-01-27','val':.95},
            {**base,'start':'2024-01-28','val':3.95},
            {**base,'start':'2025-01-27','val':.95},
        ]}},'Assets':{'units':{'USD':[{**base,'val':140000000000}]}}}}}
        submissions={'tickers':['NVDA'],'filings':{'recent':{'accessionNumber':['abc'],'acceptanceDateTime':['2025-05-28T20:15:00Z']}}}
        return facts,submissions
    def test_excludes_annual_eps_and_exact_duplicates(self):
        rows=records(*self.payload(),__import__('datetime').date(2022,9,16),'2026-10-01T00:00:00Z')
        self.assertEqual(len(rows),2);self.assertEqual(next(r for r in rows if r['metric']=='eps_diluted')['value'],.95)
    def test_preserves_acceptance_timestamp(self):
        rows=records(*self.payload(),__import__('datetime').date(2022,9,16),'2026-10-01T00:00:00Z')
        self.assertEqual(rows[0]['available_utc'],'2025-05-28T20:15:00Z')
    def test_wrong_company_fails_closed(self):
        facts,sub=self.payload();facts['cik']=1
        with self.assertRaises(ValueError):records(facts,sub,__import__('datetime').date(2022,9,16),'2026-10-01T00:00:00Z')

if __name__=='__main__':unittest.main()
