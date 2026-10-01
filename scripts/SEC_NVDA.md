SEC-only NVDA acceptance test on a GitHub-hosted Actions runner. Manually start
`sec-nvda.yml`. No API key, FMP, database or Hetzner S3 configuration is needed.

The worker fetches official SEC Company Facts/XBRL and submissions for NVIDIA
(CIK 0001045810). It declares its project identity in the User-Agent, limits
requests to at most one per second and retries 403/429/5xx/network failures with
bounded exponential backoff. A custom contact User-Agent can be supplied at dispatch.

The artifact includes NVDA-quarterly.parquet, both raw SEC responses and summary.json.
Parquet keeps source concepts, values, units, reporting periods, filings,
accession IDs and availability evidence. It retains revisions, excludes annual/
YTD duration facts from quarter observations, and never derives Q4 EPS by
subtracting annual EPS. Instantaneous balance-sheet facts are labeled separately.
The default research boundary remains 2022-09-16. Missing directly reported
quarter values remain missing; this first test does not fabricate all quarters.

Source FY/FP fields describe the reporting filing and may refer to a later year
for comparative disclosures; period_start/end identify the actual observation.

```powershell
gh workflow run sec-nvda.yml --repo stevenljvisser-creator/option --ref main
gh run list --repo stevenljvisser-creator/option --workflow sec-nvda.yml
gh run view RUN_ID --repo stevenljvisser-creator/option --log
gh run download RUN_ID --repo stevenljvisser-creator/option --dir .\sec-nvda
```

The test does not deploy or change any other OptionEdge functionality. FMP and
Hetzner S3 integration are a later step after this test succeeds.
