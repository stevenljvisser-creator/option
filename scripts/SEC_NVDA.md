SEC-only NVDA acceptance test on a GitHub-hosted Actions runner. Manually start
`sec-nvda.yml`. No API key, FMP, database or Hetzner S3 configuration is needed. SEC_CONTACT_EMAIL must contain a real contact email.

The worker fetches official SEC Company Facts/XBRL and submissions for NVIDIA
(CIK 0001045810). It declares its project identity in the User-Agent, limits
requests to at most one per second and retries 403/429/5xx/network failures with
bounded exponential backoff. The User-Agent is `OptionEdge stevenljvisser-creator/option <email>`, constructed from SEC_CONTACT_EMAIL (an existing Actions secret or repository variable). The email is never hardcoded. Missing/placeholder emails fail before network requests. Full response status and headers are logged for each SEC HTTP error.

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

Configure contact securely without putting it in source code:

```powershell
gh secret set SEC_CONTACT_EMAIL --repo stevenljvisser-creator/option
```

This command prompts for the value locally. Then manually dispatch the workflow.
