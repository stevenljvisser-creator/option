# Eerste optimalisatie: opslaginstellingen per featurejob

Checkpoint vóór functionele wijzigingen: `689d4b2` (baseline applicatiecode nog gelijk aan `205bffc`).

Gewijzigd: core/runtime_settings.py leest uitsluitend vijf storagekeys via één SELECT; core/storage.py biedt een immutable jobcontext via ContextVar; worker/tasks.py bindt die context in hoofdthread en expliciet in iedere executor-taak. Andere callers behouden individuele live settingslezingen. Clientpool, requestinhoud, datasets, featureberekeningen en outputkeys zijn ongewijzigd. Geen dependencies toegevoegd, geen productie aangeraakt.

Instellingenrotatie tijdens een featurejob wordt pas bij de volgende job toegepast. Een lopende job blijft de oorspronkelijke bucket/credentials gebruiken; bij intrekking van credentials kunnen requests falen volgens het bestaande retry-/exceptionpad. Geen automatische credentials-refresh midden in een job, omdat dat verschillende buckets binnen één output zou kunnen mengen. Access/secret zijn uitgesloten van de dataclass-repr. Niet als persistente metadata opslaan.

Verificatie:
- `python3 analysis/clean-feature-store/test_storage_snapshot.py -v`: 6 geslaagde isolatietests, inclusief echte run_features/_run_features-bron met mocked I/O/DB; geen applicatie-imports of netwerkverbindingen.
- `python3 analysis/clean-feature-store/benchmark_snapshot.py`: 40.000 mockrequests, van 240.000 individuele settingsreads naar één snapshotread; selectieve query is afzonderlijk getest. Requesttrace SHA256 identiek: `2dbdf8bf58d8b4691efdb9a30d6248334d8fcd01038d303dedf43814d9860280`.
- Mock wrapperruntime .1664 → .1164 seconden (~1,43× in deze run). Dit is geen werkelijke DB/S3- of featurestore-speedup; timing omvat geen echte latencies en is gevoelig voor scheduling.
- Syntaxcompile van de drie gewijzigde applicatiebestanden en git diff --check geslaagd.
- Baseline call-count-script blijft uitvoerbaar; historische JSON blijft behouden.

Grenzen: pandas, boto3 en SQLAlchemy-integratieruntime zijn lokaal niet beschikbaar; echte SELECTlatency, recordthroughput en feature-outputhashgelijkheid zijn nog niet gemeten. De isolatietests gebruiken de daadwerkelijke bronfuncties, maar vervangen dependencies. Geen conclusie dat de volledige rebuild nu minuten of uren duurt. Volgende stap is een complete bestaande tickerdag in een geïsoleerde ontwikkelomgeving, met read-only inputs en outputs in een aparte lokale directory; geen volledige rebuild.

Rollback: `git revert <commit van deze optimalisatie>` maakt de drie applicatiebestanden weer functioneel gelijk aan checkpoint 689d4b2. Geen datamigratie, objectwrites of infrastructuurwijzigingen uitgevoerd. Bestaande baseline en analyse blijven behouden. Geen andere performancevoorstellen geïmplementeerd.
