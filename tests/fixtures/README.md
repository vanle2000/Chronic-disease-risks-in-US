# Test fixtures

| File | Provenance |
|---|---|
| `cdi_sample.json` | **Recorded.** Six real rows from `https://data.cdc.gov/resource/hksd-2xuw.json`, `yearstart=2021`, ordered by `:id`, captured 2026-09-03. |
| `acs_sample.json` | **Synthetic.** Built to the documented ACS array-of-arrays contract (header row, then value rows). Not recorded, because the Census API requires a key and CI has none. Values are illustrative and are not used for any assertion about population size — only about response *shape*. |

The distinction matters. `cdi_sample.json` can be trusted as evidence of what the
API actually returns, so `TestCdiFixtureShape` uses it to guard the assumptions
the pipeline is built on: lowercase field names, GeoJSON `geolocation`, and the
presence of stable `*id` columns.

`acs_sample.json` proves only that the parser handles the documented shape. The
first real run against a keyed endpoint is what will confirm the contract, and
this fixture should be replaced with a recorded response at that point.

To re-record the CDI fixture:

```bash
curl -s "https://data.cdc.gov/resource/hksd-2xuw.json?\$where=yearstart=2021&\$order=:id&\$limit=6" \
  -o tests/fixtures/cdi_sample.json
```
