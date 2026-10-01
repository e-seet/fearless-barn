# Do ADS-B interactions with the tabular features add anything? — pre-registration (bounding check)

Written 2026-10-01, before running. The user asked to try ADS-B inside the tree model. That route
needs a Feb–Jun 2025 download (started), four tree retrains, new correctors and pipeline changes.
The only thing a tree adds over the v46 corrector is **interaction between ADS-B and the other
columns** (stand, runway, operator, aircraft type, off-block features). This check measures that
first.

## Variants (both use the v46 corrector parameters, the 7 airports and covered rows only)

- **A (incumbent):** the v46 inputs: `all_nd` ADS-B features, `base`, airport and hour.
- **B:** A's inputs **plus every tabular column** from `D.load_part` (`with_aobt` featureset; the
  categoricals stay categorical).
- **B_perm (control):** B with only the ADS-B columns shuffled within airport, in both training and
  scoring. It measures what B gets from the tabular columns alone, that is, refitting residuals on
  information the base already had.

## Surface: leave one month out over the 7 months with held-out bases

| month(s) | base |
|---|---|
| Jan, Jul | seasonal_full test |
| Aug | forward_inner val |
| Sep, Oct | forward val |
| Nov, Dec | forward test |

For each of the 7 months:
- train A, B and B_perm on the other 6;
- score all rows of the held-out month; only covered rows at the 7 airports are corrected.

Both variants get identical treatment. Earlier selection on these months, if anything, favours A.

## PASS (B), all of:

1. pooled RMSE over the 7 held-out months: B < A;
2. B < A in at least 5 of the 7 months;
3. B < B_perm pooled.

- **PASS →** B is itself a cheap v47 candidate: no retrains, no Feb–Jun data. The full tree route
  is then reconsidered against it.
- **FAIL →** putting ADS-B inside the tree is very unlikely to help beyond the corrector. Report to
  the user and recommend stopping that route.

**Cut-off:** any v47 must be built by 2026-10-06, leaving one upload and a fallback re-upload before
the 9 October freeze.
