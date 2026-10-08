# LIRF no-record specialist, hybrid form — pre-registration

Written 2026-10-08, after the plain specialist failed (`RESULT.md`: −14.0 pooled, but +2.5 with the best 20
rows removed), and before any hybrid number is computed. User decision: "yes test the hybrid specialist too".

**Hybrid H.** On LIRF departures without an NM record, use the specialist S (unchanged from
`PREREGISTERED.md`) only where the departure took off **more than 6 h after its scheduled time**
(g = MVT − SCHED > 21,600 s). Everywhere else keep the reference.

**Why 6 h, fixed in advance.** 6 h is the cut-off two published solutions use for the same population:
outgoing-boat's `--nm-min-ms 21600` line rule, and joris-decombe's "past six hours late". It was not tuned
on our results. Other cut-offs may be printed for information; they do not decide anything.

**Two surfaces, both required:**
1. **The 7 months (Jan, Jul–Dec 2025)**, same S and same reference as the failed test. This surface has
   now been seen, so it cannot pass the hybrid on its own.
2. **Held-out Feb–Jun 2025** (`tests_final/months_of_concern.py`), never used to judge S. The reference
   there is the shipped design (base → stage 2 → LFPG corrector).

**PASS (on each surface):**
- pooled RMSE better;
- still better with the 20 largest-gain rows removed;
- better in at least 5/7 months (7-month surface), or in at least 2/3 of the changed months (Feb–Jun).

**Build if PASS on both:** H goes into the final design (a stage after stage 5, before the LOBT clip if
that is adopted), in v64 for the board and in the final file. It is documented with credit to
quirky-honey and zestful-fountain.
