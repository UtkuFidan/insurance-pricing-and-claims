# Claims reserving in plain language

The problem. An insurer sells a policy today and pays claims later, sometimes years later. At any
balance-sheet date it owes money for accidents that have already happened but are not fully paid, or
not even reported yet. The estimate of that unpaid amount is the reserve. The part for accidents
not yet reported (or not yet fully developed) is called IBNR (incurred but not reported).
Under-reserving overstates profit and can threaten solvency; over-reserving ties up capital.

The triangle. Claims are grouped by the year the accident happened (the *origin year*, rows) and
by how long after that they are measured (the *development age*, columns). Only the upper-left
triangle is observed, because recent accident years have had less time to develop.

```
            12m    24m    36m   ...
origin 2001  x      x      x         fully observed
origin 2002  x      x      ?         ? = future, to be estimated
origin 2003  x      ?      ?
```

Chain-ladder. Look at how claims grew from one age to the next in past years (the *development
factor*, e.g. 24m / 12m). Multiply each year's latest figure by the factors still to come to get the
*ultimate* claims. Reserve (IBNR) = ultimate - what is already recorded. It assumes the past
development pattern repeats.

Mack standard error. The chain-ladder gives one number. Mack's formula gives a *standard error*
around it, built from how much the observed factors vary. It tells you how uncertain that single
number is, without any simulation.

Bornhuetter-Ferguson (BF). For the newest years the triangle has very little data, so the
chain-ladder can swing wildly. BF blends in an outside *a-priori* view (premium x expected loss
ratio) for the part not yet emerged. Young years lean on the prior; mature years lean on the data.

Bootstrap ODP. Re-draw the triangle's residuals thousands of times, re-run the chain-ladder on
each draw, and add random process noise. The spread of the resulting totals is a reserve range:
the median is a central estimate, while the 75th or 95th percentile shows how much extra margin
would be needed for a given confidence level.

What none of this covers. Model risk (the past may not repeat), inflation shifts, changes in
claim handling, large-loss events, and reinsurance recoveries. A reserve range from a single triangle
is a starting point for an actuary's judgment, not a replacement for it.
