# GTA VI Twitch CCU Forecaster

![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)
[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01/0xDimzy/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01/0xDimzy/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecasts how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (2026-11-19 UTC), given the first hour's count — five
quantiles (5%, 25%, 50%, 75%, 95%) for each of hours 2 to 24.

## Method

`main.py` (Python 3.13, standard library only, no network) fits its model at
run time from the case file itself — nothing is hard-coded to any launch:

1. It keeps only the comparables that unlocked at midnight local time,
   rolling across zones (the way console launches, GTA VI included, unlock),
   dropping simultaneous-global and early-access-contaminated launches,
   whose release-day shape is different.
2. For each kept launch and each hour it forms the log ratio of that hour's
   viewers to the launch's own hour-1 print.
3. Per hour it regresses that log ratio on log hour-1 viewers: bigger
   launches ramp less in relative terms. The case's hour-1 print is clamped
   to the range the kept launches span, so a launch bigger than anything
   observed takes the shape of the biggest observed launches instead of
   extrapolating a fitted slope past the data.
4. Quantiles come from the regression residuals, with the spread shrunk
   x0.8 (the raw residual spread is wider than pinball-optimal on the
   practice cases).

Hours with too few usable launches fall back to persistence with a
random-walk-widening spread. The program is deterministic: two runs on the
same case write the same bytes.

## Practice check (honest, no memorised answers)

Scored against the practice cases' SullyGnome hourly counts (the kit's own
source), this model averages a pinball loss of ~0.164 over the four practice
launches, against ~0.387 for the house baseline (persistence). Per launch:
Wolverine 0.194, Red Dead Redemption 2 0.118, God of War Ragnarök 0.117,
Spider-Man 2 0.229.

## Run

The grader runs `python3 -P -s /bot/main.py` with the case mounted at
`/input/<id>.in` and writes `/output/<id>.out`. Locally:

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see `LICENSE`.
