clear all
set more off

import delimited "boeing_returns_data.csv", clear

capture rename boeing r_boeing
capture rename market r_market

gen statadate = date(date, "YMD")
format statadate %td
tsset statadate

* ------------------------------------------------------------------
* FIX: the original do-file marked event windows using calendar-date
* cutoffs (statadate >= X & statadate <= Y). Because trading calendars
* skip weekends unevenly, a calendar-date cutoff does not reliably
* land on the same OFFSET trading day for every event. Checking each
* event by hand against this dataset's trading days shows 5 of 6
* windows below picked up an extra (8th) trading day -- i.e. they
* actually spanned [-1,+6], not the [-1,+5] described in the paper.
*
* This version builds each dummy from the trading-day RANK of the
* event date (same logic already used correctly in the Python
* scripts' `event_idx + offset` indexing), guaranteeing exactly 7
* trading days ([-1,+5]) per event regardless of weekends/holidays.
* ------------------------------------------------------------------

gen long obs_n = _n
sort statadate
replace obs_n = _n

local event_dates "02apr2025 09apr2025 15apr2025 12may2025 10oct2025 30oct2025"
local event_names  "ev_lib_eo ev_145tariff ev_deliv_ban ev_geneva_truce ev_100pct_threat ev_busan_summit"

local i = 1
foreach d of local event_dates {
    local name : word `i' of `event_names'
    quietly summarize obs_n if statadate == td(`d')
    local anchor = r(mean)
    gen `name' = inrange(obs_n, `anchor' - 1, `anchor' + 5)
    local i = `i' + 1
}

drop obs_n

regress r_boeing r_market ev_lib_eo ev_145tariff ev_deliv_ban ev_geneva_truce ev_100pct_threat ev_busan_summit, robust

test ev_lib_eo ev_145tariff ev_deliv_ban ev_geneva_truce ev_100pct_threat ev_busan_summit
