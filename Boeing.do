clear all
set more off

import delimited "boeing_returns_data.csv", clear

* Standardize variable names if Python exported them without the "r_" prefix
capture rename boeing r_boeing
capture rename market r_market

* Set up date variable and time series tracking
gen statadate = date(date, "YMD")
format statadate %td
tsset statadate

* Define event windows (-1 to +5 trading days around announcement)
gen ev_libtariffs = (statadate >= td(01apr2025) & statadate <= td(10apr2025))
gen ev_145tariffs = (statadate >= td(08apr2025) & statadate <= td(16apr2025))
gen ev_delivban   = (statadate >= td(14apr2025) & statadate <= td(23apr2025))
gen ev_truce      = (statadate >= td(14may2025) & statadate <= td(22may2025))
gen ev_rareearth  = (statadate >= td(11oct2025) & statadate <= td(20oct2025))
gen ev_summitdeal = (statadate >= td(24oct2025) & statadate <= td(04nov2025))

* Main market model regression with robust standard errors
regress r_boeing r_market ev_libtariffs ev_145tariffs ev_delivban ev_truce ev_rareearth ev_summitdeal, robust

* Test for joint significance of all event dummies
test ev_libtariffs ev_145tariffs ev_delivban ev_truce ev_rareearth ev_summitdeal
