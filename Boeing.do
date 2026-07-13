clear all
set more off

import delimited "boeing_returns_data.csv", clear

capture rename boeing r_boeing
capture rename market r_market

gen statadate = date(date, "YMD")
format statadate %td
tsset statadate

gen ev_lib_eo = (statadate >= td(01apr2025) & statadate <= td(10apr2025)) 
gen ev_145tariff = (statadate >= td(08apr2025) & statadate <= td(16apr2025)) 
gen ev_deliv_ban = (statadate >= td(14apr2025) & statadate <= td(23apr2025)) 
gen ev_geneva_truce = (statadate >= td(09may2025) & statadate <= td(20may2025)) 
gen ev_100pct_threat = (statadate >= td(09oct2025) & statadate <= td(20oct2025)) 
gen ev_busan_summit = (statadate >= td(29oct2025) & statadate <= td(07nov2025)) 

regress r_boeing r_market ev_lib_eo ev_145tariff ev_deliv_ban ev_geneva_truce ev_100pct_threat ev_busan_summit, robust

test ev_lib_eo ev_145tariff ev_deliv_ban ev_geneva_truce ev_100pct_threat ev_busan_summit
