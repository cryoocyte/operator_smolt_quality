SELECT site_id, locus_id, locus_population_id, mortality_reason_id, event_date, mortality_count, cast(mortality_weight as decimal(18,2))
FROM public.evt_mortality_mrts
where licensee_id = 117 and site_id in (63827, 63858) and event_date > '2015-01-01'