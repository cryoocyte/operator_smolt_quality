SELECT locus_id, event_date, treatment_method_id, reason, prescription, batch, amount, active_substance_id, is_net_raised, is_bottom_raised, tarp_type_id, locus_population_id
FROM public.evt_treatment AS et
JOIN public.locus l 
	ON l.id = et.locus_id 
JOIN public.site s
	ON s.id = l.site_id
where et.licensee_id =117 and event_date > '2015-01-01'  AND  s.orgunitname IN ('Rio Petrohue', 'UPS')
order by locus_id, event_date