SELECT
	event_date,locus_id,open_count,open_weight,close_count,close_weight,degree_days
FROM
	public.evt_inventory ei 
	JOIN public.locus l 
		ON l.id = ei.locus_id 
	JOIN public.site s
		ON s.id = l.site_id
WHERE 
	ei.licensee_id = 117
	AND s.orgunitname IN ('Rio Petrohue', 'UPS')
	AND ei.event_date >= '2015-01-01'
	
