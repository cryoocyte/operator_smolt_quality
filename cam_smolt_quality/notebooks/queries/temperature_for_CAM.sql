SELECT site_id, locus_group_id, event_date, cast(avg(result_number) as decimal(16,1)) as value
FROM public.evt_logbook
where licensee_id = 117 and logbook_event_id=34-- and site_id =63858
group by site_id, locus_group_id, event_date
order by site_id, locus_group_id, event_date