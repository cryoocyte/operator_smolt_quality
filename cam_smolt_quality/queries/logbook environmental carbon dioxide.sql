SELECT site_id, locus_group_id, logbook_event_id, event_date, cast(avg(result_number) as decimal(16,1)) as value, STRING_AGG(result_string, ', ') as string
FROM public.evt_logbook
where licensee_id = 117 and logbook_event_id in (109,264)   --and locus_group_id = 555-- and site_id =63858
group by site_id, locus_group_id, event_date, logbook_event_id
order by locus_group_id, logbook_event_id, event_date, site_id 