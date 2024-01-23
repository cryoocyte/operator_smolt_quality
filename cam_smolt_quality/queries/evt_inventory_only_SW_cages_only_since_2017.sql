select ei.event_date, ei.locus_id, ei.locus_population_id, ei.open_count, ei.open_weight::int ,ei.close_count , ei.close_weight::int, ei.degree_days,coalesce(efm.feed_amount,0)::int as feed_amount
from evt_inventory ei 
join locus l on l.id = ei.locus_id
join site s on s.id = l.site_id
join site_type st on st.id = s.type_id
left join (
select locus_population_id , start_reg_time::date, sum(amount) as feed_amount
from evt_feed_mrts efm
group by locus_population_id , start_reg_time
) efm on efm.locus_population_id = ei.locus_population_id and efm.start_reg_time=ei.event_date
where ei.licensee_id = 117 and event_date > '2017-01-01' and type='Seawater'