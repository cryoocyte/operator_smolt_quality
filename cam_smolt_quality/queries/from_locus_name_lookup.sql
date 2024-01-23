select tt1.*, tt2.from_locus_name, tt2.from_date from (
select t2.event_date, t2.locus_id, t2.light_regime_type_id
from (
SELECT locus_id , max(event_date) max_event_date
FROM public.evt_light_regime
where licensee_id = 117--and event_date > '2023-03-01'
group by locus_id) t1 join public.evt_light_regime t2 on t1.max_event_date=t2.event_date and t1.locus_id=t2.locus_id
where t2.licensee_id = 117
) tt1 left join (
select ves.from_locus_name, ves.from_locus_id, ves.from_date::date --ves.from_locus_name , ves.from_locus_id , t2.max_from_date
from (select distinct from_locus_name, from_locus_id, from_date from public.v_evt_stocking where licensee_id = 117) ves inner join (
SELECT from_locus_name, max(from_date) max_from_date --, from_locus_id, , from_site_name, from_site_id, from_locus_population_id, from_trans_loc_pop_id, from_count, from_biomass, from_avg_weight, from_mrts_mov_id, licensee_id, to_date, to_site_name, to_site_id, to_locus_name, to_locus_id, to_locus_population_id, to_trans_loc_pop_id, to_count, to_biomass, to_avg_weight, to_mrts_mov_id
FROM public.v_evt_stocking
where licensee_id =117
group by from_locus_name 
) t2 on from_date = max_from_date and ves.from_locus_name = t2.from_locus_name 
order by ves.from_locus_name 
) tt2 on tt1.locus_id=tt2.from_locus_id
order by from_locus_name