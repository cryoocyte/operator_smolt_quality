select t2.type_from2 as origin_site_type,t1.ponding_date,t1.target_seawater_locus_id, t1.target_seawater_locus_population_id, t2.transport_date, t2.pretransfer_fw_locus_id, t2.pretransfer_fw_locus_population_id, t2.fish_count_shipped_out, t2.avg_weight_g_stocked --, t1.transport_locus_population_id
from (
SELECT st.type as type_from1, em.event_date::date as ponding_date,em.to_locus_id as target_seawater_locus_id, em.to_locus_population_id as target_seawater_locus_population_id,  em.from_locus_population_id as transport_locus_population_id, em.fish_count, em.avg_weight, em.fish_count as fish_count_stocked
FROM public.evt_movement em
join lkp_movement_type lmt on lmt.id=em.movement_type_id 
left join locus l on l.id = em.to_locus_id
left join site s on s.id = l.site_id
left join site_type st on st.id = s.type_id
where event_date::date >= '2017-01-01'
and lmt.name='P' and st.type='Seawater' --and to_locus_id = 7183878
and em.licensee_id = 117
) t1 left join (
SELECT st2.type as type_from2,em.event_date::date as transport_date,em.to_locus_population_id as transport_locus_population_id, em.from_locus_population_id as pretransfer_fw_locus_population_id, em.from_locus_id as pretransfer_fw_locus_id, em.fish_count as fish_count_shipped_out , em.avg_weight as avg_weight_g_stocked
FROM public.evt_movement em
join lkp_movement_type lmt on lmt.id=em.movement_type_id 
join locus l2 on l2.id = em.from_locus_id
join site s2 on s2.id = l2.site_id
join site_type st2 on st2.id = s2.type_id
where lmt.name='SO' --and st2.type in ('Freshwater', 'Freshwater Reproduction')
and em.licensee_id = 117
) t2 on t1.transport_locus_population_id= t2.transport_locus_population_id
order by t1.ponding_date, t1.target_seawater_locus_id, t2.pretransfer_fw_locus_population_id