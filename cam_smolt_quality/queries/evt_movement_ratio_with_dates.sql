select emr.final_locus_population_id, historic_locus_population_id, lp.locus_id as historic_locus_id , starttime::date, endtime::date,count_ratio--, t1.first_inventory_date--, smolt_site_id
from evt_movement_ratio emr 
join locus_population lp on lp.id=emr.historic_locus_population_id 
join (
select final_locus_population_id, min(lp.starttime)::date as first_inventory_date, max(lp.endtime)::date as shipout_date
from evt_movement_ratio emr
join locus_population lp on lp.id=emr.historic_locus_population_id 
where lp.licensee_id = 117--final_locus_population_id = 194512144
group by final_locus_population_id
--only locus_population first_inventory_date from start of 2015
having min(lp.starttime)::date>'2015-01-01' 
) t1 on emr.final_locus_population_id = t1.final_locus_population_id
where emr.licensee_id = 117 and lp.product_type = 'N' --and emr.final_locus_population_id = 194512144 --and starttime=em.event_date::date
order by final_locus_population_id,lp.locus_id, starttime