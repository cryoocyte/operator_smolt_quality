select final_locus_population_id as pretransfer_fw_locus_population_id, lp2.locus_id as pretransfer_fw_locus_id, min(lp.starttime)::date as first_movement_date, min(efm.start_reg_time)::date as first_feeding_date, max(lp.endtime)::date as shipout_date
from evt_movement_ratio emr
left join evt_feed_mrts efm on efm.locus_population_id = emr.historic_locus_population_id 
join locus_population lp on lp.id=emr.historic_locus_population_id 
join locus_population lp2 on lp2.id=emr.final_locus_population_id 
where lp2.licensee_id = 117--final_locus_population_id = 194512144
group by final_locus_population_id, lp2.locus_id
--having min(lp.starttime)::date>'2015-01-01' 
order by final_locus_population_id