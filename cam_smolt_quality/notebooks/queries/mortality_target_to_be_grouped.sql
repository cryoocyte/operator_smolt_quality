select *
--ttt1.locus_id,ttt1.fish_group_id
--, avg(total_count)::int as avg_count
--, sum(total_mortality/total_count) as total_mortality_perc
--, sum(transport_mortality/total_count) as transport_mortality_perc
--, sum(nontransport_mortality/total_count) as nontransport_mortality_perc
from (
select 
tt2.transfer_date,
tt1.*, case when stock_count>0 then (stock_biomass*1000/stock_count)::int else 0 end as stock_weight, open_count+stock_count as total_count
from (
--this ecto baseline view groups inventory data from the locus_population level to the locus level
--it also includes summed daily mortality counts and percentages by day without mortality reasons
with
  inventory_data as (
    select
     i.event_date
      , i.licensee_id
      , i.site_id
      , i.type_id
      , i.locus_id
      , i.fish_group_id
      , max(open_count) open_count
      , max(open_weight) open_weight
      , max(close_count) close_count
      , max(close_weight) close_weight
    from
      (select
  i.id
  , i.event_date
  , s.licensee_id
  , l.site_id
  , s.type_id
  , i.locus_id
  , i.locus_population_id
  , fg.id fish_group_id
  , close_count
  , close_weight
  , close_biomass
  , open_count
  , open_weight
  , open_biomass
from
  evt_inventory i
  join locus l on
    l.id = i.locus_id
  join site s on
    s.id = l.site_id
  join locus_population lp on
    lp.id = i.locus_population_id
  join v_fish_group fg on
    fg.id = lp.fish_group_id
      ) as i
      join locus l on
        l.id = i.locus_id
      join v_fish_group fg on
        fg.id = i.fish_group_id
      join site s on
        s.id = i.site_id
      join licensee lc on
        lc.id = s.licensee_id
      join site_type st on
        st.id = s.type_id
      join ref_species sp on
        sp.id = fg.species_id
    where
      s.type_id <> 8
    group by
      1
      , 2
      , 3
      , 4
      , 5
      , 6
    order by
      1
 )
  --crete daily stocking counts by to_locus_id to add to the inventorty data above
  , stock_values as (
    select
      to_date event_date
      , to_locus_id locus_id
      , coalesce(sum(to_count), 0) as stock_count
      , coalesce(sum(to_biomass), 0) as stock_biomass
    from
      (select
  FROM_DATE
  , s.LICENSEE_ID
  , FROM_SITE_NAME
  , FROM_SITE_ID
  , FROM_LOCUS_NAME
  , FROM_LOCUS_ID
  , fg.name FROM_FISH_GROUP_NAME
  , fg.id FROM_FISH_GROUP_ID
  , fg.CAM_GENERATION as cam_fw_generation
  , fg.CAM_YEAR_CLASS as cam_fw_year_class
  , sum(FROM_COUNT) FROM_COUNT
  , avg(FROM_AVG_WEIGHT) FROM_AVG_WEIGHT
  , sum(FROM_BIOMASS) FROM_BIOMASS
  , TO_DATE
  , TYPE_ID
  , TO_SITE_NAME
  , TO_SITE_ID
  , TO_LOCUS_NAME
  , TO_LOCUS_ID
  , TO_FISH_GROUP_ID
  , TO_FISH_GROUP_NAME
  , sum(TO_COUNT) TO_COUNT
  , avg(TO_AVG_WEIGHT) TO_AVG_WEIGHT
  , sum(TO_BIOMASS) TO_BIOMASS
from
  (select
  from_date::date from_date
  , es.licensee_id
  , from_site_name
  , from_site_id
  , from_locus_name
  , from_locus_id
  , from_locus_population_id
  , from_fg.id from_fish_group_id
  , from_fg.name from_fish_group_name
  , from_fg.cam_generation as from_generation  --only works for CAM right now
  , from_fg.cam_year_class as from_year_class  --only works for CAM right now
  --, from_trans_loc_pop_id
  , from_count::numeric from_count
  , from_avg_weight::numeric from_avg_weight
  , from_biomass::numeric from_biomass
  , to_date::date to_date
  , s.type_id
  , to_site_name
  , to_site_id
  , to_locus_name
  , to_locus_id
  , to_locus_population_id
  , to_fg.id to_fish_group_id
  , to_fg.name to_fish_group_name
  , to_fg.cam_generation as to_generation  --only works for CAM right now
  , to_fg.cam_year_class as to_year_class  --only works for CAM right now
  , to_count::numeric to_count
  , to_avg_weight::numeric to_avg_weight
  , to_biomass::numeric to_biomass
  ,  min(from_date::date) over(partition by from_fg.id, from_locus_id) as transfer_date -- updated from_fish_group_id 2021-06-03 - Israel 
from
  v_evt_stocking es
  join site s on
    s.id = es.to_site_id
  join locus_population to_lp on
    to_lp.id = es.to_locus_population_id
  join v_fish_group to_fg on
    to_fg.id = to_lp.fish_group_id
  join locus_population from_lp on
    from_lp.id = es.from_locus_population_id
  join v_fish_group from_fg on
    from_fg.id = from_lp.fish_group_id
  ) as s
  join locus_population lp on
    lp.id = s.from_locus_population_id
  join v_fish_group fg on
    fg.id = lp.fish_group_id
group by
  1
  , 2
  , 3
  , 4
  , 5
  , 6
  , 7
  , 8
  , 9
  , 10
  , 14
  , 15
  , 16
  , 17
  , 18
  , 19
  , 20
  , 21
order by
  1 desc
      )as s
    group by
      1
      , 2
  )
-- join tables to get final view
select
  id.*
  , coalesce(stock_count, 0) as stock_count
  , coalesce(stock_biomass, 0) as stock_biomass
from
  inventory_data id
  left join stock_values sv on
    sv.locus_id = id.locus_id
    and sv.event_date = id.event_date
) tt1 join (
select t1.*, t2.from_fish_group_count from (
select
  from_date::date from_date
  , from_site_name
  , from_site_id
  , from_locus_id
  , from_locus_population_id
  , from_fg.id from_fish_group_id
  , from_fg.name from_fish_group_name
  , from_fg.cam_generation as from_generation  --only works for CAM right now
  , from_fg.cam_year_class as from_year_class  --only works for CAM right now
  , from_count::numeric from_count_stocking
  , from_avg_weight::int from_avg_weight
  , to_date::date to_date
  , s.type_id
  , to_site_name
  , to_site_id
  , to_locus_id
  , to_locus_population_id
  , to_fg.id to_fish_group_id
  , to_fg.name to_fish_group_name
  , to_fg.cam_generation as to_generation  --only works for CAM right now
  , to_fg.cam_year_class as to_year_class  --only works for CAM right now
  , to_count::numeric to_count_stocking
  , to_avg_weight::int to_avg_weight
  ,  min(from_date::date) over(partition by from_fg.id, from_locus_id) as transfer_date -- updated from_fish_group_id 2021-06-03 - Israel 
from
  v_evt_stocking es
  join site s on
    s.id = es.to_site_id
  join locus_population to_lp on
    to_lp.id = es.to_locus_population_id
  join v_fish_group to_fg on
    to_fg.id = to_lp.fish_group_id
  join locus_population from_lp on
    from_lp.id = es.from_locus_population_id
  join v_fish_group from_fg on
    from_fg.id = from_lp.fish_group_id
  left join v_fish_group_cam vfgc1 on
  	from_fg.id = vfgc1.id
  left join v_fish_group_cam vfgc2 on
  	to_fg.id = vfgc2.id
) t1 join (
select to_locus_id,to_fish_group_id, max(from_count) as max_from_count, count(distinct from_fish_group_id) as from_fish_group_count
from (
select
   from_locus_id
  , from_locus_population_id
  , from_fg.id from_fish_group_id
  , from_fg.name from_fish_group_name
  , from_count::numeric from_count
  , to_locus_id
  , to_locus_population_id
  , to_fg.id to_fish_group_id
  , to_fg.name to_fish_group_name
from
  v_evt_stocking es
  join site s on
    s.id = es.to_site_id
  join locus_population to_lp on
    to_lp.id = es.to_locus_population_id
  join v_fish_group to_fg on
    to_fg.id = to_lp.fish_group_id
  join locus_population from_lp on
    from_lp.id = es.from_locus_population_id
  join v_fish_group from_fg on
    from_fg.id = from_lp.fish_group_id
) t1
group by to_locus_id, to_fish_group_id) t2
on t2.max_from_count = t1.from_count_stocking and t2.to_fish_group_id = t1.to_fish_group_id and t2.to_locus_id = t1.to_locus_id
) tt2 on tt1.locus_id=tt2.to_locus_id and tt1.fish_group_id=tt2.to_fish_group_id
where tt1.event_date-tt2.transfer_date between -1 and 90 --to be changed between 1 and 120
and transfer_date>='2017-01-01'
and open_count+stock_count>0
--and tt2.to_fish_group_id=11 and tt2.to_locus_id=3046036
--order by tt1.fish_group_id,tt1.locus_id,event_date
) ttt1 left join (
select m1.*, COALESCE(m2.transport_mortality, 0) as transport_mortality, m1.total_mortality-COALESCE(m2.transport_mortality, 0) as nontransport_mortality
from (select event_date, locus_id, fish_group_id
--, min(transfer_date) as transfer_date
, sum(mortality_count) as total_mortality--tt2.transfer_date,tt1.*
from (
select
  event_date
  , m.licensee_id
  , m.site_id
  , s.type_id
  , m.locus_id
  , m.locus_population_id
  , lp.fish_group_id
  , m.mortality_reason_id
  , mr.mortality_reason
  , mr.infectious
  , mrg.group_name as mortality_reason_group
  , mortality_count
  , mortality_weight
  , (mortality_count*mortality_weight)/1000 mortality_biomass
from
  evt_mortality_mrts m
  join lkp_mortality_reason mr on
    mr.id = m.mortality_reason_id
  join site s on
    s.id = m.site_id
  join locus_population lp on
    lp.id = m.locus_population_id
  left join lkp_mortality_reason_group mrg on
    mrg.id = mr.mortality_reason_group_id
) tt1 join (
select t1.*, t2.from_fish_group_count from (
select
  from_date::date from_date
  , from_site_name
  , from_site_id
  , from_locus_id
  , from_locus_population_id
  , from_fg.id from_fish_group_id
  , from_fg.name from_fish_group_name
  , from_fg.cam_generation as from_generation  --only works for CAM right now
  , from_fg.cam_year_class as from_year_class  --only works for CAM right now
  , from_count::numeric from_count_stocking
  , from_avg_weight::int from_avg_weight
  , to_date::date to_date
  , s.type_id
  , to_site_name
  , to_site_id
  , to_locus_id
  , to_locus_population_id
  , to_fg.id to_fish_group_id
  , to_fg.name to_fish_group_name
  , to_fg.cam_generation as to_generation  --only works for CAM right now
  , to_fg.cam_year_class as to_year_class  --only works for CAM right now
  , to_count::numeric to_count_stocking
  , to_avg_weight::int to_avg_weight
  ,  min(from_date::date) over(partition by from_fg.id, from_locus_id) as transfer_date -- updated from_fish_group_id 2021-06-03 - Israel 
from
  v_evt_stocking es
  join site s on
    s.id = es.to_site_id
  join locus_population to_lp on
    to_lp.id = es.to_locus_population_id
  join v_fish_group to_fg on
    to_fg.id = to_lp.fish_group_id
  join locus_population from_lp on
    from_lp.id = es.from_locus_population_id
  join v_fish_group from_fg on
    from_fg.id = from_lp.fish_group_id
  left join v_fish_group_cam vfgc1 on
  	from_fg.id = vfgc1.id
  left join v_fish_group_cam vfgc2 on
  	to_fg.id = vfgc2.id
) t1 join (
select to_locus_id,to_fish_group_id, max(from_count) as max_from_count, count(distinct from_fish_group_id) as from_fish_group_count
from (
select
   from_locus_id
  , from_locus_population_id
  , from_fg.id from_fish_group_id
  , from_fg.name from_fish_group_name
  , from_count::numeric from_count
  , to_locus_id
  , to_locus_population_id
  , to_fg.id to_fish_group_id
  , to_fg.name to_fish_group_name
from
  v_evt_stocking es
  join site s on
    s.id = es.to_site_id
  join locus_population to_lp on
    to_lp.id = es.to_locus_population_id
  join v_fish_group to_fg on
    to_fg.id = to_lp.fish_group_id
  join locus_population from_lp on
    from_lp.id = es.from_locus_population_id
  join v_fish_group from_fg on
    from_fg.id = from_lp.fish_group_id
) t1
group by to_locus_id, to_fish_group_id) t2
on t2.max_from_count = t1.from_count_stocking and t2.to_fish_group_id = t1.to_fish_group_id and t2.to_locus_id = t1.to_locus_id
) tt2 on tt1.locus_id=tt2.to_locus_id and tt1.fish_group_id=tt2.to_fish_group_id
where tt1.event_date-tt2.transfer_date between -1 and 90 --to be changed between 1 and 120
and transfer_date>='2017-01-01'
--and tt2.to_fish_group_id=107 and tt2.to_locus_id=3047487
group by event_date, locus_id, fish_group_id
--order by tt1.fish_group_id,tt1.locus_id,event_date
) m1 left join (
select event_date, locus_id, fish_group_id, sum(mortality_count) as transport_mortality--tt2.transfer_date,tt1.*
from (
select
  event_date
  , m.licensee_id
  , m.site_id
  , s.type_id
  , m.locus_id
  , m.locus_population_id
  , lp.fish_group_id
  , m.mortality_reason_id
  , mr.mortality_reason
  , mr.infectious
  , mrg.group_name as mortality_reason_group
  , mortality_count
  , mortality_weight
  , (mortality_count*mortality_weight)/1000 mortality_biomass
from
  evt_mortality_mrts m
  join lkp_mortality_reason mr on
    mr.id = m.mortality_reason_id
  join site s on
    s.id = m.site_id
  join locus_population lp on
    lp.id = m.locus_population_id
  left join lkp_mortality_reason_group mrg on
    mrg.id = mr.mortality_reason_group_id
) tt1 join (
select t1.*, t2.from_fish_group_count from (
select
  from_date::date from_date
  , from_site_name
  , from_site_id
  , from_locus_id
  , from_locus_population_id
  , from_fg.id from_fish_group_id
  , from_fg.name from_fish_group_name
  , from_fg.cam_generation as from_generation  --only works for CAM right now
  , from_fg.cam_year_class as from_year_class  --only works for CAM right now
  , from_count::numeric from_count_stocking
  , from_avg_weight::int from_avg_weight
  , to_date::date to_date
  , s.type_id
  , to_site_name
  , to_site_id
  , to_locus_id
  , to_locus_population_id
  , to_fg.id to_fish_group_id
  , to_fg.name to_fish_group_name
  , to_fg.cam_generation as to_generation  --only works for CAM right now
  , to_fg.cam_year_class as to_year_class  --only works for CAM right now
  , to_count::numeric to_count_stocking
  , to_avg_weight::int to_avg_weight
  ,  min(from_date::date) over(partition by from_fg.id, from_locus_id) as transfer_date -- updated from_fish_group_id 2021-06-03 - Israel 
from
  v_evt_stocking es
  join site s on
    s.id = es.to_site_id
  join locus_population to_lp on
    to_lp.id = es.to_locus_population_id
  join v_fish_group to_fg on
    to_fg.id = to_lp.fish_group_id
  join locus_population from_lp on
    from_lp.id = es.from_locus_population_id
  join v_fish_group from_fg on
    from_fg.id = from_lp.fish_group_id
) t1 join (
select to_locus_id,to_fish_group_id, max(from_count) as max_from_count, count(distinct from_fish_group_id) as from_fish_group_count
from (
select
   from_locus_id
  , from_locus_population_id
  , from_fg.id from_fish_group_id
  , from_fg.name from_fish_group_name
  , from_count::numeric from_count
  , to_locus_id
  , to_locus_population_id
  , to_fg.id to_fish_group_id
  , to_fg.name to_fish_group_name
from
  v_evt_stocking es
  join site s on
    s.id = es.to_site_id
  join locus_population to_lp on
    to_lp.id = es.to_locus_population_id
  join v_fish_group to_fg on
    to_fg.id = to_lp.fish_group_id
  join locus_population from_lp on
    from_lp.id = es.from_locus_population_id
  join v_fish_group from_fg on
    from_fg.id = from_lp.fish_group_id
) t1
group by to_locus_id, to_fish_group_id) t2
on t2.max_from_count = t1.from_count_stocking and t2.to_fish_group_id = t1.to_fish_group_id and t2.to_locus_id = t1.to_locus_id
) tt2 on tt1.locus_id=tt2.to_locus_id and tt1.fish_group_id=tt2.to_fish_group_id
where tt1.event_date-tt2.transfer_date between -1 and 90
and transfer_date>='2017-01-01'
and mortality_reason_id=58 --mortality reason = Transporte
group by event_date, locus_id, fish_group_id
--order by tt1.fish_group_id,tt1.locus_id,event_date
) m2 on m1.event_date=m2.event_date and m1.locus_id=m2.locus_id and m1.fish_group_id=m2.fish_group_id
--where m1.fish_group_id=11 and m1.locus_id=3046036
--order by m1.fish_group_id,m1.locus_id,event_date
) ttt2 on ttt1.event_date=ttt2.event_date and ttt1.locus_id=ttt2.locus_id-- and ttt1.fish_group_id=ttt2.fish_group_id
--group by ttt1.locus_id,ttt1.fish_group_id