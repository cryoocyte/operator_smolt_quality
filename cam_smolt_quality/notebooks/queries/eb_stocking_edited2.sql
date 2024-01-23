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
  left join v_fish_group_cam vfgc1 on
  	from_fg.id = vfgc1.id
  left join v_fish_group_cam vfgc2 on
  	to_fg.id = vfgc2.id
) t1
group by to_locus_id, to_fish_group_id) t2
on t2.max_from_count = t1.from_count_stocking and t2.to_fish_group_id = t1.to_fish_group_id and t2.to_locus_id = t1.to_locus_id