select "Date" as event_ts, locus_id, locus_group_id as unit_group_id, "CO2 (mg/lt)" as sensor_type_value, 'Carbon dioxide' as sensor_name--, "Comments"
from (
with main_data as (
  select j.id as ecto_job_id
	   , j."number" as job_platform_id
	   , j.created_by as job_created_by
	   , j.date_created as job_date_created
     , j.start_date_time::date as job_start_date
     , j.last_updated_by as job_last_updated_by
     , j.title as job_title
       , ji.job_workflow_input_id
       , ji.input_group_id
       ,ji.id as input_id
       , ljwit.title as workflow_input_title
       , jwi."position" as workflow_input_position
       , jt.id as task_id
       , jt.job_workflow_task_id
       , jwt.title as workflow_task_title
       , jwt."position" as workflow_task_position
       , coalesce(ljwit.options ->> evtji.result_string, evtji.result_string) as result_string
       , evtji.result_date
       , evtji.result_time
       , evtji.result_number
       , evtji.result_boolean
       , evtji.result_json
       , f."path" as file_path
       , l.id as locus_id
       , l.containername as locus
       ,coalesce(lg.locus_group_id,lg2.locus_group_id) as locus_group_id
       ,coalesce(lg.locus_group_name,lg2.locus_group_name) as locus_group_name
       ,ejt.id as evt_task_id
       ,s.id as site_id
       ,s.orgunitname as site
       ,evtji.id as event_input_id
       ,ljwit.options as input_options
       ,j.date_completed as job_date_completed
       ,evtji.result_datetime
       ,jig.job_workflow_input_group_id
       ,jig.title as input_group_title
       ,jt.state as task_state
       ,ji.date_created as input_date_time_created
	FROM public.evt_job_input as evtji
         join public.job as j on j.id = evtji.job_id
         join public.job_input as ji on evtji.input_id = ji.id
         join public.job_task as jt on evtji.task_id = jt.id
         join public.job_workflow_task as jwt on jwt.id = jt.job_workflow_task_id
         join public.job_workflow_input as jwi on ji.job_workflow_input_id = jwi.id
         --join public.lkp_job_workflow_input_translation as ljwit on ljwit.translatable_id = jwi.id
         join public.lkp_job_input_translation as ljwit on ljwit.translatable_id = ji.id
         join public.evt_job_task as ejt on evtji.evt_job_task_id = ejt.id
         left join public.locus as l on ejt.locus_id = l.id
         left join public.locus_group as lg on ejt.locus_group_id = lg.locus_group_id
         left join public.locus_group as lg2 on l.locus_group_id = lg2.locus_group_id
         left join public.file as f on evtji.result_file_id = f.id
         left join public.site as s on j.site_id =s.id
         left join public.job_input_group as jig on ji.input_group_id=jig.id
         where (j.date_deleted IS NULL AND jt.date_deleted IS NULL AND ji.date_deleted IS NULL)
          and evtji.licensee_id = 117
          and j.job_workflow_id = 235
          and ljwit.locale = 'es'
)
select evt_task_id
      ,min(site_id) as site_id
      ,min(site) as site
      ,min(locus_group_id) as locus_group_id
      ,min(locus_group_name) as locus_group_name
      ,min(locus_id) as locus_id
      ,min(locus) as locus
      ,min(case when job_workflow_input_id=4977 then result_date end) as "Date"
      ,min(case when job_workflow_input_id=3633 then result_number end) as "CO2 (mg/lt)"
      ,min(case when job_workflow_input_id=3634 then result_string end) as "Comments"
from main_data
group by evt_task_id
--having min(case when job_workflow_input_id=3634 then result_string end) is not null
) t1
