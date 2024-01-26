with 
unselected_inputs as (
select 
distinct
j.number as job_id,
ljwt.title as job_name,
s.orgunitname as site,
Coalesce(lg.locus_group_name, lg2.locus_group_name) as unit_group,
Coalesce(lg.locus_group_id, lg2.locus_group_id) as unit_group_id,
l.containername as unit,
ljit.title,
eji.result_number result_number,
eji.result_date::text result_date
from Public.job j 
join Public.evt_job_task ejt on ejt.job_id = j.id
join Public.job_task jt on ejt.task_id = jt.id 
join Public.evt_job_input eji on ejt.id = eji.evt_job_task_id 
join Public.job_input ji on eji.input_id = ji.id 
join Public.job_workflow jw on j.job_workflow_id = jw.id
---
join Public.lkp_job_task_translation ljtt on jt.id = ljtt.translatable_id 
join Public.lkp_job_input_translation ljit on ji.id = ljit.translatable_id 
join Public.lkp_job_translation ljt on ljt.translatable_id = j.id
join Public.lkp_job_workflow_translation ljwt on ljwt.translatable_id = jw.id
---
left join Public.site s on s.id = ejt.site_id
left join Public.locus l on l.id = ejt.locus_id
left join Public.locus_group lg on lg.locus_group_id = ejt.locus_group_id
left join Public.locus_group lg2 on lg2.locus_group_id = l.locus_group_id
---
where 
 ljwt.title <>'test'
and j.licensee_id = 117
and jw.id = 202
and ljit.locale='es'
and ljtt.locale='es'
and ljt.locale='es'
and ljwt.locale='es'
--
and j.date_deleted is null
and jt.date_deleted is null
and ji.date_deleted is null
and jw.date_deleted is null
  ),
not_grouped as (
select 
job_id,
case when title = 'Fecha del registro' then result_date end as date,
site,
unit_group,
unit_group_id,
unit,
case when title <> 'Fecha del registro' then result_number end as value,
case when title <> 'Fecha del registro' then title end as sensor
from unselected_inputs
where 
(
result_date::text <> ''
or result_number::text <> ''
)
 ),
dates as (
  select distinct 
date
, site
, unit_group
, unit
, job_id
  from not_grouped
  ),
types as (
select
d2.date event_ts
--, d1.site
--, d1.unit_group
, d1.unit_group_id
--, d1.unit
, d1.value sensor_type_value
, d1.sensor sensor_name
, case 
when d1.sensor='Transmitancia' then 'Conductivity' 
when d1.sensor in ('Alcalinidad','pH') then 'PH' 
when d1.sensor='Sal (ppt)' then 'Salinity' 
when d1.sensor='Solidos susp' then 'Solids' 
when d1.sensor='Temperatura (°C)' then 'Temperature' 
when d1.sensor='PO4' then 'Total phosphorus' 
when d1.sensor='Turbidez' then 'Turbidity' 
when d1.sensor in ('NH4','NO2','NO3','N-NH3') then 'Nitrogen' 
when d1.sensor in ('Volumen M3','M3 Ingresados') then 'M3 Jobs' 
when d1.sensor='Dureza' then 'Dureza' 
when d1.sensor='Recambio (%)' then 'Recambio' 
end as type_name
--,case 
--  when d1.site='Rio Petrohue' then 'Jobs - Petrohué'
--  when d1.site='UPS' then 'Jobs - UPS'
--  end as sensor_source
from not_grouped d1
left join dates d2 on d1.job_id=d2.job_id
and d1.site=d2.site
and d1.unit_group=d2.unit_group
--and d1.unit=d2.unit
where d2.date is not null
and d1.sensor is not null
)
select t.*
--, st.unit_name
from types t
left join 
(select distinct type_name type_name2, unit_name
from Public.sensor_type) st on st.type_name2=t.type_name
order by unit_group_id, event_ts