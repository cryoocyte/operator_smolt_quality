SELECT ei.locus_id, ei.locus_population_id, ei.event_date, open_count, CAST(open_weight as numeric(10,2)) as open_weight_g, close_count, CAST(close_weight as numeric(10,2)) as close_weight_g--ei.locus_id, fish_group_id, stocking_weight,event_date --tbl_transfer_date.transfer_date, l.site_id, to_lp.fish_group_id, event_date, ei.licensee_id, ei.locus_id, locus_population_id, client_model_id, open_count, open_weight, open_biomass, close_count, close_weight, degree_days, close_biomass
FROM evt_inventory ei 
join locus l on ei.locus_id = l.id
join locus_population to_lp on to_lp.id = ei.locus_population_id
join site s on s.id = l.site_id
join site_type st on st.id = s.type_id
where ei.licensee_id = 117 and l.site_id in (63827, 63858) and event_date>'2015-01-01' --and type='Freshwater' 
order by ei.locus_id, ei.locus_population_id, event_date