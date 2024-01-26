select distinct mrts_feed_id,
feed_name
from evt_feed_mrts efm join
(
SELECT l.id as locus_id, l.site_id--, st.type--, s.type_id
FROM locus l
join site s on s.id = l.site_id
join site_type st on st.id = s.type_id
where l.site_id in (63827, 63858)
) t1 on efm.locus_id = t1.locus_id