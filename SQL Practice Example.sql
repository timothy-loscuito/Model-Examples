CREATE TABLE meat_poultry_egg_establishments (
    establishment_number text CONSTRAINT est_number_key PRIMARY KEY,
    company text,
    street text,
    city text,
    st text,
    zip text,
    phone text,
    grant_date date,
    activities text,
    dbas text
);

COPY meat_poultry_egg_establishments
FROM 'C:\Users\timot\practical-sql-2-main\practical-sql-2-main\Chapter_10\MPI_Directory_by_Establishment_Name.csv'
WITH (FORMAT CSV, HEADER);

CREATE INDEX company_idx ON meat_poultry_egg_establishments (company);

select count(*)
from public.meat_poultry_egg_establishments

select company, street, city, st, count(*) as address_count
from public.meat_poultry_egg_establishments
group by company, street, city, st
having count(*) > 1
order by address_count desc

select company, street, city, st, count(*) as address_count
from public.meat_poultry_egg_establishments
group by company, street, city, st
order by address_count desc

select *
from public.meat_poultry_egg_establishments
where company = 'Crider, Inc.' and street = '1 Plant Avenue' and city = 'Stillmore' and st = 'GA'


create table public.meat_poultry_egg_establishments_backup as
select *,
	   FALSE as meat_processing,
	   FALSE as poultry_processing 
from public.meat_poultry_egg_establishments;

update public.meat_poultry_egg_establishments_backup
set meat_processing = TRUE
where activities like '%Meat Processing%';

update public.meat_poultry_egg_establishments_backup
set poultry_processing = TRUE
where activities like '%Poultry Processing%';

select meat_processing, poultry_processing, count(*)
from public.meat_poultry_egg_establishments_backup
where meat_processing = true and poultry_processing = true
group by meat_processing, poultry_processing

select meat_processing, count(*)
from public.meat_poultry_egg_establishments_backup
where meat_processing = true
group by meat_processing

select count(*)
from public.meat_poultry_egg_establishments_backup
where poultry_processing = true




CREATE TABLE acs_2014_2018_stats (
    geoid text CONSTRAINT geoid_key PRIMARY KEY,
    county text NOT NULL,
    st text NOT NULL,
    pct_travel_60_min numeric(5,2),
    pct_bachelors_higher numeric(5,2),
    pct_masters_higher numeric(5,2),
    median_hh_income integer,
    CHECK (pct_masters_higher <= pct_bachelors_higher)
);

COPY acs_2014_2018_stats
FROM 'C:\Users\timot\practical-sql-2-main\practical-sql-2-main\Chapter_11\acs_2014_2018_stats.csv'
WITH (FORMAT CSV, HEADER);

SELECT * FROM acs_2014_2018_stats;

select cast(corr(median_hh_income,pct_bachelors_higher) as numeric (10,6)) as bachelors_income_r
from public.acs_2014_2018_stats;

select distinct st
from public.acs_2014_2018_stats
order by st

SELECT count(distinct (store, category)) from public.store_sales;

select count(*) from public.store_sales;

SELECT COUNT(*) AS distinct_pk_count
FROM ( 
	SELECT DISTINCT store, category
	FROM public.store_sales
) AS subquery;


select category,
	   store,
	   unit_sales,
	   rank() over (partition by category order by unit_sales desc) as category_rank
from public.store_sales
order by category, category_rank;





CREATE TABLE us_exports (
    year smallint,
    month smallint,
    citrus_export_value bigint,	
    soybeans_export_value bigint,
	constraint year_month primary key (year, month)
);

COPY us_exports
FROM 'C:\Users\timot\practical-sql-2-main\practical-sql-2-main\Chapter_11\us_exports.csv'
WITH (FORMAT CSV, HEADER);


select year,
	   month,
	   round(sum(soybeans_export_value) 
	   		over (order by year, month rows between 11 preceding and current row) / 1000000000, 2)
	   			as twelve_mo_roll_sum_in_bill
from us_exports
order by year, month, twelve_mo_roll_sum_in_bill;



select fscskey,
	   round((cast(visits as numeric) / popu_lsa) * 1000, 1) as visits_per_1000,
	   rank() over (order by round((cast(visits as numeric) / popu_lsa) * 1000, 1) desc) as visits_rank
from public.pls_fy2018_libraries
where popu_lsa >= 250000
order by visits_rank;

select count(distinct fscskey)
from pls_fy2018_libraries
where popu_lsa >= 250000;

select * from teachers_lab_access

select * from teachers

select first_name,
	   last_name,
	   lab_name,
	   access_time
from teachers
join teachers_lab_access on teachers.id = teachers_lab_access.teacher_id
order by last_name;


select last_name,
	   lab_name,
	   access_time,
	   access_order
from teachers as t
join
	(select teacher_id,
			lab_name,
			access_time,
	   		row_number() over (partition by teacher_id order by access_time desc) as access_order
	from teachers_lab_access
	order by teacher_id, access_order) as act
on t.id = act.teacher_id
where access_order IN (1,2)
order by last_name, access_order;



CREATE TABLE temperature_readings (
    station_name text,
    observation_date date,
    max_temp integer,
    min_temp integer,
    CONSTRAINT temp_key PRIMARY KEY (station_name, observation_date)
);

COPY temperature_readings
FROM 'C:\Users\timot\practical-sql-2-main\practical-sql-2-main\Chapter_13\temperature_readings.csv'
WITH (FORMAT CSV, HEADER);

select * from temperature_readings


with temps_collasped (station_name, max_temperature_group) as 
	(select station_name,
			case when max_temp >= 90 then '90 or more'
				 when max_temp < 90 and max_temp > 79 then 'the 80s'
				 when max_temp <= 79 then '79 or less'
				 else 'No reading'
		 	end
	 from temperature_readings)
select station_name, max_temperature_group, count(*)
from temps_collasped
where station_name like 'WAIK%'
group by station_name, max_temperature_group
order by count(*) desc;



