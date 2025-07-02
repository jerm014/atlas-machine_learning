-- display the maximum temperature (Fahrenheit) for each state,
-- ordered by state name in ascending order.

SELECT state, MAX(value) AS max_temp
FROM temperatures
GROUP BY state
ORDER BY state ASC;
