-- display the average temperature (Fahrenheit) by city
-- from the 'temperatures' table, ordered by temperature (descending)

SELECT city, AVG(avg_temp) AS avg_fahrenheit
FROM temperatures
GROUP BY city
ORDER BY avg_fahrenheit DESC;
