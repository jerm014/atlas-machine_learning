-- display the average temperature (Fahrenheit) by city
-- from the 'temperatures' table, ordered by temperature (descending)

DESCRIBE temperatures;

SELECT city, AVG(temperature) AS avg_temp
FROM temperatures
GROUP BY city
ORDER BY avg_fahrenheit DESC;
