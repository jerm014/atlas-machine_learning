-- list all records with a score >= 10 from the second_table
-- displays score and name, ordered by score (highest first)

SELECT score, name
FROM second_table
WHERE score >= 10
ORDER BY score DESC;
