-- list all genres by their total rating
-- display the genre name and the sum of ratings for all shows belonging to that genre
-- sorted: descending order by the rating sum

SELECT
    tg.name,
    SUM(tsr.rate) AS rating
FROM
    tv_genres AS tg
INNER JOIN
    tv_show_genres AS tsg ON tg.id = tsg.genre_id
INNER JOIN
    tv_show_ratings AS tsr ON tsg.show_id = tsr.show_id
GROUP BY
    tg.name
ORDER BY
    rating DESC;