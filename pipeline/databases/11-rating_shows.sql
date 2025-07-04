-- list all shows from tvshows_rate by their rating
-- display the show title and the sum of its ratings
-- sorted: descending order by the rating sum

SELECT
    tv_shows.title,
    SUM(tv_show_ratings.rate) AS rating
FROM
    tv_shows
INNER JOIN
    tv_show_ratings ON tv_shows.id = tv_show_ratings.show_id
GROUP BY
    tv_shows.title
ORDER BY
    rating DESC;
