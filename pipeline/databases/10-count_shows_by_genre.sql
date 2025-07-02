-- list all genres from tvshows and display the number of shows linked to each
-- Only genres with at least one show are displayed
-- Results are sorted by the number of shows linked, in descending order

SELECT
    tv_genres.name AS genre,
    COUNT(tv_show_genres.show_id) AS number_of_shows
FROM
    tv_genres
INNER JOIN
    tv_show_genres ON tv_genres.id = tv_show_genres.genre_id
GROUP BY
    tv_genres.name
ORDER BY
    number_of_shows DESC;
