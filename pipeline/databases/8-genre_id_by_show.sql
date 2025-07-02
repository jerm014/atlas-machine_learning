-- list all shows that have at least one genre linked
-- It displays the show title and the linked genre ID
-- Results are sorted by title and then genre ID in ascending order

SELECT tv_shows.title, tv_show_genres.genre_id
FROM tv_shows
INNER JOIN tv_show_genres ON tv_shows.id = tv_show_genres.show_id
ORDER BY tv_shows.title ASC, tv_show_genres.genre_id ASC;
