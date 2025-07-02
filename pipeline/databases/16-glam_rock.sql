-- documentation goes here.

SELECT
    band_name,
    (LEAST(IFNULL(split, 2020), 2020) - formed) AS lifespan
FROM
    metal_bands
WHERE
    style LIKE '%Glam rock%' -- Using LIKE for cases where style might contain multiple values?
ORDER BY
    lifespan DESC;
