-- Comment!

CREATE VIEW need_meeting AS
SELECT name
FROM students
WHERE
    score < 80
    AND (
        last_meeting IS NULL
        OR last_meeting < DATE_SUB(CURDATE(), INTERVAL 1 MONTH)
        -- Alternatively, if CURDATE() is not allowed or for a fixed date context:
        -- OR last_meeting < '2025-06-01'
    );
