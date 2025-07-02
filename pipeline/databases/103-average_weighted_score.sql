-- Comment!

DELIMITER //

CREATE PROCEDURE ComputeAverageWeightedScoreForUser(
    IN p_user_id INT
)
BEGIN
    DECLARE weighted_avg_score FLOAT;

    -- Calculate the average weighted score for the specified user
    SELECT
        SUM(c.score * p.weight) / SUM(p.weight)
    INTO
        weighted_avg_score
    FROM
        corrections AS c
    JOIN
        projects AS p ON c.project_id = p.id
    WHERE
        c.user_id = p_user_id;

    -- Update the average_score in the users table
    -- Use IFNULL to handle cases where a user might have no corrections,
    -- which would result in a NULL average, setting it to 0.
    UPDATE users
    SET average_score = IFNULL(weighted_avg_score, 0)
    WHERE id = p_user_id;
END;
//

DELIMITER ;
