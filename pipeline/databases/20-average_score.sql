-- create a stored procedure ComputeAverageScoreForUser
-- that computes and stores the average score for a given user

DELIMITER //

CREATE PROCEDURE ComputeAverageScoreForUser(
    IN p_user_id INT
)
BEGIN
    DECLARE avg_score_val FLOAT;

    -- Calculate the average score for the specified user
    SELECT AVG(score)
    INTO avg_score_val
    FROM corrections
    WHERE user_id = p_user_id;

    -- Update the average_score in the users table
    -- Use IFNULL to handle cases where a user might have no corrections?
    -- (AVG returns NULL)
    UPDATE users
    SET average_score = IFNULL(avg_score_val, 0)
    WHERE id = p_user_id;
END;
//

DELIMITER ;
