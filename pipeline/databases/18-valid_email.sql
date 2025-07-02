-- create a trigger that resets the 'valid_email' attribute to 0
-- when the 'email' attribute of a user is changed.

DELIMITER //

CREATE TRIGGER reset_valid_email_on_email_change
BEFORE UPDATE ON users
FOR EACH ROW
BEGIN
    -- Check if the email address has changed
    IF OLD.email <> NEW.email THEN
        SET NEW.valid_email = 0;
    END IF;
END;
//

DELIMITER ;
