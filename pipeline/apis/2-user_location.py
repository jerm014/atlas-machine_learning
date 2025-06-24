#!/usr/bin/env python3
"""user location"""
import requests
import sys
import datetime


def main():
    """
    Fetches the location of a GitHub user from the provided API URL.
    Handles different HTTP status codes for user existence and rate limiting.
    What else do you want?
    """
    # Check if the correct number of command-line arguments are provided
    if len(sys.argv) != 2:
        print("Get it right: ./2-user_location.py <GitHub_API_URL>")
        sys.exit(1)

    # get the API URL from the command-line arguments
    url = sys.argv[1]

    try:
        # make a GET request to the GitHub API
        response = requests.get(url)

        # check the status code of the response
        if response.status_code == 200:
            # If successful (200 OK), parse the JSON response
            user_data = response.json()
            # Print the 'location' field, or "None" if it doesn't exist
            print(user_data.get('location', 'None'))
        elif response.status_code == 404:
            # If user not found
            print("Not found")
        elif response.status_code == 403:
            # if rate limit exceeded (403 Forbidden) then get the
            # X-Ratelimit-Reset header, which is a Unix timestamp
            reset_timestamp = int(
                response.headers.get('X-Ratelimit-Reset', 0)
            )

            # get current Unix timestamp
            current_timestamp = int(datetime.datetime.now().timestamp())

            # calculate time difference in seconds
            time_difference_seconds = reset_timestamp - current_timestamp

            # convert seconds to minutes, make sure it's at least 0
            minutes_to_reset = max(0, time_difference_seconds // 60)

            print(f"Reset in {minutes_to_reset} min")
        else:
            # handle any other unexpected status codes?
            print(f"Error: Received status code {response.status_code}")

    except requests.exceptions.RequestException as e:
        # request-related errors
        print(f"An error occurred during the request: {e}")
    except ValueError:
        # errors if the JSON parsing fails
        print("Error: Could not decode JSON response.")
    except Exception as e:
        # other unexpected errors
        print(f"An unexpected error occurred: {e}")


# Ensure the main function is called only when the script is executed directly
if __name__ == '__main__':
    main()
