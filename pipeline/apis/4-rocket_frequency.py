#!/usr/bin/env python3
"""SpaceX Stuff"""
import requests
import sys
import datetime


def get_spacex_data(url):
    """
    Fetches JSON data from a given URL, handles common HTTP errors.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e.response.status_code} - "
              f"{e.response.reason}", file=sys.stderr)
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error occurred: {e}", file=sys.stderr)
        return None
    except requests.exceptions.Timeout as e:
        print(f"Timeout error occurred: {e}", file=sys.stderr)
        return None
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the request: {e}", file=sys.stderr)
        return None
    except ValueError:
        print("Error: Could not decode JSON response.", file=sys.stderr)
        return None


def main():
    """
    Fetches all SpaceX launches and displays the number of launches per rocket,
    sorted by launch count (descending) and then rocket name (ascending).
    """
    launches_url = "https://api.spacexdata.com/v4/launches"
    rockets_url = "https://api.spacexdata.com/v4/rockets"

    all_launches = get_spacex_data(launches_url)
    if not all_launches:
        sys.exit(1)

    all_rockets = get_spacex_data(rockets_url)
    if not all_rockets:
        sys.exit(1)

    # dictionary to map rocket IDs to their names
    rocket_id_to_name = {
        rocket.get('id'): rocket.get('name', 'Unknown Rocket')
        for rocket in all_rockets
    }

    # count launches per rocket ID
    launch_counts = {}
    for launch in all_launches:
        rocket_id = launch.get('rocket')
        if rocket_id:
            launch_counts[rocket_id] = launch_counts.get(rocket_id, 0) + 1

    # sort:
    # Convert rocket IDs to names and create a list of tuples
    # (rocket_name, launch_count)
    processed_counts = []
    for rocket_id, count in launch_counts.items():
        rocket_name = rocket_id_to_name.get(rocket_id, 'Unknown Rocket')
        processed_counts.append((rocket_name, count))

    # Sort the results:
    # 1. by launch count in descneding order (negative for reverse sort)
    # 2. by rocket name in ascending order (alphabetical)
    processed_counts.sort(key=lambda x: (-x[1], x[0]))

    for name, count in processed_counts:
        print(f"{name}: {count}")


if __name__ == '__main__':
    main()
