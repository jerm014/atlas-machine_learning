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
    Fetches and displays information about the latest SpaceX launch.
    """
    latest_launch_url = "https://api.spacexdata.com/v5/launches/latest"
    rocket_url_template = "https://api.spacexdata.com/v4/rockets/{}"
    launchpad_url_template = "https://api.spacexdata.com/v4/launchpads/{}"

    # Get the latest launch data
    latest_launch = get_spacex_data(latest_launch_url)

    if not latest_launch:
        sys.exit(1)

    # initial information
    launch_name = latest_launch.get('name', 'N/A')
    date_local_str = latest_launch.get('date_local', 'N/A')
    rocket_id = latest_launch.get('rocket')
    launchpad_id = latest_launch.get('launchpad')

    # fetch rocket name
    rocket_name = "N/A"
    if rocket_id:
        rocket_data = get_spacex_data(rocket_url_template.format(rocket_id))
        if rocket_data:
            rocket_name = rocket_data.get('name', 'N/A')

    # fetch launchpad name and locality
    launchpad_name = "N/A"
    launchpad_locality = "N/A"
    if launchpad_id:
        launchpad_data = get_spacex_data(
            launchpad_url_template.format(launchpad_id)
        )
        if launchpad_data:
            launchpad_name = launchpad_data.get('name', 'N/A')
            launchpad_locality = launchpad_data.get('locality', 'N/A')

    try:
        dt_object = datetime.datetime.fromisoformat(date_local_str)
        formatted_date = dt_object.strftime('%Y-%m-%d %H:%M:%S')
    except ValueError:
        # well, nevermind
        formatted_date = date_local_str

    output = (f"{launch_name} ({formatted_date}) {rocket_name} - "
              f"{launchpad_name} ({launchpad_locality})")
    print(output)


if __name__ == '__main__':
    main()
