#!/usr/bin/env python3
"""swapi project"""

import requests


def availableShips(passengerCount):
    """
    Retrieves a list of Star Wars ships from the SWAPI that can hold a given
    number of passengers.

    Args:
        passengerCount (int): The minimum number of passengers the ship must be
                              able to hold.

    Returns:
        list: A list of ship names that can accommodate the specified number
              of passengers. Returns an empty list if no ships are found or
              if an error occurs.
    """
    ships_available = []
    url = "https://swapi.dev/api/starships/"

    while url:
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
            data = response.json()

            for ship in data.get("results", []):
                # Get the passenger count for the ship.
                # Handle cases where 'passengers' might be 'unknown', 'n/a', or invalid.
                try:
                    passengers_str = ship.get("passengers", "0").replace(",", "")
                    ship_passengers = int(passengers_str)
                except ValueError:
                    ship_passengers = 0  # Treat 'unknown' or non-numeric as 0

                # Check if the ship can hold the given number of passengers
                if ship_passengers >= passengerCount:
                    ships_available.append(ship["name"])

            url = data.get("next")  # Get the URL for the next page, or None if no more pages
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data from SWAPI: {e}")
            return [] # Return empty list on network or API error
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return [] # Return empty list on other unexpected errors

    return ships_available
