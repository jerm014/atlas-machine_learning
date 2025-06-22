#!/usr/bin/env python3
"""
This module provides functions to interact with the Star Wars API (SWAPI).

All API requests are made with SSL verification disabled due to a known
expired certificate on the public SWAPI server... (2025-06-20)
"""
import requests


def fetchUrlData(url: str, context: str = "data") -> dict | None:
    """
    Helper function to fetch JSON data from a given URL.
    """
    try:
        # Using verify=False due to expired certificate on the public API
        # server
        response = requests.get(url, verify=False)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {context} from SWAPI at {url}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while fetching {context}: {e}")
        return None


def sentientPlanets() -> list[str]:
    """
    Retrieves a list of names of home planets of all sentient species from the
    SWAPI. A species is considered sentient if its 'classification' or
    'designation' attribute contains the word 'sentient'.

    Returns:
        list: A list of unique home planet names. Returns an empty list if
              no sentient species are found or if an error occurs.
    """
    sentient_planet_names = set()  # Use a set for unique planet names
    species_url = "https://swapi.dev/api/species/"

    while species_url:
        species_data = fetchUrlData(species_url, "species")
        if not species_data:
            # If fetching species data fails, stop processing
            return []

        for species in species_data.get("results", []):
            classification = species.get("classification", "").lower()
            designation = species.get("designation", "").lower()

            # Check if the species is sentient
            if "sentient" in classification or "sentient" in designation:
                homeworld_url = species.get("homeworld")
                if homeworld_url:
                    planet_data = _fetch_url_data(
                        homeworld_url, f"homeworld for {species.get('name')}"
                    )
                    if planet_data:
                        planet_name = planet_data.get("name")
                        if planet_name:
                            sentient_planet_names.add(planet_name)

        species_url = species_data.get("next")  # URL for next page

    # Convert set to list and sort for consistent output
    return sorted(list(sentient_planet_names))
