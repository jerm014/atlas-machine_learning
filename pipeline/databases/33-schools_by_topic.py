#!/usr/bin/env python3
""" documentation """
import pymongo


def schools_by_topic(mongo_collection, topic):
    """
    returns list of schools having a specific topic
    """
    schools = []
    cursor = mongo_collection.find({"topics": topic})
    
    for school in cursor:
        schools.append(school)
        
    return schools
