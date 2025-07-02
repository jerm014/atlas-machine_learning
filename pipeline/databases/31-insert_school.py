#!/usr/bin/env python3
""" documentation """
import pymongo


def insert_school(mongo_collection, **kwargs):
    """
    inserts a new document into a MongoDB collection based on keyword arguments
    """
    if not kwargs:
        # If no kwargs are provided, insert an empty document or handle as an error.
        # For simplicity, we'll insert an empty document here.
        result = mongo_collection.insert_one({})
    else:
        # The kwargs directly form the document to be inserted
        result = mongo_collection.insert_one(kwargs)
    
    return result.inserted_id
