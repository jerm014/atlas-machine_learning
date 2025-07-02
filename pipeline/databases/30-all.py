#!/usr/bin/env python
""" documentation """

import pymongo


def list_all(mongo_collection):
    """
    Lists all documents in a MongoDB collection.
    """
    documents = []
    # Find all documents in the collection
    cursor = mongo_collection.find({})
    
    # Iterate through the cursor and add each document to the list
    for document in cursor:
        documents.append(document)
        
    return documents
