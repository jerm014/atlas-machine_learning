#!/usr/bin/env python3
""" documentation """
import pymongo


def update_topics(mongo_collection, name, topics):
    """
    change topics of school documents based on the school name
    """
    query = {"name": name}
    update_operation = {"$set": {"topics": topics}}
    mongo_collection.update_many(query, update_operation)
