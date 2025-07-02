#!/usr/bin/env python3
""" documentation """
import pymongo

methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]


def nginx_log_stats():
    """
    Provides some stats about Nginx logs stored in MongoDB.
    """
    client = None
    client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")

    db = client.logs
    nginx_collection = db.nginx

    total_logs = nginx_collection.count_documents({})

    print(f"{total_logs} logs")

    print("Methods:")

    for method in methods:
        count = nginx_collection.count_documents({"method": method})
        print(f"\tmethod {method}: {count}")

    status_get_count = nginx_collection.count_documents({"method": "GET", "path": "/status"})
    print(f"{status_get_count} status check")


if __name__ == "__main__":
    nginx_log_stats()
