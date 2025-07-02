#!/usr/bin/env python3
""" documentation """
import pymongo

methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]


def nginx_log_stats():
    """
    Provides some stats about Nginx logs stored in MongoDB.
    Displays total logs, counts for different HTTP methods,
    and a specific count for '/status' GET requests.
    Handles cases where the collection might be empty or connection fails.
    """
    def print_zero_stats():
        print("0 logs")
        print("Methods:")
        for method in methods:
            print(f"\tmethod {method}: 0")
        print("0 status check")

    client = None
    try:
        client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")

        db = client.logs
        nginx_collection = db.nginx

        total_logs = nginx_collection.count_documents({})
        if total_logs==0:
            print_zero_stats()

        print(f"{total_logs} logs")

        print("Methods:")

        for method in methods:
            count = nginx_collection.count_documents({"method": method})
            print(f"\tmethod {method}: {count}")

        status_get_count = nginx_collection.count_documents({"method": "GET", "path": "/status"})
        print(f"{status_get_count} status check")

    except pymongo.errors.ConnectionFailure as e:
        print(f"Error: Could not connect to MongoDB. {e}")
        print_zero_stats()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print_zero_stats()
    finally:
        if client:
            client.close()
