#!/usr/bin/env python3
""" documentation """
import pymongo


def nginx_log_stats():
    """
    Provides some stats about Nginx logs stored in MongoDB.
    Displays total logs, counts for different HTTP methods,
    and a specific count for '/status' GET requests.
    Handles cases where the collection might be empty or connection fails.
    """
    client = None
    try:
        client = pymongo.MongoClient("mongodb://127.0.0.1:27017/")

        db = client.logs
        nginx_collection = db.nginx

        total_logs = nginx_collection.count_documents({})
        print(f"{total_logs} logs")

        print("Methods:")
        methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
        for method in methods:
            count = nginx_collection.count_documents({"method": method})
            print(f"\tmethod {method}: {count}")

        status_get_count = nginx_collection.count_documents({"method": "GET", "path": "/status"})
        print(f"{status_get_count} status check")

    except pymongo.errors.ConnectionFailure as e:
        print(f"Could not connect to MongoDB: {e}")
        print("0 logs")
        print("Methods:")
        methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
        for method in methods:
            print(f"\tmethod {method}: 0")
        print("0 status check")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("0 logs")
        print("Methods:")
        methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
        for method in methods:
            print(f"\tmethod {method}: 0")
        print("0 status check")
    finally:
        if client:
            client.close()
