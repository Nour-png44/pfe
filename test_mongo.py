from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")

db = client["ebsct"]
collection = db["user"]

print("Databases:", client.list_database_names())
print("Collections:", db.list_collection_names())
print("Count:", collection.count_documents({}))