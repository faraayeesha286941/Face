from deepface import DeepFace
import pandas as pd

try:
    # df = DeepFace.find(img_path = "query_image.jpg", db_path = "database")
    dfs = DeepFace.find(img_path="Yuzuru_Hanyu-Sochi_2014.jpg", db_path="./database", enforce_detection=False)

    if dfs and not dfs[0].empty:
        print("Found matching faces:")
        print(dfs[0])
    else:
        print("No matching faces found.")

except Exception as e:
    print(f"An error occurred: {e}")
