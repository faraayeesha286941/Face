from deepface import DeepFace

try:
    # result = DeepFace.verify(img1_path = "image1.jpg", img2_path = "image2.jpg")
    result = DeepFace.verify(img1_path="path/to/your/image1.jpg", img2_path="path/to/your/image2.jpg")

    if result["verified"]:
        print("The two images are of the same person.")
    else:
        print("The two images are of different people.")

    print(result)

except Exception as e:
    print(f"An error occurred: {e}")
