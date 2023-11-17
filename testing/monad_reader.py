

def translation(coords: dict, vector: dict):
    print(f"Coordinates: {coords} | Translation vector: {vector}")
    new_coords = {
        "x": coords["x"]  + vector["x"],
        "y": coords["y"]  + vector["y"],
    }

    if (new_coords["x"] >= 0) and  (new_coords["y"] >= 0) and (new_coords["x"] <= 10) and  (new_coords["y"] <= 10):
        return new_coords  # in-bounds
    else:
        return Nothing