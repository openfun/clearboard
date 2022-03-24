"""Module implementing functions to load and save coord in file"""


def save_coords(path: str, coords: list[list[int]]):
    """Save coords in a file. All directories in path must exist"""
    with open(path, "w", encoding="utf-8") as file:
        for point in coords:
            file.write(f"{point[0]},{point[1]}")
            file.write("\n")
        file.close()


def get_coords(path: str):
    """Load coords from a file"""
    with open(path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        coords = map(lambda line: line.split(","), lines)
        coords = list(map(lambda coord: [int(coord[0]), int(coord[1])], coords))
        file.close()
    return coords
