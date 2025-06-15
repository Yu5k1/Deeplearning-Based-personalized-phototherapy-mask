import csv

def read_matrix_from_csv(file_path):
    """
    Read a CSV file and convert it into a light therapy input matrix (List[List[int]]).
    Skip the header row and ensure each row contains four columns: zone, red, green, blue.
    """
    matrix = []
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            if len(row) != 4:
                continue  # Skip malformed rows
            try:
                zone_data = [int(val) for val in row]
                matrix.append(zone_data)
            except ValueError:
                continue  # Skip rows with non-numeric values
    return matrix

def split_matrix_dynamically(matrix):
    """
    Split the input matrix into multiple rounds with non-repeating colors in each round
    (each zone lights up only one color per round).
    If a zone has multiple colors (R, G, B all set to 1), it will be split across multiple rounds
    with priority R > G > B.
    """
    rounds = []

    # Convert RGB info of each zone into a task queue
    zone_tasks = []
    for row in matrix:
        zone, r, g, b = row
        task = {
            "zone": zone,
            "colors": []
        }
        if r:
            task["colors"].append("R")
        if g:
            task["colors"].append("G")
        if b:
            task["colors"].append("B")
        if task["colors"]:
            zone_tasks.append(task)

    # Schedule by rounds
    while any(task["colors"] for task in zone_tasks):
        current_round = []
        for task in zone_tasks:
            if task["colors"]:
                color = task["colors"].pop(0)  # Take one color per round (FIFO)
                if color == "R":
                    current_round.append([task["zone"], 1, 0, 0])
                elif color == "G":
                    current_round.append([task["zone"], 0, 1, 0])
                elif color == "B":
                    current_round.append([task["zone"], 0, 0, 1])
        if current_round:
            rounds.append(current_round)

    return rounds

def generate_commands(matrix):
    """
    Convert a 5x4 input matrix into a list of string commands.
    Each row format: PINn:R,G,B
    """
    commands = []
    for row in matrix:
        zone, red, green, blue = row
        r = 255 if red else 0
        g = 255 if green else 0
        b = 255 if blue else 0
        zone = int(zone)
        cmd = f"PIN{str(zone + 6)}:{r},{g},{b}"
        commands.append(cmd)
    return commands

def generate_shutdown_commands(input_matrix):
    # Generate LED shutdown commands for all zones
    shutdown_matrix = [[row[0], 0, 0, 0] for row in input_matrix]
    return generate_commands(shutdown_matrix)
