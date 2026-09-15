import csv
import os


def write_episode_csv(path, episodes):
    """Write one row per evaluated episode: episode index, success (0/1), reward, steps."""
    with open(path, 'w', newline='') as fil:
        writer = csv.writer(fil)
        writer.writerow(['episode', 'success', 'reward', 'steps'])
        for ep in episodes:
            writer.writerow([ep['episode'], ep['success'], ep['reward'], ep['steps']])


def append_csv_row(path, header, row):
    """Append one row to a CSV, writing the header first if the file doesn't exist yet."""
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as fil:
        writer = csv.writer(fil)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)
