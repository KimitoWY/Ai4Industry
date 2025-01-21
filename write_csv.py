import os
import csv

def generate_csv_from_directory(directory_path, output_csv_path):
    """
    Generate a CSV file with filenames and a default class '0' from the specified directory.

    Args:
        directory_path (str): Path to the directory containing the images.
        output_csv_path (str): Path to the output CSV file.
    """
    try:
        # Open the CSV file for writing
        with open(output_csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)

            # Write the header
            writer.writerow(['name', 'class'])

            # Loop through files in the directory
            for filename in os.listdir(directory_path):
                # Ensure it's a file (not a folder)
                if os.path.isfile(os.path.join(directory_path, filename)):
                    # Add the filename and class (0) to the CSV
                    writer.writerow([filename, 0])

        print(f"CSV file generated successfully: {output_csv_path}")

    except Exception as e:
        print(f"An error occurred: {e}")
