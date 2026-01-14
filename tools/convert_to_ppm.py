from PIL import Image
import sys
import os

def convert_to_ppm(input_path, output_path=None):
    if not os.path.exists(input_path):
        print("Input file not found")
        return

    if output_path is None:
        name = os.path.splitext(input_path)[0]
        output_path = name + ".ppm"

    img = Image.open(input_path)
    img = img.convert("RGB")  # ensure RGB format

    img.save(output_path, format="PPM")

    print(f"Converted: {input_path} -> {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_to_ppm.py input_image [output.ppm]")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        convert_to_ppm(input_file, output_file)