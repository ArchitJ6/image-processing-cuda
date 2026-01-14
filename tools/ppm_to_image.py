from PIL import Image
import sys
import os

def ppm_to_image(input_path, output_path=None):
    if not os.path.exists(input_path):
        print("File not found")
        return

    if output_path is None:
        name = os.path.splitext(input_path)[0]
        output_path = name + ".png"   # default output

    img = Image.open(input_path)
    img.save(output_path)

    print(f"Converted: {input_path} -> {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ppm_to_image.py input.ppm [output.png]")
    else:
        ppm_to_image(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)