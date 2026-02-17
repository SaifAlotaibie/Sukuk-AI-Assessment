import os
import fitz
from PIL import Image

input_folder = "Data Set"
output_folder = "pages_raw"

os.makedirs(output_folder, exist_ok=True)

total_pages = 0
dpi = 200
zoom = dpi / 72

for file in os.listdir(input_folder):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(input_folder, file)
        print(f"Processing {file} ...")

        doc = fitz.open(pdf_path)

        for i, page in enumerate(doc):
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            image_name = f"{file[:-4]}_page_{i+1}.jpg"
            img.save(os.path.join(output_folder, image_name), "JPEG")
            total_pages += 1

        doc.close()

print(f"\nDone! Total pages converted: {total_pages}")
