"""
convert_tif_to_png.py
========================

WHAT THIS SCRIPT DOES (plain-English summary)
-----------------------------------------------
Looks inside a folder for image files ending in ".tif" or ".tiff", and
saves a copy of each one next to it as a ".png" file instead. PNG is a
more widely-supported format (e.g. it previews directly in most file
browsers, web pages, and chat apps, unlike TIFF), so this is just a
format conversion -- the picture itself doesn't change.

This is a LOSSLESS conversion for ordinary 8-bit images (the normal kind
a microscope viewer, like RELION's Zap window, saves as a screenshot):
every pixel comes out identical, just repackaged into a different file
format. Nothing about the image content is changed.

By default, the original .tif files are left alone (so nothing is
deleted unless you ask for it) -- see the `--delete-originals` option
below.

HOW TO RUN THIS SCRIPT
-------------------------
Convert every .tif/.tiff file in a folder, keeping the originals:

    python3 convert_tif_to_png.py --i "pictures tomograms"

Convert every .tif/.tiff file AND delete the original .tif files
afterwards (only once each conversion has been checked to have worked):

    python3 convert_tif_to_png.py --i "pictures tomograms" --delete-originals

If you don't say --i, it looks in the current folder.
"""

# --- Tools this script needs ---
import argparse   # reads --i and --delete-originals typed on the command line
import os         # for listing files in a folder and building file paths

from PIL import Image  # "Pillow", the standard Python library for opening/saving images


def find_tif_files(folder_path):
    """
    Looks inside `folder_path` and returns a list of filenames that end in
    ".tif" or ".tiff" (matched without caring about UPPER/lower case).
    """
    tif_filenames = []
    for filename in sorted(os.listdir(folder_path)):
        lowercase_name = filename.lower()
        if lowercase_name.endswith(".tif") or lowercase_name.endswith(".tiff"):
            tif_filenames.append(filename)
    return tif_filenames


def convert_one_file(folder_path, tif_filename):
    """
    Opens one .tif file and saves it as a .png file with the same name
    (just a different file extension), in the same folder.

    Returns the full path to the new .png file.
    """
    tif_path = os.path.join(folder_path, tif_filename)

    # Build the new filename by removing ".tif"/".tiff" and adding ".png".
    name_without_extension = os.path.splitext(tif_filename)[0]
    png_filename = name_without_extension + ".png"
    png_path = os.path.join(folder_path, png_filename)

    image = Image.open(tif_path)
    image.save(png_path)

    return png_path


def main():
    # --- Read what the user typed on the command line (or use the defaults) ---
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--i", default=".",
                         help="folder to look in for .tif/.tiff files (default: current folder)")
    parser.add_argument("--delete-originals", action="store_true",
                         help="if given, deletes each .tif file after it has been converted "
                              "(default: leaves the original .tif files alone)")
    args = parser.parse_args()

    # --- Step 1: find every .tif/.tiff file in the folder ---
    tif_filenames = find_tif_files(args.i)

    if not tif_filenames:
        print(f"No .tif or .tiff files found in {args.i}")
        return

    # --- Step 2: convert each one, one at a time ---
    for tif_filename in tif_filenames:
        png_path = convert_one_file(args.i, tif_filename)
        print(f"{tif_filename}  ->  {os.path.basename(png_path)}")

        if args.delete_originals:
            tif_path = os.path.join(args.i, tif_filename)
            os.remove(tif_path)
            print(f"  (deleted original {tif_filename})")

    print(f"\nConverted {len(tif_filenames)} file(s).")


# This just means "if this file is run directly, call main()". It's a
# standard Python convention and can be ignored.
if __name__ == "__main__":
    main()
