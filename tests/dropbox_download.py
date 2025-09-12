"""Download a Dropbox folder for CI integration tests.

This small CLI downloads a Dropbox folder as a zip to ``<local_folder>.zip`` and
optionally extracts it beside the destination folder, removing the zip afterwards.

Usage examples
--------------
- Using environment secret (preferred in CI):
  ``python -m tests.dropbox_download "/MMAI25Hackathon" "MMAI25Hackathon" --unzip``
  where ``DROPBOX_ACCESS_TOKEN`` (or ``DROPBOX_TOKEN``) is set in the environment.
- Providing the token explicitly:
  ``python -m tests.dropbox_download "/remote/path" "local_dir" --access_token <TOKEN> --unzip``
"""

import argparse
import os
import zipfile

import dropbox

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a folder from Dropbox.")
    parser.add_argument("dropbox_folder", type=str, help="Path to the Dropbox folder to download.")
    parser.add_argument("local_folder", type=str, help="Local directory to save downloaded files.")
    # Accept either DROPBOX_ACCESS_TOKEN or DROPBOX_TOKEN
    default_token = os.getenv("DROPBOX_ACCESS_TOKEN") or os.getenv("DROPBOX_TOKEN")
    parser.add_argument("--access_token", type=str, default=default_token, help="Dropbox access token.")
    parser.add_argument("--unzip", action="store_true", help="Unzip and remove the zip afterwards.")
    args = parser.parse_args()

    # get data from dropbox
    dbx = dropbox.Dropbox(args.access_token)

    # Ensure local folder exists
    os.makedirs(args.local_folder, exist_ok=True)
    zip_path = args.local_folder + ".zip"
    dbx.files_download_zip_to_file(zip_path, args.dropbox_folder)

    if args.unzip:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.join(args.local_folder, ".."))

        print(f"Unzipped to {args.local_folder}")
        os.remove(zip_path)
        print(f"Removed zip file {zip_path}")
