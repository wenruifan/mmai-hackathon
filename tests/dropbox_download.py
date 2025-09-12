import argparse
import os
import zipfile

import dropbox

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a folder from Dropbox (by default downloads MMAI25Hackathon dataset)."
    )
    parser.add_argument(
        "--dropbox-folder",
        type=str,
        help="Path to the Dropbox folder to download.",
        default="/MMAI25Hackathon",
    )
    parser.add_argument(
        "--local-folder",
        type=str,
        help="Local directory to save downloaded files.",
        default="./MMAI25Hackathon",
    )
    parser.add_argument(
        "--access-token",
        type=str,
        help="Dropbox access token.",
        default=os.getenv("DROPBOX_TOKEN"),
    )
    parser.add_argument(
        "--unzip",
        action=argparse.BooleanOptionalAction,
        help="Unzip the downloaded folder and remove the zip file.",
        default=True,
    )
    args = parser.parse_args()

    # get data from dropbox (assumes DROPBOX_TOKEN is set in env)
    dbx = dropbox.Dropbox(args.access_token)

    # Create local folder if it doesn't exist yet before downloading
    os.makedirs(args.local_folder, exist_ok=True)
    dbx.files_download_zip_to_file(args.local_folder + ".zip", args.dropbox_folder)

    if args.unzip:
        with zipfile.ZipFile(args.local_folder + ".zip", "r") as zip_ref:
            zip_ref.extractall(args.local_folder + "/..")

        print(f"Unzipped to {args.local_folder}")
        os.remove(args.local_folder + ".zip")
        print(f"Removed zip file {args.local_folder}.zip")
