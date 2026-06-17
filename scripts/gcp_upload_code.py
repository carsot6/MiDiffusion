import os
import zipfile
import subprocess
import sys

BUCKET_NAME = "ingka-b2bda-iifb-dev-self_ds_casot6"
ZIP_PATH = "/tmp/midiffusion_gcp_transfer.zip"
PROJECTS_DIR = "/Users/carlos.soto1/Projects/BLINKA"

def zip_directory(zip_file, source_dir, archive_name, excludes=None):
    if excludes is None:
        excludes = []
        
    for root, dirs, files in os.walk(source_dir):
        # Exclude directories
        dirs[:] = [d for d in dirs if not any(ex in os.path.join(root, d) for ex in excludes) and d not in ['.git', '__pycache__', '.ipynb_checkpoints', '.vscode', 'node_modules', 'output', 'predicted_results', '.venv', 'venv']]
        
        for file in files:
            file_path = os.path.join(root, file)
            # Skip excluded files
            if any(ex in file_path for ex in excludes):
                continue
            if file in ['.DS_Store', 'desktop.ini']:
                continue
                
            relative_path = os.path.relpath(file_path, source_dir)
            archive_path = os.path.join(archive_name, relative_path)
            
            # Print occasionally to show progress
            print(f"Adding: {archive_path}")
            zip_file.write(file_path, archive_path)

def create_and_upload_package():
    print(f"Creating ZIP archive of the code in {ZIP_PATH}...")
    
    with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Zip MiDiffusion
        midiffusion_dir = os.path.join(PROJECTS_DIR, "MiDiffusion")
        zip_directory(zip_file, midiffusion_dir, "MiDiffusion", 
                      excludes=['output/', 'predicted_results/', 'MiDiffusion.egg-info/'])
        
        # Zip ThreedFront
        threedfront_dir = os.path.join(PROJECTS_DIR, "ThreedFront")
        zip_directory(zip_file, threedfront_dir, "ThreedFront", 
                      excludes=['output/', 'threed_front.egg-info/'])

    print("Uploading code to Google Cloud Storage...")
    destination_gcs = f"gs://{BUCKET_NAME}/midiffusion_job/code.zip"
    
    # Use gcloud storage cp to upload
    try:
        subprocess.check_call(["gcloud", "storage", "cp", ZIP_PATH, destination_gcs])
        print(f"Successfully uploaded code package to {destination_gcs}")
        # Clean up local zip file
        os.remove(ZIP_PATH)
        print("Local temporary zip file removed.")
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        sys.exit(1)

if __name__ == "__main__":
    create_and_upload_package()
