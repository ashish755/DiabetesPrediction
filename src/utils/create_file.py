import os
import shutil

upload_folder = "/tmp/documents"


def create_file(filedata):
    global upload_folder
    file_object = filedata.file
    # create empty file to copy the file_object to
    if not os.path.exists(upload_folder):
        os.mkdir(upload_folder)
    path = os.path.join(upload_folder, filedata.filename)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file_object, buffer)

    return path
