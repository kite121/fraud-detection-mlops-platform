import minio
import os
from dotenv import load_dotenv

load_dotenv()

# create a client
client = minio.Minio(
    endpoint="localhost:9000",
    access_key=os.getenv("MINIO_ROOT_USER"),
    secret_key=os.getenv("MINIO_ROOT_PASSWORD"),
    secure=False
)


def create_bucket(bucket_name):
    if client.bucket_exists(bucket_name):
        print("Bucket {} already exists".format(bucket_name))
        return
    print(f'Creating bucket {bucket_name}')
    return client.make_bucket(bucket_name)


b_name = "test-bucket"
f_name = r"C:\Users\valee\PycharmProjects\DNP_project(MLOps)\ssh.jpg"
o_name = "mem.jpg"

create_bucket(b_name)
res = client.fput_object(bucket_name=b_name, object_name=f_name, file_path=f_name)
print(
    "Created {0} with etag: {1}, version-id : {2}".format(
        res.object_name,res.etag,res.version_id
    )
)
