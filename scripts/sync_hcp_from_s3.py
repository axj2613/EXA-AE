"""One-time sync of the HCP fMRI corpus from S3 to a local directory, preserving the
subject-organized layout (<local_root>/<subject_id>/<recording>.npz) that HCPWindowDataset
expects. Run this locally once, then upload the resulting directory as a private Kaggle Dataset;
the training notebook mounts it read-only at /kaggle/input/... and never touches S3 again.

Mirrors the S3 access pattern in get_cls.py (bucket hcp-processed-inads, prefix aal_424/, keys
of the form aal_424/<subject_id>/<recording>.npz, boto3 profile "ajha"). It is resumable: keys
already present locally with the right size are skipped, so an interrupted sync can be re-run.

Usage:
    python scripts/sync_hcp_from_s3.py <local_root> [--bucket ...] [--prefix ...] [--profile ...]
"""

import argparse
import os

import boto3
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("local_root", help="local directory to mirror the corpus into")
    parser.add_argument("--bucket", default="hcp-processed-inads")
    parser.add_argument("--prefix", default="aal_424/")
    parser.add_argument("--profile", default="ajha")
    args = parser.parse_args()

    s3 = boto3.Session(profile_name=args.profile).client("s3")

    # enumerate all .npz object keys under the prefix
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for response in paginator.paginate(Bucket=args.bucket, Prefix=args.prefix):
        for obj in response.get("Contents", []):
            if obj["Key"].endswith(".npz"):
                keys.append((obj["Key"], obj["Size"]))
    print(f"found {len(keys)} .npz objects under s3://{args.bucket}/{args.prefix}")

    downloaded, skipped = 0, 0
    for key, size in tqdm(keys, desc="syncing"):
        # keys look like "aal_424/<subject_id>/<recording>.npz"; strip the prefix to get the
        # relative subject/recording path and mirror it locally.
        relative = key[len(args.prefix):] if key.startswith(args.prefix) else key
        local_path = os.path.join(args.local_root, relative)

        if os.path.exists(local_path) and os.path.getsize(local_path) == size:
            skipped += 1
            continue

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(args.bucket, key, local_path)
        downloaded += 1

    print(f"done: {downloaded} downloaded, {skipped} already present, into '{args.local_root}'")


if __name__ == "__main__":
    main()
