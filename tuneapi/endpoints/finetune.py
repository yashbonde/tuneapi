"""
Use the `Threads API <https://studio.tune.app/docs/concepts/finetuning>`_ for training models on the Tune AI platform.
"""

# Copyright © 2024- Frello Technology Private Limited

import os
import requests
from typing import Optional, List
from dataclasses import dataclass, asdict

from tuneapi import utils as tu
from tuneapi import types as tt
from tuneapi import apis as ta

from tuneapi.endpoints.common import get_sub


@dataclass
class FTDataset:
    path: str
    type: str


class FinetuningAPI:
    def __init__(
        self,
        tune_org_id: str = None,
        tune_api_key: str = None,
        base_url: str = "https://studio.tune.app/",
    ):
        self.tune_org_id = tune_org_id or tu.ENV.TUNEORG_ID()
        self.tune_api_key = tune_api_key or tu.ENV.TUNEAPI_TOKEN()
        self.base_url = base_url
        if not self.tune_api_key:
            raise ValueError("Either pass tune_api_key or set Env var TUNEAPI_TOKEN")
        self.sub = get_sub(
            base_url + "tune.Studio/", self.tune_org_id, self.tune_api_key
        )

    def upload_dataset_file(self, filepath: str, name: str):
        # first we get the presigned URL for the dataset
        data = self.sub.UploadDataset(
            "post",
            json={
                "auth": {
                    "organization": self.tune_org_id,
                },
                "dataset": {
                    "name": name,
                    "contentType": "application/jsonl",
                    "datasetType": "chat",
                    "size": os.stat(filepath).st_size,
                },
            },
        )
        with open(filepath, "rb") as f:
            files = {"file": (filepath, f)}
            http_response = requests.post(
                data["code"]["s3Url"],
                data=data["code"]["s3Meta"],
                files=files,
            )
            if http_response.status_code == 204:
                tu.logger.info("Upload successful!")
            else:
                raise ValueError(
                    f"Upload failed with status code: {http_response.status_code} and response: {http_response.text}"
                )

        return FTDataset(path="datasets/chat/" + name, type="relic")

    def upload_dataset(
        self,
        threads: tt.ThreadsList | str,
        name: str,
        ds_folder: str = "tuneds",
        override: bool = False,
    ) -> FTDataset:
        """
        Upload a list of threads to the Tune AI platform to be used for finetuning a model.

        Args:
            - dataset: A list of threads to be uploaded.
            - filepath: The filepath to save the dataset to for uploading.
        """
        if isinstance(threads, str):
            threads = tt.ThreadsList.from_disk(threads)
        else:
            if not len(threads):
                raise ValueError("Threads list cannot be empty.")
        os.makedirs(ds_folder, exist_ok=True)
        fp = threads.to_disk(
            ds_folder + "/" + name,
            fmt="ft",
            override=override,
        )
        tu.logger.info(f"Dataset saved to {fp}")

        # first we get the presigned URL for the dataset
        ds_name = name + ".jsonl"
        data = self.sub.UploadDataset(
            "post",
            json={
                "auth": {
                    "organization": self.tune_org_id,
                },
                "dataset": {
                    "name": ds_name,
                    "contentType": "application/jsonl",
                    "datasetType": "chat",
                    "size": os.stat(fp).st_size,
                },
            },
        )
        with open(fp, "rb") as f:
            files = {"file": (fp, f)}
            http_response = requests.post(
                data["code"]["s3Url"],
                data=data["code"]["s3Meta"],
                files=files,
            )
            if http_response.status_code == 204:
                tu.logger.info("Upload successful!")
            else:
                raise ValueError(
                    f"Upload failed with status code: {http_response.status_code} and response: {http_response.text}"
                )

        return FTDataset(path="datasets/chat/" + ds_name, type="relic")

    def finetune(
        self,
        name: str,
        datasets: List[FTDataset],
        base_model: str = "tune:5fmycsn2",
        num_epochs: int = 3,
        lr: float = 1e-4,
        training_config: Optional[dict] = {},
    ) -> ta.TuneModel:
        """
        Read more about supported models `here <https://studio.tune.app/docs/concepts/finetuning#creating-a-fine-tune-job>`_.
        """
        datasets_list = []
        for x in datasets:
            if not isinstance(x, FTDataset) and isinstance(x, dict):
                datasets_list.append(asdict(FTDataset(**x)))
            elif not isinstance(x, FTDataset):
                raise ValueError("Datasets should be of type FTDataset or a dict")
            else:
                datasets_list.append(asdict(x))

        data = self.sub.CreateFinetuneJob(
            "post",
            json={
                "auth": {"organization": self.tune_org_id},
                "job": {
                    "name": name,
                    "baseModel": base_model,
                    "datasets": datasets_list,
                    "resource": {"gpu": "nvidia-l4", "gpuCount": "1", "maxRetries": 1},
                    "trainingConfig": {
                        "num_epochs": num_epochs,
                        "learning_rate": lr,
                        **training_config,
                    },
                },
            },
        )
        job_id = data["id"]

        tu.logger.info(
            f"Finetuning job created with ID: {job_id}. Check progress at: {self.base_url}finetuning/{job_id}?org_id={self.tune_org_id}"
        )
        model = ta.TuneModel(
            id=data["name"] + "-model-" + job_id,
            api_token=self.tune_api_key,
            org_id=self.tune_org_id,
        )
        model.set_api_token(self.tune_api_key)
        return model

    def get_job(self, job_id: str):
        data = self.sub.GetFinetuneJob(
            "post",
            json={
                "auth": {"organization": self.tune_org_id},
                "id": job_id,
            },
        )
        return data
