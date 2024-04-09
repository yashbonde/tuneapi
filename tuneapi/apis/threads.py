# Copyright © 2024- Frello Technology Private Limited

import json
import requests
from typing import Optional, Dict, Any, Tuple, List

from tuneapi.utils import ENV, SimplerTimes as stime, from_json, to_json
from tuneapi import types as tt


class Threads:
    def upload_thread(
        self,
        thread: tt.Thread,
        tune_org_id: Optional[str] = None,
        tune_api_key: Optional[str] = None,
    ):
        tune_api_key = tune_api_key or ENV.TUNE_API_KEY()
        tune_org_id = tune_org_id or ENV.TUNE_ORG_ID("")
        headers = {"x-tune-key": tune_api_key}

        if tune_org_id:
            headers["X-Organization-Id"] = tune_org_id
