# Copyright © 2024- Frello Technology Private Limited

import json
import requests
from typing import Optional, Dict, Any, Tuple, List

from tuneapi.utils import ENV, SimplerTimes as stime, from_json, to_json
from tuneapi.types import Thread, human, Message


https://studio.rc.tune.app/v1/threads?limit=1000