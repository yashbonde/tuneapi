# Copyright © 2024- Frello Technology Private Limited

import os

from tuneapi.utils.env import ENV
from tuneapi.utils.logger import (
    logger,
    get_logger,
    warning_with_fix,
)
from tuneapi.utils.serdeser import (
    to_json,
    from_json,
    dict_to_structpb,
    structpb_to_dict,
)
from tuneapi.utils.networking import (
    UnAuthException,
    DoNotRetryException,
    exponential_backoff,
)
from tuneapi.utils.parallel import (
    batched,
    threaded_map,
)
from tuneapi.utils.randomness import (
    get_random_string,
)
from tuneapi.utils.fs import (
    get_files_in_folder,
    folder,
    joinp,
)
from tuneapi.utils.terminal import (
    hr,
)
from tuneapi.utils.mime import (
    get_mime_type,
)
from tuneapi.utils.misc import (
    SimplerTimes,
    unsafe_exit,
    safe_exit,
)
from tuneapi.utils.subway import (
    SubwayClientError,
    SubwayServerError,
    Subway,
    get_session,
)


os.makedirs(ENV.DEFAULT_FOLDER(os.path.expanduser("~/.tuneapi")), exist_ok=True)
if ENV.BLOB_ENGINE() == "local":
    os.makedirs(ENV.BLOB_STORAGE(), exist_ok=True)
