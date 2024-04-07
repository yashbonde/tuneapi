# Copyright © 2024- Frello Technology Private Limited

import os

from tuneapi.utils.env import ENV
from tuneapi.utils.logger import (
    get_logger,
    warning_with_fix,
    logger,
)
from tuneapi.utils.serdeser import (
    to_json,
    from_json,
    dict_to_structpb,
    structpb_to_dict,
    to_b64,
    from_b64,
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
    load_module_from_path,
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
    hashstr,
    encrypt,
    decrypt,
)
from tuneapi.utils.subway import (
    SubwayClientError,
    SubwayServerError,
    Subway,
    get_session,
    get_subway,
)

# sometimes services would need to not have tune dir, in which case anyone using this package can just
# turn it off.
if not ENV.TUNE_DISABLE_HOME_DIR("0") == "1":
    # create the default folder for all the things that this library needs
    os.makedirs(ENV.DEFAULT_FOLDER(os.path.expanduser("~/.tuneapi")), exist_ok=True)
    if ENV.BLOB_ENGINE() == "local":
        os.makedirs(ENV.BLOB_STORAGE(), exist_ok=True)
