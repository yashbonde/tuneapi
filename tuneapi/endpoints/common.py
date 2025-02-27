# Copyright © 2024-2025 Frello Technology Private Limited
# Copyright © 2025-2025 Yash Bonde github.com/yashbonde
# MIT License

from functools import cache

import tuneapi.utils as tu


@cache
def get_sub(
    base_url,
    tune_org_id: str,
    tune_api_key: str,
) -> tu.Subway:

    sess = tu.Subway._get_session()
    sess.headers.update({"x-tune-key": tune_api_key})
    if tune_org_id:
        sess.headers.update({"X-Org-Id": tune_org_id})
    return tu.Subway(base_url, sess)
