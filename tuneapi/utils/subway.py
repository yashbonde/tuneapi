# Copyright © 2024- Frello Technology Private Limited
# REMEMBER: nothing from outside tune should be imported in utils

from requests import Session
from typing import Dict, Any, Optional, Tuple

from tuneapi.utils.logger import logger
from tuneapi.utils.misc import SimplerTimes


class SubwayClientError(Exception):
    """Raised if 399 < status_code < 500"""

    def __init__(self, *args, code: str):
        self.code = code
        super().__init__(*args)


class SubwayServerError(Exception):
    """Raised if 499 < status_code < 600"""

    def __init__(self, *args, code: str):
        self.code = code
        super().__init__(*args)


def get_session(
    token: Optional[str] = "",
    bearer: Optional[str] = "",
    **headers,
) -> Session:
    sess = Session()
    if token:
        sess.headers.update({"token": token})
    if bearer:
        sess.headers.update({"Authorization": f"Bearer {bearer}"})
    if headers:
        sess.headers.update(headers)
    return sess


class Subway:
    """
    Simple code that allows writing APIs by `.attr.ing` them. This is inspired from gRPC style functional calls which
    hides the complexity of underlying networking. This is useful when you are trying to debug live server directly.

    Note:
      User is solely responsible for checking if the certain API endpoint exists or not. This simply wraps the API
      calls and does not do any validation.

    Args:
      _url (str): The url to use for the client
      _session (requests.Session): The session to use for the client
    """

    def __init__(self, _url: str, _session: Session):
        self._url = _url.rstrip("/")
        self._session = _session
        self._last_update = SimplerTimes.get_now_i64()

    def __repr__(self):
        return self._url

    def __getattr__(self, attr: str):
        # https://stackoverflow.com/questions/3278077/difference-between-getattr-vs-getattribute
        return Subway(f"{self._url}/{attr}", self._session)

    def u(self, attr: str) -> "Subway":
        """In cases where the api might start with a number you cannot write in python, this method can be used to
        access the attribute.

        Example:
          >>> stub.9jisjfi      # python will cry, invalid syntax: cannot start with a number
          >>> stub.u('9jisjfi') # do this instead

        Args:
          attr (str): The attribute to access

        Returns:
          Subway: The new subway object
        """
        return getattr(self, attr)

    def _renew_session(self):
        """Renew the session"""
        _session = Session()
        if self._session.headers:
            _session.headers.update(**self._session.headers)
        self._session = _session

    @staticmethod
    def _get_session(
        token: Optional[str] = "",
        bearer: Optional[str] = "",
        **headers,
    ) -> Session:
        return get_session(token=token, bearer=bearer, **headers)

    def __call__(
        self,
        method="get",
        json={},
        trailing="",
        data=None,
        params: Dict = {},
        html: bool = False,
        verbose=False,
        ret_code: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, Any], ...]:
        """Call the API endpoint as if it is a function.

        Usage:

        >>> from tuneapi.utils import Subway, get_session
        >>> sub = Subway("http://0.0.0.0:8080", get_session())
        >>> fn = sub.swagger
        >>> data, status = fn()
        >>> if err:
        ...     print(f"API: {fn} faied with status_code: {status}")

        Args:
          method (str, optional): The method to use. Defaults to "get".
          trailing (str, optional): The trailing url to use. Defaults to "".
          json (Dict[str, Any], optional): The json to use. Defaults to {}.
          data ([type], optional): The data to use. Defaults to None.
          params (Dict, optional): The params to use. Defaults to {}.
          verbose (bool, optional): Whether to print the response or not. Defaults to False.
        """
        fn = getattr(self._session, method.lower())
        url = f"{self._url}{trailing}"
        if verbose:
            _msg = f"{method.upper()} {url}"
            if json:
                _msg += f"\n  json: {json}"
            if data:
                _msg += f"\n  data: {data}"
            if params:
                _msg += f"\n  params: {params}"
            logger.info(_msg)

        items = {}
        if json:
            items["json"] = json
        if data:
            items["data"] = data
        if params:
            items["params"] = params

        if SimplerTimes.get_now_i64() - self._last_update > 120:
            self._renew_session()
        r = fn(url, **items, **kwargs)
        if 399 < r.status_code < 500:
            raise SubwayClientError(r.content.decode(), code=r.status_code)
        if 499 < r.status_code < 600:
            raise SubwayServerError(r.content.decode(), code=r.status_code)

        try:
            data = r.json()
        except:
            if html:
                try:
                    data = r.content.decode()
                except:
                    raise ValueError("Cannot decode the content in html mode")
            else:
                raise ValueError(
                    "Cannot decode the content, pass html=True to decode html content"
                )

        # return items
        if not ret_code:
            return data
        else:
            return data, r.status_code


def get_subway(
    url="http://localhost:8080",
    headers: Optional[Dict[str, Any]] = None,
) -> Subway:
    sess = Session()
    if headers:
        sess.headers.update(headers)
    return Subway(url, sess)
