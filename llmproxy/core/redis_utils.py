import inspect
from typing import Awaitable, TypeVar, Union, cast

_T = TypeVar("_T")


async def await_redis_result(result: Union[Awaitable[_T], _T]) -> _T:
    if inspect.isawaitable(result):
        return await cast(Awaitable[_T], result)
    return cast(_T, result)
