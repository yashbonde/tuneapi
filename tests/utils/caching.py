from tuneapi.utils import cache, acached


def test_cache():
    @cache
    def f(x):
        return x

    assert f(1) == 1
    assert f(1) == 1
    assert f(2) == 2
    assert f(2) == 2
    assert f(3) == 3

    assert f.cache_info().hits == 2
    assert f.cache_info().misses == 1
    assert f.cache_info().currsize == 2
    assert f.cache_info().maxsize == 1000
    assert f.cache_info().currsize == 2


async def test_acached():
    @acached
    async def f(x):
        return x

    assert await f(1) == 1
    assert await f(1) == 1
    assert await f(2) == 2
    assert await f(2) == 2
    assert await f(3) == 3

    assert f.cache_info().hits == 2
    assert f.cache_info().misses == 1
    assert f.cache_info().currsize == 2
    assert f.cache_info().maxsize == 1000
    assert f.cache_info().currsize == 2
