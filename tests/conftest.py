"""测试套件密闭性保障。

此前测试不联网依赖「CI 恰好清空了环境变量」——本地开发者 export 了真实
key 时，涉及网络路径的测试可能真发 HTTP（限流/慢/泄漏 key 到第三方）。
这里 autouse 清空外部凭据 + data_collectors 在 import 时捕获的模块级常量，
使 tests.yml 的 env 清空变成双保险而非唯一防线。
需要真实网络的测试应显式标记 skip 并说明原因。
"""

from __future__ import annotations

import pytest

_EXTERNAL_KEYS = (
    "QWEATHER_API_KEY",
    "FIRMS_MAP_KEY",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OLLAMA_BASE_URL",
    "CARM_ROOT",
)


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    for key in _EXTERNAL_KEYS:
        monkeypatch.setenv(key, "")
    # data_collectors 在 import 时把 key 读进了模块常量，清 env 不够
    from earthbench import data_collectors as dc

    monkeypatch.setattr(dc, "QWEATHER_API_KEY", "")
    monkeypatch.setattr(dc, "FIRMS_MAP_KEY", "")
    from earthbench import verification as v

    monkeypatch.setattr(v, "FIRMS_MAP_KEY", "")
    yield
