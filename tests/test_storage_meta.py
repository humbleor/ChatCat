import os

import pytest

from backend.agent.agent import storage

pytestmark = pytest.mark.skipif(
    not os.getenv("DATABASE_URL") and not os.path.exists("/tmp/chatcat_test_pg"),
    reason="需要本地 PostgreSQL + Redis",
)


def test_load_with_meta_roundtrip():
    uid, sid = "test_user", "test_session"
    # 无账号会被 storage 忽略，这里用不存在的账号验证返回形状
    messages, meta = storage.load_with_meta(uid, sid)
    assert isinstance(messages, list)
    assert isinstance(meta, dict)
