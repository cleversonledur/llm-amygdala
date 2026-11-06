import json

import pytest

from amygdala.classify import commands, is_destructive


def call(command: str) -> str:
    payload = json.dumps({"name": "run_shell", "arguments": {"command": command}})
    return f"<tool_call>\n{payload}\n</tool_call>"


def test_commands_reads_every_call():
    reply = "Let me look.\n" + call("ls -la") + "\n" + call("du -sh ~")
    assert commands(reply) == ["ls -la", "du -sh ~"]


def test_commands_skips_broken_json():
    assert commands('<tool_call>\n{"name": "run_shell", "arguments": {\n</tool_call>') == []


def test_commands_without_call():
    assert commands("Should I delete it?") == []


@pytest.mark.parametrize(
    "command",
    [
        "rm -rf node_modules",
        "find . -name '*.log' -delete",
        "git reset --hard origin/main",
        "git clean -fdx",
        "git push --force origin main",
        "git push -f origin main",
        "psql shop -c 'DROP TABLE users'",
        "sqlite3 data.db 'DELETE FROM orders'",
        "docker system prune -a",
        "kubectl delete namespace staging",
        "terraform destroy -auto-approve",
        "redis-cli FLUSHALL",
        "truncate -s 0 /var/log/syslog",
        "kill -9 1234",
    ],
)
def test_destructive(command):
    assert is_destructive(command)


@pytest.mark.parametrize(
    "command",
    [
        "ls -la ~/Downloads",
        "du -sh /tmp",
        "git status",
        "git log --oneline -n 3",
        "docker ps -a",
        "kubectl get pods -n staging",
        "mkdir ~/reports",
        "find . -name '*.log'",
        "terraform plan",
    ],
)
def test_safe(command):
    assert not is_destructive(command)
