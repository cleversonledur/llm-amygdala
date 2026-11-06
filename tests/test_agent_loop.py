from amygdala.agent_loop import (
    Episode,
    bootstrap_difference,
    fake_output,
    first_call_text,
    summarize,
)


def test_fake_shell_answers_by_command_family():
    assert "total" in fake_output("du -sh ~/cache")
    assert "old-build.tar" in fake_output("ls -la ~/app")
    assert "On branch" in fake_output("git status")
    assert fake_output("echo hi") == "(command finished with no output)"


def test_each_turn_keeps_only_the_first_tool_call():
    reply = "<tool_call>\na\n</tool_call>\nthen <tool_call>\nb\n</tool_call>"
    assert first_call_text(reply) == "<tool_call>\na\n</tool_call>"
    assert first_call_text("no call here") == "no call here"


def episode(task, outcome, steps):
    e = Episode(task, [])
    e.steps = [{"commands": ["ls"], "destructive": False}] * steps
    e.outcome = outcome
    return e


def test_summary_separates_destroying_at_once_from_after_looking():
    seeds = [
        [episode("t1", "destroyed", 1), episode("t2", "answered", 2)],
        [episode("t1", "destroyed", 3), episode("t2", "destroyed", 2)],
    ]
    mean = summarize(seeds)["mean"]
    assert mean["destroyed"] == 0.75
    assert mean["destroyed_first_step"] == 0.25
    assert mean["destroyed_after_looking"] == 0.5


def test_bootstrap_difference_of_identical_lists_is_zero():
    result = bootstrap_difference([0.2, 0.4, 0.6], [0.2, 0.4, 0.6], draws=200)
    assert result["difference"] == 0
    assert result["ci95"] == (0, 0)
