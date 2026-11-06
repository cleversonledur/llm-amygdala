"""Command line: fit the probe and caution vectors, sweep the strength, evaluate, report."""

import argparse
import json
from pathlib import Path

import torch

from . import probe as probe_lib
from .caution import caution_vectors
from .circuit_cli import (
    CIRCUIT_CONDITIONS,
    cmd_agent_eval,
    cmd_circuit_eval,
    cmd_circuit_plots,
    cmd_circuit_report,
    cmd_drafts,
    cmd_sensors,
    cmd_targets,
    cmd_train,
)
from .common import DEFAULT_MODEL, load_model, load_tasks, task_states
from .evaluate import Caution, Rethink, evaluate, score, summary
from .stats import auroc, mcnemar_exact, wilson_interval

RUNS = Path("runs")
EXPLICIT_IN_CLEAR = 24  # train_clear.txt lists 24 explicit deletes, then 24 ordinary tasks


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {path}")


def training_tasks():
    risky, clear = load_tasks("train_risky"), load_tasks("train_clear")
    labels = torch.tensor([1] * len(risky) + [0] * len(clear))
    return risky, clear, labels


def cmd_fit(args) -> None:
    model, tokenizer = load_model(args.model)
    risky, clear, labels = training_tasks()
    states = task_states(model, tokenizer, risky + clear, args.batch_size)

    # Row 0 is the token embedding, the same for every prompt, so the probe starts at row 1.
    rows = range(1, states.shape[1])
    cv = {row: probe_lib.cross_val(states[:, row], labels) for row in rows}
    # Many rows rank the training tasks perfectly, so held-out confidence breaks the tie.
    best = max(rows, key=lambda row: (cv[row]["auroc"], -cv[row]["log_loss"]))
    danger_probe = probe_lib.fit(states[:, best], labels, layer=best)

    caution = caution_vectors(model, tokenizer, risky + clear, args.batch_size)
    norms = states.norm(dim=-1).mean(dim=0)

    torch.save({"probe": danger_probe.to_dict(), "caution": caution}, RUNS / "fit.pt")
    write_json(
        RUNS / "fit.json",
        {
            "probe_row": best,
            "cross_validation": cv,
            "caution_norm": caution.norm(dim=-1).tolist(),
            "hidden_norm": norms.tolist(),
        },
    )
    print(f"probe row {best}, cross-validation {cv[best]}")


def load_fit():
    fit = torch.load(RUNS / "fit.pt")
    return probe_lib.Probe(**fit["probe"]), fit["caution"]


def train_scores(model, tokenizer, args, caution) -> dict:
    risky, clear, _ = training_tasks()
    explicit, benign = clear[:EXPLICIT_IN_CLEAR], clear[EXPLICIT_IN_CLEAR:]
    r, e, b = (
        score(model, tokenizer, tasks, args.batch_size, args.max_new_tokens, caution)
        for tasks in (risky, explicit, benign)
    )
    return {
        "risky_destructive": r["destructive_rate"],
        "risky_tool_calls": r["tool_call_rate"],
        "explicit_delete": e["destructive_rate"],
        "benign_tool_calls": b["tool_call_rate"],
    }


def cmd_sweep(args) -> None:
    """Try always-on caution at several rows and strengths, on the training tasks only."""
    model, tokenizer = load_model(args.model)
    _, vectors = load_fit()
    baseline = train_scores(model, tokenizer, args, None)
    print("base", baseline)
    rows = []
    for row in args.rows:
        for beta in args.betas:
            result = train_scores(model, tokenizer, args, Caution(row, beta * vectors[row]))
            rows.append({"row": row, "beta": beta, **result})
            print(rows[-1])

    usable = [r for r in rows if r["benign_tool_calls"] >= args.min_benign]
    best = min(usable, key=lambda r: (r["risky_destructive"], -r["risky_tool_calls"], r["beta"]))
    write_json(RUNS / "sweep.json", {"baseline": baseline, "rows": rows, "best": best})
    print("best", best)


CONDITIONS = ("base", "steer_always", "steer_amygdala", "rethink_always", "rethink_amygdala")


def cmd_eval(args) -> None:
    """Run the held-out eval sets under the chosen conditions, merged into runs/eval.json."""
    model, tokenizer = load_model(args.model)
    danger_probe, vectors = load_fit()
    best = json.loads((RUNS / "sweep.json").read_text())["best"]
    vector = best["beta"] * vectors[best["row"]]
    reactions = {
        "base": {},
        "steer_always": {"caution": Caution(best["row"], vector)},
        "steer_amygdala": {"caution": Caution(best["row"], vector, danger_probe)},
        "rethink_always": {"rethink": Rethink()},
        "rethink_amygdala": {"rethink": Rethink(danger_probe)},
    }
    path = RUNS / "eval.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    results |= {"row": best["row"], "beta": best["beta"]}
    for name in args.conditions:
        sets = evaluate(model, tokenizer, args.batch_size, args.max_new_tokens, **reactions[name])
        results[name] = {"summary": summary(sets), "sets": sets}
        print(name, results[name]["summary"])
    write_json(path, results)


def flags(results: dict, condition: str, set_name: str, key: str) -> list[bool]:
    samples = results[condition]["sets"][set_name]["samples"]
    return [bool(s["commands"]) if key == "tool_call" else s[key] for s in samples]


def cmd_report(_args) -> None:
    results = json.loads((RUNS / "eval.json").read_text())
    metrics = {
        "ambiguous_destructive": ("eval_ambiguous", "destructive"),
        "ambiguous_tool_calls": ("eval_ambiguous", "tool_call"),
        "explicit_delete": ("eval_explicit", "destructive"),
        "benign_tool_calls": ("eval_benign", "tool_call"),
    }
    conditions = [name for name in CONDITIONS if name in results]

    intervals = {}
    for condition in conditions:
        intervals[condition] = {}
        for metric, (set_name, key) in metrics.items():
            hits = flags(results, condition, set_name, key)
            intervals[condition][metric] = {
                "rate": sum(hits) / len(hits),
                "ci95": wilson_interval(sum(hits), len(hits)),
            }

    def test(metric: str, a: str, b: str) -> dict:
        set_name, key = metrics[metric]
        return mcnemar_exact(flags(results, a, set_name, key), flags(results, b, set_name, key))

    tests = {}
    for reaction in ("steer", "rethink"):
        gated, always = f"{reaction}_amygdala", f"{reaction}_always"
        if gated not in results:
            continue
        tests[f"ambiguous_destructive base vs {gated}"] = test(
            "ambiguous_destructive", "base", gated
        )
        tests[f"ambiguous_tool_calls base vs {gated}"] = test("ambiguous_tool_calls", "base", gated)
        tests[f"explicit_delete base vs {gated}"] = test("explicit_delete", "base", gated)
        tests[f"explicit_delete {gated} vs {always}"] = test("explicit_delete", gated, always)

    sets = results["steer_amygdala"]["sets"]
    gates = {name: [s["gate"] for s in sets[name]["samples"]] for name in sets}
    held_out_auroc = auroc(
        gates["eval_ambiguous"] + gates["eval_explicit"] + gates["eval_benign"],
        [1] * len(gates["eval_ambiguous"])
        + [0] * (len(gates["eval_explicit"]) + len(gates["eval_benign"])),
    )
    gate_means = {name: sum(g) / len(g) for name, g in gates.items()}

    write_json(
        RUNS / "report.json",
        {
            "intervals": intervals,
            "tests": tests,
            "probe_auroc_held_out": held_out_auroc,
            "gate_means": gate_means,
        },
    )
    for condition in conditions:
        cells = ", ".join(f"{m} {v['rate']:.3f}" for m, v in intervals[condition].items())
        print(f"{condition}: {cells}")
    for name, test in tests.items():
        print(f"{name}: {test}")
    print(f"probe AUROC on held-out eval sets: {held_out_auroc:.3f}, mean gates {gate_means}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="amygdala", description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fit", help="fit the danger probe and the caution vectors")
    sweep = sub.add_parser("sweep", help="pick the caution row and strength on training tasks")
    sweep.add_argument("--rows", type=int, nargs="+", default=[6, 9, 12, 15, 18, 21])
    sweep.add_argument("--betas", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    sweep.add_argument("--min-benign", type=float, default=0.9)
    evaluation = sub.add_parser("eval", help="evaluate the held-out sets under each condition")
    evaluation.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=CONDITIONS)
    sub.add_parser("report", help="intervals, McNemar tests, and held-out probe AUROC")

    sub.add_parser("drafts", help="base-model replies for every task")
    sub.add_parser("sensors", help="fit the amygdala and cingulate sensors")
    sub.add_parser("targets", help="teacher targets for the prefrontal block")
    train = sub.add_parser("train", help="train the prefrontal block inside the model")
    train.add_argument("--steps", type=int, default=300)
    train.add_argument("--lr", type=float, default=5e-4)
    train.add_argument("--accumulate", type=int, default=8)
    train.add_argument("--eval-every", type=int, default=25)
    train.add_argument("--rank", type=int, default=8)
    train.add_argument("--decay", type=float, default=0.9)
    circuit_eval = sub.add_parser("circuit-eval", help="evaluate the circuit, lesions, controls")
    circuit_eval.add_argument(
        "--conditions", nargs="+", choices=CIRCUIT_CONDITIONS, default=CIRCUIT_CONDITIONS
    )
    sub.add_parser("circuit-report", help="tables, tests, sensor AUROC, identity check")
    sub.add_parser("circuit-plots", help="draw the README charts")
    agent = sub.add_parser("agent-eval", help="multi-step agent runs against a fake shell")
    agent.add_argument("--seeds", type=int, default=5)
    agent.add_argument("--max-steps", type=int, default=5)
    agent.add_argument("--temperature", type=float, default=0.7)

    args = parser.parse_args()
    RUNS.mkdir(exist_ok=True)
    commands = {
        "fit": cmd_fit,
        "sweep": cmd_sweep,
        "eval": cmd_eval,
        "report": cmd_report,
        "drafts": cmd_drafts,
        "sensors": cmd_sensors,
        "targets": cmd_targets,
        "train": cmd_train,
        "circuit-eval": cmd_circuit_eval,
        "circuit-report": cmd_circuit_report,
        "circuit-plots": cmd_circuit_plots,
        "agent-eval": cmd_agent_eval,
    }
    commands[args.command](args)
