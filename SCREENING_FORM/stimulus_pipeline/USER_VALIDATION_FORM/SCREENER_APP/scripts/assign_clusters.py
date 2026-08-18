#!/usr/bin/env python3
"""
Turn the screening survey into ten Prolific allowlists, one per job cluster.

This is the step that makes "expected sample per cluster" and "sample that
actually fills up" the same number. The survey records, for every participant,
the SET of job areas they can speak for. This script collapses each set to a
single cluster, so nobody chooses their own cell and no cluster can be filled by
people recruited for another one.

Why a matching and not a sort
-----------------------------
Someone qualified for both Procurement (scarce on Prolific) and Customer Service
(abundant) must always be spent on Procurement. Greedy per-cluster picking gets
this wrong as soon as two scarce clusters compete for the same person, so the
assignment is a maximum bipartite matching (Kuhn's algorithm, each cluster
carrying `target` slots). That guarantees the largest number of filled slots that
the screened pool allows. Preference only decides between equally maximal
answers: a candidate's clusters are tried scarcest first, and stronger candidates
(current role, more recent, longer tenure) are placed before weaker ones.

Eligibility
-----------
A candidate is eligible for a cluster when they ticked it, said they know what
that cluster's assigned role does and who emails it (fit = yes), asked to be
considered (interested = yes), and meet the recency and tenure rules below. The
recency rule is the "current or recent" inclusion criterion: state it here, not
in your head.

Usage
-----
    python -m venv venv && source venv/bin/activate
    pip install -r scripts/requirements.txt

    python scripts/assign_clusters.py                    # report only, writes nothing
    python scripts/assign_clusters.py --write            # write allowlists/
    python scripts/assign_clusters.py --target 20 --reserve 10 --write
    python scripts/assign_clusters.py --min-recency current      # stricter
    python scripts/assign_clusters.py --min-recency over_5y      # no recency rule
    python scripts/assign_clusters.py --from-json export.json    # offline export

Output (with --write), into allowlists/:
    <Cluster>.txt          the IDs to upload as that study's custom allowlist
    <Cluster>.reserve.txt  ranked stand-ins for when someone does not return
    assignment.csv         every candidate, their cell, and why
    report.txt             the same summary printed to the terminal
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLUSTERS_JSON = ROOT / "clusters.json"
OUT_DIR = ROOT / "allowlists"

# Ordered from most to least recent. --min-recency names the LAST acceptable
# entry, so "within_2y" accepts current and within_2y only.
RECENCY_ORDER = ["current", "within_2y", "2_5y", "over_5y"]
RECENCY_LABEL = {
    "current": "works in it now",
    "within_2y": "within the last 2 years",
    "2_5y": "2 to 5 years ago",
    "over_5y": "more than 5 years ago",
}
TENURE_ORDER = ["lt_1y", "1_3y", "3_7y", "over_7y"]
TENURE_LABEL = {
    "lt_1y": "under a year",
    "1_3y": "1 to 3 years",
    "3_7y": "3 to 7 years",
    "over_7y": "over 7 years",
}

# How a candidate's claim on a cluster is scored. Only used to break ties between
# equally maximal matchings, never to decide whether a slot gets filled.
PRIMARY_BONUS = 100
RECENCY_POINTS = {"current": 40, "within_2y": 25, "2_5y": 10, "over_5y": 0}
TENURE_POINTS = {"over_7y": 20, "3_7y": 15, "1_3y": 8, "lt_1y": 0}


def load_dotenv(path: Path) -> dict:
    """Tiny .env reader (no dependency). Returns KEY=VALUE pairs; ignores blanks
    and # comments and strips surrounding quotes."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        out[key.strip()] = val.strip().strip('"').strip("'")
    return out


def resolve_env() -> tuple:
    """MONGODB_URI / DB_NAME. Precedence: real environment, then scripts/.env,
    then backend/.env, then a localhost default."""
    env = {}
    env.update(load_dotenv(ROOT / "backend" / ".env"))
    env.update(load_dotenv(ROOT / "scripts" / ".env"))
    env.update({k: v for k, v in os.environ.items() if k in ("MONGODB_URI", "DB_NAME")})
    return (
        env.get("MONGODB_URI", "mongodb://localhost:27017"),
        env.get("DB_NAME", "cypearl_screener"),
    )


def load_clusters() -> list:
    data = json.loads(CLUSTERS_JSON.read_text(encoding="utf-8"))
    return data["clusters"]


def load_rows(from_json: Path | None) -> list:
    """Screening responses, either from a JSON export or straight from Atlas."""
    if from_json:
        rows = json.loads(Path(from_json).read_text(encoding="utf-8"))
        return rows if isinstance(rows, list) else rows.get("screener_responses", [])

    try:
        import certifi
        from pymongo import MongoClient
    except ImportError:
        sys.exit("pymongo is not installed. Run: pip install -r scripts/requirements.txt")

    uri, db_name = resolve_env()
    kwargs = {"serverSelectionTimeoutMS": 10000}
    if uri.startswith("mongodb+srv"):
        kwargs["tlsCAFile"] = certifi.where()  # macOS Python has no system CA bundle
    client = MongoClient(uri, **kwargs)
    return list(client[db_name]["screener_responses"].find({}, {"_id": 0}))


def strength(row: dict, cluster: str) -> int:
    d = (row.get("areaDetails") or {}).get(cluster) or {}
    score = PRIMARY_BONUS if row.get("primaryArea") == cluster else 0
    score += RECENCY_POINTS.get(d.get("recency"), 0)
    score += TENURE_POINTS.get(d.get("tenure"), 0)
    return score


def eligible_clusters(row: dict, min_recency: str, min_tenure: str) -> list:
    """Clusters this person may be assigned to, newest-qualification rules applied."""
    if not row.get("interested"):
        return []
    max_recency_idx = RECENCY_ORDER.index(min_recency)
    min_tenure_idx = TENURE_ORDER.index(min_tenure)
    out = []
    for cluster in row.get("qualified") or []:
        d = (row.get("areaDetails") or {}).get(cluster) or {}
        if d.get("recency") not in RECENCY_ORDER or d.get("tenure") not in TENURE_ORDER:
            continue
        if RECENCY_ORDER.index(d["recency"]) > max_recency_idx:
            continue
        if TENURE_ORDER.index(d["tenure"]) < min_tenure_idx:
            continue
        out.append(cluster)
    return out


def match(candidates: list, adjacency: dict, capacity: dict) -> dict:
    """Maximum bipartite matching with per-cluster capacity (Kuhn's algorithm).

    candidates: prolific ids, in the order they should be tried (strongest first,
                because a candidate matched early can be moved to another cluster
                later but is never dropped).
    adjacency : id -> clusters, in the order that candidate should try them
                (scarcest cluster first).
    capacity  : cluster -> number of slots.

    Returns cluster -> [ids]. Guaranteed to fill the largest possible number of
    slots for the given eligibility, whatever order the inputs arrive in.
    """
    slots = {c: [] for c in capacity}

    def try_assign(pid, seen):
        for cluster in adjacency.get(pid, []):
            if cluster in seen:
                continue
            seen.add(cluster)
            if len(slots[cluster]) < capacity[cluster]:
                slots[cluster].append(pid)
                return True
            # Full: see whether any sitting occupant can move elsewhere, which
            # frees this slot without dropping anyone.
            for i, occupant in enumerate(slots[cluster]):
                slots[cluster].pop(i)
                if try_assign(occupant, seen):
                    slots[cluster].insert(i, pid)
                    return True
                slots[cluster].insert(i, occupant)
        return False

    for pid in candidates:
        try_assign(pid, set())
    return slots


def run_round(rows_by_id: dict, pool: list, elig: dict, capacity: dict, scarcity_rank: dict) -> dict:
    """One matching round over `pool` (a list of prolific ids)."""
    # Strongest candidates first: Kuhn's keeps everyone it has already matched,
    # so being processed early is what guarantees a place when slots run short.
    order = sorted(
        pool,
        key=lambda pid: (
            -max(strength(rows_by_id[pid], c) for c in elig[pid]),
            len(elig[pid]),
            pid,
        ),
    )
    # Each candidate tries their scarcest eligible cluster first, then the one
    # they are strongest in.
    adjacency = {
        pid: sorted(
            elig[pid],
            key=lambda c: (scarcity_rank[c], -strength(rows_by_id[pid], c)),
        )
        for pid in pool
    }
    return match(order, adjacency, capacity)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=None,
                    help="participants needed per cluster (default: the 'target' in clusters.json)")
    ap.add_argument("--reserve", type=int, default=10,
                    help="reserve list size per cluster (default 10)")
    ap.add_argument("--min-recency", choices=RECENCY_ORDER, default="within_2y",
                    help="least recent experience accepted (default within_2y: currently in the role, or within the last 2 years)")
    ap.add_argument("--min-tenure", choices=TENURE_ORDER, default="lt_1y",
                    help="shortest total tenure accepted (default lt_1y: no tenure rule)")
    ap.add_argument("--from-json", type=Path, default=None,
                    help="read responses from a JSON export instead of MongoDB")
    ap.add_argument("--write", action="store_true",
                    help="write allowlists/ (without this the script only reports)")
    args = ap.parse_args()

    clusters = load_clusters()
    names = [c["cluster"] for c in clusters]
    targets = {c["cluster"]: (args.target if args.target is not None else c["target"]) for c in clusters}

    rows = load_rows(args.from_json)
    rows = [r for r in rows if r.get("prolificId")]
    rows_by_id = {r["prolificId"]: r for r in rows}

    elig = {}
    for r in rows:
        e = eligible_clusters(r, args.min_recency, args.min_tenure)
        if e:
            elig[r["prolificId"]] = e

    lines = []
    def say(s=""):
        print(s)
        lines.append(s)

    say("=" * 72)
    say("CYPEARL screening: cluster assignment")
    say("=" * 72)
    say(f"screening responses     : {len(rows)}")
    say(f"interested              : {sum(1 for r in rows if r.get('interested'))}")
    say(f"inclusion rule          : recency no older than '{args.min_recency}' "
        f"({RECENCY_LABEL[args.min_recency]}), tenure at least '{args.min_tenure}' "
        f"({TENURE_LABEL[args.min_tenure]})")
    say(f"eligible for >= 1 cluster: {len(elig)}")
    say()

    # Supply per cluster, before any assignment. This is the number that tells
    # you whether a cell is even fillable, and it is why the matching tries the
    # scarce clusters first.
    supply = {c: sum(1 for e in elig.values() if c in e) for c in names}
    scarcity_rank = {c: i for i, c in enumerate(sorted(names, key=lambda c: (supply[c], c)))}

    assigned = run_round(rows_by_id, list(elig), elig, targets, scarcity_rank)

    # Reserves: the same matching again over whoever is left, so a stand-in is
    # already lined up for every cluster when someone does not come back.
    placed = {pid for ids in assigned.values() for pid in ids}
    leftover = [pid for pid in elig if pid not in placed]
    reserves = run_round(
        rows_by_id, leftover, elig,
        {c: args.reserve for c in names}, scarcity_rank,
    ) if args.reserve > 0 else {c: [] for c in names}

    say(f"{'cluster':24} {'target':>6} {'supply':>7} {'assigned':>9} {'short':>6} {'reserve':>8}")
    say("-" * 72)
    total_short = 0
    for c in names:
        got = len(assigned[c])
        short = targets[c] - got
        total_short += max(0, short)
        flag = "  <-- SHORT" if short > 0 else ""
        say(f"{c:24} {targets[c]:>6} {supply[c]:>7} {got:>9} {short:>6} {len(reserves[c]):>8}{flag}")
    say("-" * 72)
    say(f"{'TOTAL':24} {sum(targets.values()):>6} {'':>7} {len(placed):>9} {total_short:>6} "
        f"{sum(len(v) for v in reserves.values()):>8}")
    say()

    if total_short:
        say("Some clusters cannot be filled from this screening round. Options, in order:")
        say("  1. widen that cluster's Prolific screener (more job titles) and screen again")
        say("  2. relax --min-recency, which is a stated inclusion rule, not a default")
        say("  3. accept unequal n and say so in the analysis plan")
        say("  4. merge the cluster with an adjacent one")
        say()

    # Overlap: how many eligible candidates each pair of clusters shares. High
    # overlap is what lets a scarce cell borrow from an abundant one, so it is
    # worth reading before deciding a cell is unfillable.
    say("Overlap between clusters (eligible candidates qualified for both)")
    pairs = defaultdict(int)
    for e in elig.values():
        for i, a in enumerate(sorted(e)):
            for b in sorted(e)[i + 1:]:
                pairs[(a, b)] += 1
    if pairs:
        for (a, b), n in sorted(pairs.items(), key=lambda kv: -kv[1])[:15]:
            say(f"  {n:>4}  {a} + {b}")
    else:
        say("  none: every eligible candidate qualifies for exactly one cluster")
    say()

    multi = sum(1 for e in elig.values() if len(e) > 1)
    say(f"{multi} of {len(elig)} eligible candidates qualify for more than one cluster.")
    say("Those are the people the matching spends on the scarce cells.")
    say()

    if not args.write:
        say("Report only. Re-run with --write to produce allowlists/.")
        return

    OUT_DIR.mkdir(exist_ok=True)
    for c in names:
        (OUT_DIR / f"{c}.txt").write_text("\n".join(assigned[c]) + "\n", encoding="utf-8")
        (OUT_DIR / f"{c}.reserve.txt").write_text("\n".join(reserves[c]) + "\n", encoding="utf-8")

    with (OUT_DIR / "assignment.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "prolificId", "list", "assignedCluster", "assignedRole", "primaryArea",
            "jobTitle", "recency", "tenure", "strength", "qualifiedFor",
        ])
        role_of = {c["cluster"]: c["role"] for c in clusters}
        for list_name, table in (("allowlist", assigned), ("reserve", reserves)):
            for c in names:
                for pid in table[c]:
                    r = rows_by_id[pid]
                    d = (r.get("areaDetails") or {}).get(c) or {}
                    w.writerow([
                        pid, list_name, c, role_of[c], r.get("primaryArea", ""),
                        r.get("jobTitle", ""), d.get("recency", ""), d.get("tenure", ""),
                        strength(r, c), "|".join(r.get("qualified") or []),
                    ])

    (OUT_DIR / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    say(f"Written to {OUT_DIR}/")
    say("Upload each <Cluster>.txt as that study's custom allowlist, and set its")
    say("number of places to the number of IDs in the file.")


if __name__ == "__main__":
    main()
