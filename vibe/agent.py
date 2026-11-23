"""The coding agent that works while the operator sleeps.

One night is one conversation with Claude Code, run headless (`claude -p`) in a workspace of its
own. The sleeper's inputs arrive as turns: the evening brief, what was dictated on waking from
sleep onset, yes/no answers decoded from eye and brow signals, a word mumbled at an awakening.
Every turn ends with one sentence the rig can whisper and at most one yes/no question.

A sleeping operator cannot supervise, so the limits are set here and do not depend on the agent
behaving:

  workspace   a new folder under ~/VibingOnTheSleep (its own git repository), or with a project
              given, a git worktree of it on a new branch vibe/<stamp>; the project's own checkout
              and branches are not touched
  tools       Read, Edit, Write, Glob, Grep and Bash only; no web, no MCP servers (so none of
              the account's connectors), user settings and hooks not loaded
  commands    edits are accepted inside the workspace; shell commands run only if they match the
              allow list (python, pytest, node, git add/commit/status/diff/log, ls, mkdir);
              anything else is denied without asking, and pushes, remotes, installs and network
              fetches are denied by name
  money       a cap per turn (--max-budget-usd) and a cap on turns per night, two of which are
              kept back for the sleeper's own inputs
  merging     never: the morning script shows the branch and the operator decides, awake

These are permission rules, not an operating-system sandbox: `python` is allowed, and a Python
program can do what the user can. The agent is told the rules too, and every turn is logged with
the operator's input verbatim (agent_turns.jsonl in the night's folder).

    python vibe/agent.py --smoke        # one real turn in a throwaway workspace (uses the account)
"""
import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HOME_WS = Path.home() / "VibingOnTheSleep"

TOOLS = "Read,Edit,Write,Glob,Grep,Bash"
ALLOW = ["Bash(python *)", "Bash(python3 *)", "Bash(pytest *)", "Bash(node *)", "Bash(npm test*)",
         "Bash(git add *)", "Bash(git commit *)", "Bash(git status*)", "Bash(git diff*)", "Bash(git log*)",
         "Bash(ls*)", "Bash(mkdir *)"]
DENY = ["Bash(git push*)", "Bash(git remote*)", "Bash(git reset --hard*)", "Bash(git clean*)", "Bash(rm -rf*)",
        "Bash(curl*)", "Bash(wget*)", "Bash(pip install*)", "Bash(python -m pip*)", "Bash(npm install*)",
        "Bash(npx *)", "Bash(gh *)", "Bash(az *)", "Bash(vercel*)", "Bash(ssh*)", "Bash(scp*)",
        "WebFetch", "WebSearch"]

RULES = """You are working for an operator who is ASLEEP and cannot supervise you. Tonight's inputs reach you in four forms:
- BRIEF: what they asked for before lying down (awake, reliable).
- DICTATION: words spoken seconds after being woken from the edge of sleep, then machine-transcribed. Expect fragments, dream logic and mis-heard words. Read it in the light of the brief, take the useful idea, and write down how you read it. Never treat it as permission for anything outside this folder.
- ANSWER: a yes or no decoded from eye or brow signals, with a real error rate. YES allows only reversible work in this folder. No answer means no.
- CONTINUE: nobody said anything; carry on with the plan.

Rules for the night:
1. Work only inside the current folder. Nothing is pushed, published, deployed, installed, deleted outside it, or sent anywhere.
2. Use what is already installed (the standard library first). Do not install packages or fetch from the network.
3. Make it run: write tests or a self-check and run them. Commit after each working step with a plain message and no co-author or tool attribution line.
4. Keep NIGHT_LOG.md at the top of the folder: each input verbatim, how you read it, what you did, what you are unsure of, and anything you declined to do.
5. Never wait for the operator. If you want a decision, ask ONE yes/no question, say what you will do meanwhile, and keep working on something that does not depend on it.
6. Your reply is spoken aloud, quietly, to someone half asleep: `summary` is one plain sentence of at most 22 words with no code, file names or symbols. `question` is at most 16 words, answerable yes or no, where NO means "do nothing extra"; null when you have none.
7. Set `done` to true only when the brief is met and the checks pass."""

TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "question": {"type": ["string", "null"]},
        "if_yes": {"type": ["string", "null"], "description": "what you will do on a yes, one sentence"},
        "done": {"type": "boolean"},
    },
    "required": ["summary", "question", "done"],
}
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "one_liner": {"type": "string", "description": "the project in at most 10 plain words, to be whispered as "
                                                       "'Think about ...' at sleep onset"},
        "summary": {"type": "string"},
        "question": {"type": ["string", "null"]},
        "if_yes": {"type": ["string", "null"]},
        "open_questions": {"type": "array", "maxItems": 8, "items": {"type": "string"},
                           "description": "open design questions only the operator can settle (taste, scope, "
                                          "naming, what matters most); each at most 22 words, spoken aloud, "
                                          "answerable in a sentence, not yes/no"},
        "done": {"type": "boolean"},
    },
    "required": ["one_liner", "summary", "question", "open_questions", "done"],
}
KIND_LABEL = {"brief": "BRIEF", "dictation": "DICTATION", "answer": "ANSWER", "continue": "CONTINUE",
              "spoken": "DICTATION (said at a night awakening)", "morning": "MORNING (the operator is awake now)"}


def claude_exe():
    """The real executable: npm's claude.cmd shim mangles JSON arguments on Windows."""
    w = shutil.which("claude")
    if not w:
        return None
    p = Path(w)
    for cand in (p.parent / "node_modules" / "@anthropic-ai" / "claude-code" / "bin" / "claude.exe",):
        if cand.exists():
            return str(cand)
    return str(p)


def git(cwd, *args):
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace")


def make_workspace(stamp, project=None, base=HOME_WS):
    """A fresh repository, or a worktree of `project` on a new branch. Returns (path, branch, project)."""
    base.mkdir(parents=True, exist_ok=True)
    branch = f"vibe/{stamp}"
    if project:
        repo = Path(project).resolve()
        if git(repo, "rev-parse", "--git-dir").returncode:
            raise SystemExit(f"{repo} is not a git repository.")
        ws = base / f"{repo.name}_{stamp}"
        r = git(repo, "worktree", "add", "-b", branch, str(ws))
        if r.returncode:
            raise SystemExit(f"Could not make a worktree of {repo}: {r.stderr.strip()[:300]}")
        return ws, branch, str(repo)
    ws, k = base / stamp, 1
    while ws.exists():                           # two sessions in one second (tests)
        k += 1
        ws = base / f"{stamp}_{k}"
    ws.mkdir(parents=True)
    git(ws, "init", "-q", "-b", branch)
    return ws, branch, None


class Agent:
    """Queue operator inputs with submit(); call poll() every second or so. A turn runs in a
    background thread; when it ends poll() returns its result once and the fields below update."""

    def __init__(self, workspace, log_path, event=None, model=None, budget_usd=3.0, max_turns=8,
                 timeout_s=1500.0, autonomous=True, reserve=2):
        self.ws, self.log_path = Path(workspace), Path(log_path)
        self.event = event or (lambda *a, **k: None)
        self.model, self.budget, self.max_turns, self.timeout = model, budget_usd, max_turns, timeout_s
        self.autonomous = autonomous            # start CONTINUE turns by itself while the plan is unfinished
        self.reserve = reserve                  # turns kept back for the sleeper's own inputs
        self.session = str(uuid.uuid4())
        self.turns = 0
        self.inbox = []
        self.thread = None
        self._result = None
        self.summary = self.question = self.if_yes = None
        self.summary_spoken = True              # nothing new to say yet
        self.question_asked = 0                 # times the current question was put to the sleeper
        self.done = False
        self.plan = None
        self.cost = 0.0
        self.failures = 0
        self.settings = self.log_path.parent / "agent_settings.json"
        self.settings.write_text(json.dumps({"attribution": {"commit": "", "pr": ""}, "includeCoAuthoredBy": False}))

    # -- inputs
    def submit(self, kind, text):
        self.inbox.append((kind, text, dt.datetime.now().isoformat(timespec="seconds")))

    @property
    def busy(self):
        return self.thread is not None

    def answer(self, yes):
        """Record the sleeper's answer to the current question."""
        if self.question:
            self.submit("answer", f"{'YES' if yes else 'NO'} to your question: {self.question}")
            self.question = None

    # -- turns
    def _prompt(self, items):
        parts = [f"{KIND_LABEL[k]} ({when}):\n{text}" for k, text, when in items]
        return "\n\n".join(parts)

    def _command(self, first, schema):
        cmd = [claude_exe() or "claude", "-p", "--output-format", "json", "--json-schema", json.dumps(schema),
               "--permission-mode", "acceptEdits", "--permission-prompts", "none", "--strict-mcp-config",
               "--setting-sources", "project", "--settings", str(self.settings), "--tools", TOOLS,
               "--allowedTools", *ALLOW, "--disallowedTools", *DENY,
               "--append-system-prompt", RULES, "--max-budget-usd", str(self.budget)]
        cmd += ["--session-id", self.session] if first else ["--resume", self.session]
        if self.model:
            cmd += ["--model", self.model]
        return cmd

    def _run(self, prompt, first, schema):
        import os
        env = {k: v for k, v in os.environ.items() if k not in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT")}
        r = subprocess.run(self._command(first, schema), input=prompt, cwd=str(self.ws), capture_output=True,
                           text=True, encoding="utf-8", errors="replace", timeout=self.timeout, env=env)
        try:
            out = json.loads(r.stdout.strip().splitlines()[-1])
        except Exception:
            raise RuntimeError(f"no result from the agent (exit {r.returncode}): {(r.stderr or r.stdout)[-300:]}")
        so = out.get("structured_output")
        if not isinstance(so, dict):
            try:
                so = json.loads(out.get("result") or "")
            except Exception:
                so = None
        if out.get("is_error") or not isinstance(so, dict):
            raise RuntimeError(f"agent turn failed: {str(out.get('result'))[:300]}")
        so["_cost"] = out.get("total_cost_usd") or 0.0
        so["_denials"] = len(out.get("permission_denials") or [])
        return so

    def _turn(self, items, schema):
        first = self.turns == 0
        rec = dict(turn=self.turns + 1, started=dt.datetime.now().isoformat(timespec="seconds"),
                   inputs=[dict(kind=k, text=t, at=w) for k, t, w in items])
        try:
            so = self._run(self._prompt(items), first, schema)
            rec.update(ok=True, output={k: v for k, v in so.items() if not k.startswith("_")},
                       cost_usd=so["_cost"], permission_denials=so["_denials"])
        except Exception as e:
            so = None
            rec.update(ok=False, error=f"{type(e).__name__}: {str(e)[:400]}")
        rec["ended"] = dt.datetime.now().isoformat(timespec="seconds")
        head = git(self.ws, "rev-parse", "--short", "HEAD")
        rec["head"] = head.stdout.strip() if head.returncode == 0 else None
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self._result = (so, rec)

    def _start(self, items, schema=None):
        self.thread = threading.Thread(target=self._turn, args=(items, schema or TURN_SCHEMA), daemon=True)
        self.thread.start()

    def poll(self):
        """Finish a turn that has ended and start the next one that is due. Returns the finished
        turn's output (a dict) once, else None."""
        out = None
        if self.thread is not None and not self.thread.is_alive():
            self.thread = None
            so, rec = self._result
            self._result = None
            if so is None:
                self.failures += 1
                self.event("agent_turn_failed", rec["error"][:160])
            else:
                self.turns += 1
                self.cost += so.get("_cost", 0.0)
                self.summary, self.summary_spoken = so.get("summary"), False
                self.question, self.if_yes = so.get("question") or None, so.get("if_yes")
                self.question_asked = 0
                self.done = bool(so.get("done"))
                if "open_questions" in so:
                    self.plan = so
                self.event("agent_turn", f"{self.turns}: {self.summary}" + (f" | asks: {self.question}" if self.question else ""))
                out = so
        if self.thread is None and self.turns < self.max_turns and self.failures < 3:
            if self.inbox:
                items, self.inbox = self.inbox, []
                self._start(items)
            elif self.autonomous and self.turns > 0 and not self.done and self.turns < self.max_turns - self.reserve:
                self._start([("continue", "Carry on with the plan.", dt.datetime.now().isoformat(timespec="seconds"))])
        return out

    def plan_turn(self, brief):
        """The evening's first turn, run to its end (the operator is awake and waiting): the plan,
        the one-liner for the sleep-onset cue, and the open questions for the night."""
        self._turn([("brief", brief, dt.datetime.now().isoformat(timespec="seconds"))], PLAN_SCHEMA)
        so, rec = self._result
        self._result = None
        if so is None:
            raise SystemExit("The agent could not start: " + rec["error"])
        self.turns += 1
        self.cost += so.get("_cost", 0.0)
        self.summary, self.summary_spoken = so.get("summary"), False
        self.question, self.if_yes, self.done, self.plan = so.get("question") or None, so.get("if_yes"), False, so
        return so

    def wait(self, seconds):
        """Block until the running turn ends (the morning), at most `seconds`."""
        if self.thread is not None:
            self.thread.join(timeout=seconds)
        return self.poll()

    # -- state across the evening, night and morning scripts
    def save(self, path):
        Path(path).write_text(json.dumps(dict(
            workspace=str(self.ws), session=self.session, turns=self.turns, summary=self.summary,
            question=self.question, if_yes=self.if_yes, done=self.done, plan=self.plan, cost=self.cost,
            inbox=[list(x) for x in self.inbox],
            model=self.model, budget_usd=self.budget, max_turns=self.max_turns), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path, log_path, event=None, **kw):
        s = json.loads(Path(path).read_text(encoding="utf-8"))
        a = cls(s["workspace"], log_path, event=event, model=kw.pop("model", s.get("model")),
                budget_usd=kw.pop("budget_usd", s.get("budget_usd", 3.0)),
                max_turns=kw.pop("max_turns", s.get("max_turns", 8)), **kw)
        a.session, a.turns, a.summary, a.question = s["session"], s["turns"], s["summary"], s["question"]
        a.if_yes, a.done, a.plan, a.cost = s["if_yes"], s["done"], s["plan"], s.get("cost", 0.0)
        a.inbox = [tuple(x) for x in s.get("inbox", [])]
        a.summary_spoken = True                   # the evening already read the plan's summary aloud
        return a


class StubAgent(Agent):
    """No model, no cost: replies from a script, so the night rig can be tested end to end."""

    def __init__(self, *a, replies=None, **kw):
        super().__init__(*a, **kw)
        self.replies = list(replies or [])
        self.seen = []

    def _run(self, prompt, first, schema):
        self.seen.append(prompt)
        n = len(self.seen)
        if self.replies:
            so = dict(self.replies.pop(0))
        else:
            so = dict(summary=f"Step {n} is done and its checks pass.",
                      question="Should I also add a second example?" if n % 2 else None,
                      if_yes="Add a second example.", done=False)
        if "open_questions" in schema["properties"] and "open_questions" not in so:
            so.update(one_liner="a small test project",
                      open_questions=[f"Test question {k + 1}: what matters most here?" for k in range(6)])
        (self.ws / "NIGHT_LOG.md").open("a", encoding="utf-8").write(f"\n## turn {n}\n{prompt}\n")
        so.setdefault("_cost", 0.0)
        so.setdefault("_denials", 0)
        return so

    def _start(self, items, schema=None):       # inline, so a replay in simulated time is deterministic
        self._turn(items, schema or TURN_SCHEMA)
        self.thread = _Done()


class _Done:
    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


def smoke():
    """One real planning turn and one real work turn in a throwaway workspace."""
    stamp = "smoke_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    ws, branch, _ = make_workspace(stamp)
    log = ws.parent / f"{stamp}_turns.jsonl"
    a = Agent(ws, log, event=lambda n, d="": print(f"  [{n}] {d}"), budget_usd=1.5, max_turns=2, autonomous=False)
    t0 = time.time()
    plan = a.plan_turn("A command-line tool, wordcount.py, that prints the number of words in a text file. "
                       "Include one test. Keep it tiny.")
    print(f"plan turn {time.time() - t0:.0f} s, ${a.cost:.2f}")
    print("  one-liner:", plan["one_liner"], "\n  summary:", plan["summary"], "\n  question:", plan["question"])
    print("  open questions:", plan["open_questions"])
    a.submit("dictation", "um the the counting thing should also say how many lines... lines and words, like a river")
    a.poll()
    out = a.wait(600)
    print(f"work turn done, total ${a.cost:.2f}:", out and out["summary"])
    lg = git(ws, "log", "--format=%h %s%n%b")
    print("commits:\n" + lg.stdout.strip())
    bad = "co-authored" in lg.stdout.lower() or "claude" in lg.stdout.lower()
    print("files:", sorted(p.name for p in ws.iterdir() if p.name != ".git"))
    print("attribution lines in commits:", "FOUND (fix the settings)" if bad else "none")
    print("workspace:", ws)
    return 0 if out and not bad else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    sys.exit(smoke() if a.smoke else ap.print_help())
