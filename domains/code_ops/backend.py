from __future__ import annotations

import ast
import random
from typing import Any, Dict, List, Optional

import torch

from engine.action_format import render_canonical_actions
from engine.actions import Action, ActionType
from engine.executor import StateExecutor
from engine.prompting import build_search_prompt
from engine.state import ReasoningState
from engine.task import ReasoningTask
from engine.traces import render_human_trace
from memory.retrieval import retrieve_context
from proof.parser import parse_actions


FUNC_NAMES = ["helper", "render", "merge", "dispatch", "compute", "normalize", "collect", "format_item"]
VAR_NAMES = ["x", "y", "item", "value", "flag", "count"]
CALLEE_NAMES = ["helper", "clean", "format_item", "transform"]


def _rid(prefix: str) -> str:
    return f"{prefix}_{random.randint(10**7, 10**8 - 1)}"


def _function_code(name: str, params: list[str], body_lines: list[str]) -> str:
    body = "\n".join(f"    {line}" for line in body_lines)
    return f"def {name}({', '.join(params)}):\n{body}"


def gen_function_name() -> ReasoningTask:
    name = random.choice(FUNC_NAMES)
    params = random.sample(VAR_NAMES, 2)
    code = _function_code(name, params, ["return 1"])
    return ReasoningTask(
        task_id=_rid("code_name"),
        domain="function_name",
        prompt=f"Read the Python function and return the function name:\n{code}",
        answer=name,
        goal="Return the top-level function name",
        meta={"family": "function_name", "code": code},
    )


def gen_parameter_count() -> ReasoningTask:
    name = random.choice(FUNC_NAMES)
    param_count = random.randint(0, 3)
    params = random.sample(VAR_NAMES, param_count)
    code = _function_code(name, params, ["return 1"])
    return ReasoningTask(
        task_id=_rid("code_params"),
        domain="parameter_count",
        prompt=f"Count the parameters of this Python function:\n{code}",
        answer=str(param_count),
        goal="Return the number of parameters as an integer",
        meta={"family": "parameter_count", "code": code},
    )


def gen_has_loop() -> ReasoningTask:
    name = random.choice(FUNC_NAMES)
    params = [random.choice(VAR_NAMES)]
    use_loop = random.choice([True, False])
    body = ["for item in range(3):", "    print(item)", "return 1"] if use_loop else ["return 1"]
    code = _function_code(name, params, body)
    return ReasoningTask(
        task_id=_rid("code_loop"),
        domain="has_loop",
        prompt=f"Does this Python function contain a loop? Return yes or no.\n{code}",
        answer="yes" if use_loop else "no",
        goal="Return yes if the function contains for/while, otherwise no",
        meta={"family": "has_loop", "code": code},
    )


def gen_first_called_function() -> ReasoningTask:
    name = random.choice(FUNC_NAMES)
    callee = random.choice(CALLEE_NAMES)
    params = [random.choice(VAR_NAMES)]
    code = _function_code(name, params, [f"return {callee}({params[0]})"])
    return ReasoningTask(
        task_id=_rid("code_call"),
        domain="first_called_function",
        prompt=f"Return the first called function name in this Python function:\n{code}",
        answer=callee,
        goal="Return the first called function name",
        meta={"family": "first_called_function", "code": code},
    )


def gen_return_literal() -> ReasoningTask:
    name = random.choice(FUNC_NAMES)
    literal = random.choice([0, 7, 11, True, False, "ok"])
    code = _function_code(name, [], [f"return {repr(literal)}"])
    return ReasoningTask(
        task_id=_rid("code_return"),
        domain="return_literal",
        prompt=f"Return the literal returned by this Python function:\n{code}",
        answer=str(literal),
        goal="Return the literal value exactly",
        meta={"family": "return_literal", "code": code},
    )


def gen_has_conditional() -> ReasoningTask:
    name = random.choice(FUNC_NAMES)
    param = random.choice(VAR_NAMES)
    use_if = random.choice([True, False])
    body = ["if value > 0:", "    return True", "return False"] if use_if else ["return True"]
    code = _function_code(name, [param], body)
    return ReasoningTask(
        task_id=_rid("code_cond"),
        domain="has_conditional",
        prompt=f"Does this Python function contain a conditional? Return yes or no.\n{code}",
        answer="yes" if use_if else "no",
        goal="Return yes if the function contains an if statement, otherwise no",
        meta={"family": "has_conditional", "code": code},
    )


def gen_assignment_count() -> ReasoningTask:
    name = random.choice(FUNC_NAMES)
    assignments = random.randint(1, 3)
    body = [f"v{i} = {i}" for i in range(assignments)] + ["return 1"]
    code = _function_code(name, [], body)
    return ReasoningTask(
        task_id=_rid("code_assign"),
        domain="assignment_count",
        prompt=f"Count assignment statements in this Python function:\n{code}",
        answer=str(assignments),
        goal="Return the number of assignment statements",
        meta={"family": "assignment_count", "code": code},
    )


def gen_called_function_count() -> ReasoningTask:
    name = random.choice(FUNC_NAMES)
    first = random.choice(CALLEE_NAMES)
    second = random.choice([callee for callee in CALLEE_NAMES if callee != first])
    distinct = random.choice([1, 2])
    body = [f"{first}(x)"]
    if distinct == 2:
        body.append(f"{second}(x)")
    else:
        body.append(f"{first}(x)")
    body.append("return x")
    code = _function_code(name, ["x"], body)
    return ReasoningTask(
        task_id=_rid("code_calls"),
        domain="called_function_count",
        prompt=f"Count distinct called functions in this Python function:\n{code}",
        answer=str(distinct),
        goal="Return the number of distinct called functions",
        meta={"family": "called_function_count", "code": code},
    )


GENERATORS = {
    "function_name": gen_function_name,
    "parameter_count": gen_parameter_count,
    "has_loop": gen_has_loop,
    "first_called_function": gen_first_called_function,
    "return_literal": gen_return_literal,
    "has_conditional": gen_has_conditional,
    "assignment_count": gen_assignment_count,
    "called_function_count": gen_called_function_count,
}


def sample_task(domains: List[str]) -> ReasoningTask:
    domain = random.choice(domains)
    return GENERATORS[domain]()


def _extract_code(arg: str) -> str:
    return arg.split("\n", 1)[-1].strip() if "\n" in arg else arg.strip()


def _first_function(code: str) -> ast.FunctionDef:
    tree = ast.parse(code)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    raise ValueError("no function found")


def function_name(arg: str, state: Any = None) -> Dict[str, Any]:
    fn = _first_function(_extract_code(arg))
    return {"ok": True, "result": fn.name, "solved": True, "answer": fn.name, "goal_progress": 1.0}


def parameter_count(arg: str, state: Any = None) -> Dict[str, Any]:
    fn = _first_function(_extract_code(arg))
    rendered = str(len(fn.args.args))
    return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}


def has_loop(arg: str, state: Any = None) -> Dict[str, Any]:
    fn = _first_function(_extract_code(arg))
    loop_found = any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(fn))
    rendered = "yes" if loop_found else "no"
    return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}


def first_called_function(arg: str, state: Any = None) -> Dict[str, Any]:
    fn = _first_function(_extract_code(arg))
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                rendered = target.id
            elif isinstance(target, ast.Attribute):
                rendered = target.attr
            else:
                rendered = "unknown"
            return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}
    return {"ok": True, "result": "none", "solved": True, "answer": "none", "goal_progress": 1.0}


def return_literal(arg: str, state: Any = None) -> Dict[str, Any]:
    fn = _first_function(_extract_code(arg))
    for node in ast.walk(fn):
        if isinstance(node, ast.Return):
            value = node.value
            if isinstance(value, ast.Constant):
                rendered = str(value.value)
                return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}
    return {"ok": False, "result": "no literal return"}


def has_conditional(arg: str, state: Any = None) -> Dict[str, Any]:
    fn = _first_function(_extract_code(arg))
    found = any(isinstance(node, ast.If) for node in ast.walk(fn))
    rendered = "yes" if found else "no"
    return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}


def assignment_count(arg: str, state: Any = None) -> Dict[str, Any]:
    fn = _first_function(_extract_code(arg))
    count = sum(isinstance(node, ast.Assign) for node in ast.walk(fn))
    rendered = str(count)
    return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}


def called_function_count(arg: str, state: Any = None) -> Dict[str, Any]:
    fn = _first_function(_extract_code(arg))
    names: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Attribute):
                names.add(target.attr)
    rendered = str(len(names))
    return {"ok": True, "result": rendered, "solved": True, "answer": rendered, "goal_progress": 1.0}


class CodeToolRegistry:
    def __init__(self) -> None:
        self.tools = {
            "function_name": function_name,
            "parameter_count": parameter_count,
            "has_loop": has_loop,
            "first_called_function": first_called_function,
            "return_literal": return_literal,
            "has_conditional": has_conditional,
            "assignment_count": assignment_count,
            "called_function_count": called_function_count,
        }

    def call(self, name: str, arg: str, state: Any = None) -> Dict[str, Any]:
        fn = self.tools.get(name)
        if fn is None:
            return {"ok": False, "result": f"unknown tool: {name}"}
        try:
            return fn(arg, state)
        except Exception as exc:
            return {"ok": False, "result": f"code tool error: {exc}"}


class CodeOpsReasoningDomain:
    name = "code_ops"
    default_curriculum_config = "config/code_ops_curriculum.yaml"

    def sample_task(self, domains: List[str]) -> ReasoningTask:
        return sample_task(domains)

    def make_state(self, task: ReasoningTask) -> ReasoningState:
        return ReasoningState(
            task_id=task.task_id,
            domain=task.domain,
            problem_text=task.prompt,
            goal=task.goal,
            expected_answer=task.answer,
            metadata=task.meta,
        )

    def manual_task(self, domain: str, prompt: str, answer: str = "") -> ReasoningTask:
        return ReasoningTask(
            task_id="manual_0001",
            domain=domain,
            prompt=prompt,
            answer=answer,
            goal="Solve the problem",
            meta={"family": domain},
        )

    def build_training_example(self, task: ReasoningTask) -> str:
        state = self.make_state(task)
        return state.serialize() + "\n" + self.build_gold_trace(task)

    def build_gold_trace(self, task: ReasoningTask) -> str:
        actions = [
            Action(type=ActionType.THINK, content={
                "function_name": "Parse the function definition and read the declared name.",
                "parameter_count": "Parse the signature and count parameters.",
                "has_loop": "Inspect the body for for/while constructs.",
                "first_called_function": "Inspect call sites and return the first called function name.",
                "return_literal": "Inspect the return statement and extract the literal value.",
                "has_conditional": "Inspect the body for if statements.",
                "assignment_count": "Count assignment statements in the function body.",
                "called_function_count": "Count distinct called functions in the body.",
            }[task.domain]),
            Action(type=ActionType.APPLY, tool=task.domain, content=task.prompt),
            Action(type=ActionType.ANSWER, content=task.answer),
        ]
        return render_canonical_actions(actions)

    def build_verifier_examples(self, task: ReasoningTask) -> tuple[str, torch.Tensor, str, torch.Tensor]:
        pos = self.make_state(task)
        pos.final_answer = task.answer
        pos.status = "solved"
        pos.derived_facts.append(task.answer)
        pos.action_history.append({"type": "ANSWER", "content": task.answer})
        pos.tool_history.append({"tool": "oracle", "result": {"ok": True, "answer": task.answer}})

        neg = self.make_state(task)
        neg.derived_facts.append("search_not_started")

        pos_t = self.build_verifier_targets(task, pos)
        neg_t = self.build_verifier_targets(task, neg, local_scores={"valid_step": 0.35, "goal_progress": 0.0, "risk_score": 0.8})
        return pos.serialize(), pos_t, neg.serialize(), neg_t

    def build_verifier_targets(
        self,
        task: ReasoningTask,
        state: ReasoningState,
        local_scores: Optional[Dict[str, float]] = None,
    ) -> torch.Tensor:
        local_scores = local_scores or {}
        has_answer = bool(state.final_answer.strip())
        correct = has_answer and self.evaluate_answer(task, state.final_answer)
        solved = state.status == "solved"

        valid_step = float(local_scores.get("valid_step", 1.0 if solved or has_answer else 0.6))
        structural_progress = min(
            1.0,
            0.1 * len(state.derived_facts)
            + 0.08 * len(state.tool_history)
            + 0.08 * len(state.action_history)
            + 0.04 * len(state.subgoals),
        )
        goal_progress = max(float(local_scores.get("goal_progress", 0.0)), structural_progress)
        if correct:
            goal_progress = max(goal_progress, 0.95)
        elif has_answer:
            goal_progress = min(0.75, max(goal_progress, 0.25))

        proof_completion = 1.0 if correct and solved else (0.2 if has_answer else min(0.2, goal_progress * 0.5))
        risk = float(local_scores.get("risk_score", 0.05 if correct else (0.8 if has_answer else 0.45)))
        branch_priority = max(0.05, min(0.98, 0.56 * goal_progress + 0.24 * valid_step + 0.20 * proof_completion))
        value_estimate = max(0.01, min(0.99, 0.45 * goal_progress + 0.30 * proof_completion + 0.20 * branch_priority + 0.10 * valid_step - 0.15 * risk))
        return torch.tensor(
            [
                max(0.02, min(0.99, valid_step)),
                max(0.0, min(0.99, goal_progress)),
                max(0.0, min(0.99, proof_completion)),
                max(0.01, min(0.99, risk)),
                branch_priority,
                value_estimate,
            ],
            dtype=torch.float32,
        )

    def evaluate_answer(self, task: ReasoningTask, candidate: str) -> bool:
        family = task.meta.get("family", task.domain)
        if family in {"has_loop", "has_conditional"}:
            return candidate.strip().lower() == task.answer.strip().lower()
        return candidate.strip() == task.answer.strip()

    def parse_actions(self, text: str) -> tuple[List[Any], float]:
        return parse_actions(text)

    def fallback_repairs(self, state: ReasoningState) -> List[Action]:
        if state.domain not in GENERATORS:
            return []
        return [Action(type=ActionType.APPLY, tool=state.domain, content=state.problem_text)]

    def allowed_action_types(self, state: ReasoningState) -> List[str]:
        actions = ["THINK", "APPLY"]
        if state.derived_facts or state.tool_history:
            actions.extend(["CHECK", "ANSWER"])
        return actions

    def allowed_tools(self, state: ReasoningState, action_type: str) -> List[str]:
        if action_type.upper() not in {"APPLY", "CHECK"}:
            return []
        return [state.domain]

    def candidate_bindings(self, state: ReasoningState, action_type: str, tool: str = "") -> List[Dict[str, str]]:
        normalized = action_type.upper()
        if normalized == "THINK":
            return [{"content": "inspect the function structure and extract the requested property"}]
        if normalized == "ANSWER":
            return [{"content": fact} for fact in state.derived_facts[-3:]]
        if normalized in {"APPLY", "CHECK"}:
            return [{"content": state.problem_text}]
        return []

    def action_schema(self, state: ReasoningState) -> Dict[str, Any]:
        return {
            "strict": True,
            "action_types": {
                action_type: {
                    "tools": self.allowed_tools(state, action_type),
                    "bindings": self.candidate_bindings(state, action_type),
                }
                for action_type in self.allowed_action_types(state)
            },
        }

    def action_format_instructions(self) -> str:
        return (
            "Emit one JSON action per line in canonical form:\n"
            'ACTION {"type":"THINK","content":"plan"}\n'
            'ACTION {"type":"APPLY","tool":"tool_name","content":"arguments"}\n'
            'ACTION {"type":"ANSWER","content":"final answer"}'
        )

    def build_search_prompt(
        self,
        state: ReasoningState,
        *,
        lemma_store: Any | None = None,
        hard_case_store: Any | None = None,
        tactic_stats: Any | None = None,
        retrieval_mode: str = "hybrid",
        embedding_model: str = "hashing",
        event_logger: Any | None = None,
    ) -> str:
        retrieval_context = None
        if lemma_store is not None and hard_case_store is not None:
            retrieval_context = retrieve_context(
                lemma_store,
                hard_case_store,
                state.domain,
                state.problem_text,
                mode=retrieval_mode,
                embedding_model=embedding_model,
                event_logger=event_logger,
            )
        tactic_hints = None
        if tactic_stats is not None:
            ranked = tactic_stats.top_tactics(state.domain, limit=3)
            tactic_hints = [f"{name} bias={bias:.2f}" for name, bias in ranked if bias != 0.5]
        return build_search_prompt(
            state,
            self.action_format_instructions(),
            retrieval_context=retrieval_context,
            tactic_hints=tactic_hints,
        )

    def state_signature(self, state: ReasoningState) -> str:
        derived = " | ".join(state.derived_facts[-3:])
        return " || ".join([state.domain, state.problem_text, derived, state.final_answer.strip()])

    def render_human_trace(self, state: ReasoningState) -> str:
        return render_human_trace(state)

    def create_executor(self) -> StateExecutor:
        return StateExecutor(CodeToolRegistry(), answer_judge=self._answer_judge)

    def _answer_judge(self, state: ReasoningState, candidate: str) -> bool:
        task = ReasoningTask(
            task_id=state.task_id,
            domain=state.domain,
            prompt=state.problem_text,
            answer=state.expected_answer,
            goal=state.goal,
            meta=state.metadata,
        )
        return self.evaluate_answer(task, candidate)

    def maybe_derive_lemma(self, task: ReasoningTask) -> None:
        return None

    def benchmark_tasks(self) -> List[ReasoningTask]:
        return [
            self.manual_task(
                "function_name",
                "Read the Python function and return the function name:\ndef helper(x, y):\n    return x + y",
                "helper",
            ),
            self.manual_task(
                "parameter_count",
                "Count the parameters of this Python function:\ndef render(item, flag):\n    return item",
                "2",
            ),
            self.manual_task(
                "has_loop",
                "Does this Python function contain a loop? Return yes or no.\ndef collect(x):\n    for item in range(3):\n        print(item)\n    return x",
                "yes",
            ),
            self.manual_task(
                "return_literal",
                "Return the literal returned by this Python function:\ndef answer():\n    return 7",
                "7",
            ),
            self.manual_task(
                "has_conditional",
                "Does this Python function contain a conditional? Return yes or no.\ndef choose(x):\n    if x > 0:\n        return True\n    return False",
                "yes",
            ),
            self.manual_task(
                "assignment_count",
                "Count assignment statements in this Python function:\ndef build():\n    x = 1\n    y = 2\n    return x + y",
                "2",
            ),
            self.manual_task(
                "called_function_count",
                "Count distinct called functions in this Python function:\ndef dispatch(x):\n    helper(x)\n    format_item(x)\n    helper(x)\n    return x",
                "2",
            ),
        ]
