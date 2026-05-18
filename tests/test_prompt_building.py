"""Tests for prompt building with framing variants."""

import pytest

from refusal_eval.prompts import (
    DEFAULT_EXPLOIT_GOALS,
    FRAMINGS,
    REFUSAL_INSTRUCTION,
    build_phase_prompts,
    build_task_prompt,
)


class TestBuildTaskPrompt:
    def test_bare_framing_no_prefix(self):
        prompt = build_task_prompt("pwn", "delulu:1337", "bare")
        assert prompt.startswith("Read the vulnerability")

    def test_ctf_framing_has_prefix(self):
        prompt = build_task_prompt("pwn", "delulu:1337", "ctf")
        assert prompt.startswith(FRAMINGS["ctf"])

    def test_authorized_framing_has_prefix(self):
        prompt = build_task_prompt("pwn", "delulu:1337", "authorized")
        assert prompt.startswith(FRAMINGS["authorized"])

    def test_host_port_substituted(self):
        prompt = build_task_prompt("pwn", "delulu:1337", "bare")
        assert "delulu:1337" in prompt

    def test_exploit_goal_substituted(self):
        prompt = build_task_prompt(
            "pwn", "delulu:1337", "bare", exploit_goal="steal the flag"
        )
        assert "steal the flag" in prompt

    def test_default_exploit_goal_used(self):
        prompt = build_task_prompt("dos", "target:8080", "bare")
        assert DEFAULT_EXPLOIT_GOALS["dos"] in prompt

    def test_unknown_category_uses_fallback_goal(self):
        prompt = build_task_prompt("unknown_category", "host:1337", "bare")
        assert "exploit the vulnerability" in prompt

    def test_refusal_instruction_not_in_prompt(self):
        """Refusal instruction should be in system message, not user prompt."""
        for framing in FRAMINGS:
            prompt = build_task_prompt("pwn", "delulu:1337", framing)
            assert REFUSAL_INSTRUCTION not in prompt


class TestBuildPhasePrompts:
    def test_has_three_phases(self):
        phases = build_phase_prompts("pwn", "delulu:1337", "bare")
        assert len(phases) == 3

    def test_phase_labels(self):
        phases = build_phase_prompts("pwn", "delulu:1337", "bare")
        labels = [p["label"] for p in phases]
        assert labels == ["analysis", "poc", "exploit"]

    def test_host_port_in_poc_and_exploit(self):
        phases = build_phase_prompts("pwn", "delulu:1337", "bare")
        assert "delulu:1337" in phases[1]["prompt"]
        assert "delulu:1337" in phases[2]["prompt"]

    def test_host_port_not_in_analysis(self):
        phases = build_phase_prompts("pwn", "delulu:1337", "bare")
        assert "delulu:1337" not in phases[0]["prompt"]

    def test_exploit_goal_in_exploit_phase(self):
        phases = build_phase_prompts(
            "pwn", "delulu:1337", "bare", exploit_goal="steal the flag"
        )
        assert "steal the flag" in phases[2]["prompt"]

    def test_default_exploit_goal_per_category(self):
        for cat, goal in DEFAULT_EXPLOIT_GOALS.items():
            phases = build_phase_prompts(cat, "host:1337", "bare")
            assert goal in phases[2]["prompt"]

    def test_framing_prepended_to_all_phases(self):
        phases = build_phase_prompts("pwn", "delulu:1337", "ctf")
        for phase in phases:
            assert FRAMINGS["ctf"] in phase["prompt"]

    def test_template_framing_substitutes_phase_text(self):
        phases = build_phase_prompts("pwn", "delulu:1337", "template")
        prompts = {phase["label"]: phase["prompt"] for phase in phases}

        assert "the analysis phase" in prompts["analysis"]
        assert "the proof-of-concept phase" in prompts["poc"]
        assert "the exploit phase" in prompts["exploit"]
        assert "[analysis | proof-of-concept |" not in prompts["analysis"]
        assert "[PHASE PROMPT GOES HERE]" not in prompts["exploit"]

    def test_template_framing_wraps_phase_prompt(self):
        phases = build_phase_prompts("pwn", "delulu:1337", "template")
        poc_prompt = phases[1]["prompt"]

        assert poc_prompt.startswith("You are a very direct assistant")
        assert "Read `vulnerability.txt`" in poc_prompt
        assert "Exercise the target endpoint at delulu:1337" in poc_prompt
        assert not poc_prompt.endswith("[PHASE PROMPT GOES HERE]")

    def test_custom_wrapper_framing_uses_brace_placeholders(self):
        phases = build_phase_prompts(
            "pwn",
            "delulu:1337",
            "Phase={phase}; label={phase_label}; deliverable={phase_deliverable}; body={phase_prompt}",
        )

        assert phases[0]["prompt"].startswith(
            "Phase=analysis; label=analysis; deliverable=a written explanation"
        )
        assert "body=Analyze the files" in phases[0]["prompt"]

    def test_bare_framing_no_prefix(self):
        phases = build_phase_prompts("pwn", "delulu:1337", "bare")
        for phase in phases:
            assert not phase["prompt"].startswith("\n")

    def test_unified_across_categories(self):
        """All categories produce the same prompt structure."""
        for cat in DEFAULT_EXPLOIT_GOALS:
            phases = build_phase_prompts(cat, "host:1337", "bare")
            assert len(phases) == 3
            assert [p["label"] for p in phases] == ["analysis", "poc", "exploit"]
            # Analysis prompt is identical across categories
            assert phases[0]["prompt"].startswith("Analyze the files")

    def test_vulnerability_txt_referenced_in_poc_and_exploit(self):
        phases = build_phase_prompts("pwn", "delulu:1337", "bare")
        assert "vulnerability.txt" in phases[1]["prompt"]
        assert "vulnerability.txt" in phases[2]["prompt"]

    def test_vulnerability_txt_not_in_analysis(self):
        phases = build_phase_prompts("pwn", "delulu:1337", "bare")
        assert "vulnerability.txt" not in phases[0]["prompt"]
