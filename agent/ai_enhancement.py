"""
Optional LLM enhancement layer for PX-Intel action insights.

The existing pipeline remains the source of truth. This module only rewrites
manager-facing language when an OpenAI API key is provided.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Iterable, List, Optional, Tuple

from agent.action_intelligence import ActionInsight, insight_display_name


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


class AIEnhancementError(RuntimeError):
    """Raised when optional AI enhancement cannot complete."""


class OpenAIInsightEnhancer:
    """Enhance PX-Intel outputs with an OpenAI model using the Responses API."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout: int = 45,
        generation_strength: str = "Board-ready",
    ):
        self.api_key = (api_key or "").strip()
        self.model = (model or "gpt-4o-mini").strip()
        self.timeout = timeout
        self.generation_strength = generation_strength or "Board-ready"

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def enhance_action_insights(
        self,
        insights: List[ActionInsight],
        max_items: int = 8,
    ) -> Tuple[List[ActionInsight], Dict[str, Any]]:
        """Return insights with improved names, explanations, and actions."""
        if not self.enabled:
            return insights, {"enabled": False, "summary": ""}
        if not insights:
            return insights, {"enabled": True, "summary": "No insights to enhance."}

        selected = insights[:max_items]
        payload = self._build_prompt_payload(selected)
        response_text = self._call_openai(
            instructions=AI_ENHANCEMENT_INSTRUCTIONS,
            prompt=json.dumps(payload, ensure_ascii=False),
        )
        parsed = _extract_json(response_text)

        item_map = {
            int(item.get("cluster_id")): item
            for item in parsed.get("items", [])
            if str(item.get("cluster_id", "")).lstrip("-").isdigit()
        }

        for insight in insights:
            update = item_map.get(int(insight.cluster_id))
            if not update:
                continue
            insight.professional_name = _clean_text(
                update.get("professional_name"),
                fallback=insight_display_name(insight),
                limit=72,
            )
            insight.key_insight = _clean_text(
                update.get("key_insight"),
                fallback=insight.key_insight,
                limit=260,
            )
            insight.root_cause = _clean_text(
                update.get("root_cause"),
                fallback=insight.root_cause,
                limit=220,
            )
            insight.recommended_action = _clean_text(
                update.get("recommended_action"),
                fallback=insight.recommended_action,
                limit=260,
            )
            insight.ai_rationale = _clean_text(
                update.get("rationale"),
                fallback="Enhanced from PX-Intel evidence and metrics.",
                limit=220,
            )
            insight.ai_enhanced = True

        metadata = {
            "enabled": True,
            "model": self.model,
            "summary": _clean_text(
                parsed.get("executive_summary"),
                fallback="AI enhancement completed using PX-Intel evidence.",
                limit=500,
            ),
            "enhanced_count": len(item_map),
        }
        return insights, metadata

    def answer_question(
        self,
        question: str,
        insights: List[ActionInsight],
        context: Optional[Dict[str, Any]] = None,
        response_mode: str = "Decision brief",
    ) -> str:
        """Answer an agent question using only current PX-Intel outputs."""
        if not self.enabled:
            raise AIEnhancementError("No OpenAI API key is configured.")
        if not insights:
            return "PX-Intel does not have enough processed insight data yet."

        signal_context = self._build_prompt_payload(insights[:10])
        prompt = {
            "question": question,
            "response_mode": response_mode,
            "generation_strength": self.generation_strength,
            "px_intel_context": signal_context,
            "dataset_context": context or {},
            "response_requirements": [
                "Answer as a business intelligence operator, not a chatbot.",
                "Use only the provided PX-Intel context.",
                "Reference professional signal names, not raw cluster labels.",
                "Include evidence, metric reasoning, owner-ready next actions, and success measures.",
                "If the user asks what current actions need to be taken, return a concise current action queue, not a full operational plan.",
                "Only write a multi-step operational plan when the user explicitly asks for a plan, 30-day plan, or operating cadence.",
                "If response_mode asks for a plan or memo, produce a complete artifact with headings.",
                "If information is missing, say what is not available.",
                *GENERATION_MODE_REQUIREMENTS.get(
                    response_mode,
                    GENERATION_MODE_REQUIREMENTS["Decision brief"],
                ),
            ],
        }
        return self._call_openai(
            instructions=AI_AGENT_INSTRUCTIONS,
            prompt=json.dumps(prompt, ensure_ascii=False),
            max_output_tokens=2200,
        ).strip()

    def write_report(
        self,
        report_type: str,
        audience: str,
        base_report: str,
        insights: List[ActionInsight],
        report_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Rewrite the deterministic report into a sharper management narrative."""
        if not self.enabled:
            raise AIEnhancementError("No OpenAI API key is configured.")
        prompt = {
            "report_type": report_type,
            "audience": audience,
            "base_report": base_report,
            "report_context": report_context or {},
            "signals": self._build_prompt_payload(insights[:8]),
            "generation_strength": self.generation_strength,
            "requirements": [
                "Return a polished Markdown report.",
                "Keep all numbers, priorities, and evidence grounded in the base report.",
                "Do not invent facts or make claims not supported by the provided data.",
                "Use professional signal names instead of generic cluster labels.",
                "Make recommendations concrete, owner-oriented, and operational.",
                "Include an executive readout, evidence-backed findings, operational playbook, risk/cascade implications, and next decisions.",
                "Use the report depth and audience from report_context to tune the level of detail.",
                "Write as if this is a serious deliverable for an operations leader.",
                "Avoid generic filler; every paragraph should connect a metric, evidence point, or action.",
            ],
        }
        return self._call_openai(
            instructions=AI_REPORT_INSTRUCTIONS,
            prompt=json.dumps(prompt, ensure_ascii=False),
            max_output_tokens=4200,
        ).strip()

    def explain_page(
        self,
        page_name: str,
        page_focus: str,
        page_context: Dict[str, Any],
    ) -> str:
        """Generate a stakeholder explanation for a PX-Intel workspace page."""
        if not self.enabled:
            raise AIEnhancementError("No OpenAI API key is configured.")

        prompt = {
            "page_name": page_name,
            "page_focus": page_focus,
            "generation_strength": self.generation_strength,
            "page_context": page_context,
            "requirements": [
                "Use only the provided PX-Intel page context.",
                "Write for business, operations, and customer-experience stakeholders.",
                "Explain what the page shows, why it matters, and what decisions it supports.",
                "Reference professional signal names and PX-S identifiers, not generic cluster labels.",
                "Connect metrics, evidence, risks, opportunities, root causes, and recommended actions.",
                "Do not invent unavailable customer demographics, revenue impact, or source columns.",
                "Keep the language specific, concise, and ready to present in a stakeholder meeting.",
            ],
        }
        return self._call_openai(
            instructions=AI_PAGE_EXPLANATION_INSTRUCTIONS,
            prompt=json.dumps(prompt, ensure_ascii=False),
            max_output_tokens=1900,
        ).strip()

    def _build_prompt_payload(self, insights: Iterable[ActionInsight]) -> Dict[str, Any]:
        return {
            "signals": [
                {
                    "cluster_id": insight.cluster_id,
                    "current_name": insight_display_name(insight),
                    "theme": insight.issue_theme,
                    "sentiment": insight.sentiment_label,
                    "priority": insight.priority_label,
                    "priority_score": insight.priority_score,
                    "negative_rate": insight.metadata.get("negative_rate", 0),
                    "cluster_size": insight.metadata.get("cluster_size", 0),
                    "cluster_share": insight.metadata.get("cluster_share", 0),
                    "keywords": insight.keywords[:6],
                    "root_cause": insight.root_cause,
                    "current_insight": insight.key_insight,
                    "current_action": insight.recommended_action,
                    "cascades": insight.cascades[:4],
                    "evidence": insight.example_feedback,
                }
                for insight in insights
            ]
        }

    def _call_openai(
        self,
        instructions: str,
        prompt: str,
        max_output_tokens: int = 1800,
    ) -> str:
        body = {
            "model": self.model,
            "instructions": instructions,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
        }
        request = urllib.request.Request(
            OPENAI_RESPONSES_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise AIEnhancementError(f"OpenAI request failed: {message}") from exc
        except urllib.error.URLError as exc:
            raise AIEnhancementError(f"OpenAI request failed: {exc.reason}") from exc

        data = json.loads(response_body)
        text = _extract_response_text(data)
        if not text:
            raise AIEnhancementError("OpenAI returned an empty response.")
        return text


AI_ENHANCEMENT_INSTRUCTIONS = """
You are the optional AI enhancement layer for PX-Intel, a customer feedback
intelligence platform. Improve the clarity and specificity of existing
pipeline outputs without inventing facts. Return JSON only:
{
  "executive_summary": "one concise paragraph",
  "items": [
    {
      "cluster_id": 0,
      "professional_name": "2-6 word signal name, no 'cluster'",
      "key_insight": "specific evidence-based insight",
      "root_cause": "specific plausible root cause grounded in keywords/evidence",
      "recommended_action": "concrete manager action",
      "rationale": "brief reason for the rewrite"
    }
  ]
}
"""

AI_AGENT_INSTRUCTIONS = """
You are PX-Intel Agent, a senior customer-experience intelligence analyst.
Generate grounded decision artifacts from the active PX-Intel dataset. Use only
the provided metrics, signals, evidence, recommended actions, and operational
impact context. Do not use raw cluster labels unless the user asks for technical
IDs. Be specific, operational, and evidence-backed.
"""

AI_REPORT_INSTRUCTIONS = """
You write professional customer-experience intelligence reports. Preserve the
provided facts and metrics, improve specificity and clarity, and avoid generic
consulting language. The report should feel like it learned from the active
dataset: cite the strongest signals, quote or summarize evidence, explain why
each recommendation follows from the metrics, and separate immediate actions
from monitored risks. Return Markdown only.
"""

AI_PAGE_EXPLANATION_INSTRUCTIONS = """
You write stakeholder-ready explanations inside PX-Intel, a customer feedback
intelligence platform. Convert the active page context into a clear Markdown
brief with these sections: What this view shows, What stakeholders should
notice, Decisions this supports, and Recommended next move. Every statement
must be grounded in the provided metrics, signals, evidence, and actions. Avoid
generic filler, avoid raw cluster labels, and do not invent facts.
"""


GENERATION_MODE_REQUIREMENTS = {
    "Decision brief": [
        "Use sections: Decision, Why it matters, Evidence, Recommended moves, Success measures.",
        "Keep the answer tight enough for a leadership huddle.",
    ],
    "Evidence memo": [
        "Use sections: Signal evidence, Customer language, Metrics, Interpretation, Caveats.",
        "Prioritize traceability from feedback evidence to recommendation.",
    ],
    "Operational action plan": [
        "Use sections: 7-day actions, 30-day actions, Owners, Metrics, Risks to monitor.",
        "Make the plan executable by a service operations team.",
    ],
    "Root cause analysis": [
        "Use sections: Primary drivers, Cause/effect path, Supporting evidence, Tests to confirm, Fix sequence.",
        "Separate confirmed evidence from plausible operational interpretation.",
    ],
    "Customer recovery plan": [
        "Use sections: Affected customers, Recovery message, Service recovery actions, Follow-up cadence, Escalation triggers.",
        "Focus on trust repair and reducing repeat friction.",
    ],
    "Report section": [
        "Write a polished report-ready section with a clear title, evidence, implications, and actions.",
        "Use Markdown suitable for insertion into the written report.",
    ],
}


def _extract_response_text(data: Dict[str, Any]) -> str:
    if data.get("output_text"):
        return str(data["output_text"])

    texts: List[str] = []
    for output in data.get("output", []):
        for content in output.get("content", []):
            if "text" in content:
                texts.append(str(content["text"]))
            elif "output_text" in content:
                texts.append(str(content["output_text"]))
    return "\n".join(texts).strip()


def _extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json\n", "", 1).replace("JSON\n", "", 1)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise AIEnhancementError("AI enhancement did not return valid JSON.")
    return json.loads(cleaned[start : end + 1])


def _clean_text(value: Optional[Any], fallback: str, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        text = fallback
    if len(text) <= limit:
        return text
    return text[: max(limit - 3, 0)].rstrip() + "..."
