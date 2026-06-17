"""
M5: Customer Experience Action Intelligence
Rule-based decision-support layer for cluster, sentiment, keyword, and causal outputs.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


SOFT_CASCADE_MAP = {
    "wait": ["staffing", "scheduling", "capacity planning"],
    "queue": ["staffing", "scheduling", "flow management"],
    "delay": ["staffing", "scheduling", "process bottlenecks"],
    "appointment": ["scheduling", "reminders", "capacity planning"],
    "staff": ["training", "communication", "service standards"],
    "communication": ["staff training", "expectation setting", "follow-up process"],
    "rude": ["coaching", "empathy training", "manager review"],
    "clean": ["maintenance", "inspection", "facility standards"],
    "dirty": ["maintenance", "inspection", "facility standards"],
    "billing": ["cost transparency", "billing support", "dispute handling"],
    "cost": ["pricing clarity", "billing support", "customer education"],
    "price": ["pricing clarity", "billing support", "customer education"],
    "quality": ["quality review", "standard operating procedures", "service recovery"],
    "treatment": ["quality review", "standard operating procedures", "follow-up process"],
}


RECOMMENDATION_RULES = {
    "wait": "Review staffing coverage, appointment spacing, and queue handoffs for peak-demand periods.",
    "queue": "Reduce queue friction with clearer routing, live wait updates, and overflow staffing triggers.",
    "delay": "Map the delay points and assign owners for the two slowest steps in the service flow.",
    "appointment": "Improve scheduling rules, reminder cadence, and rescheduling options.",
    "staff": "Run focused coaching on tone, empathy, and issue escalation for customer-facing teams.",
    "communication": "Create clearer status updates, handoff scripts, and follow-up expectations.",
    "rude": "Review service conduct patterns and reinforce behavior standards with team leads.",
    "clean": "Increase inspection cadence and define visible ownership for facility readiness.",
    "dirty": "Treat cleanliness comments as a facility readiness issue and audit high-traffic areas first.",
    "billing": "Improve billing explanations, escalation paths, and proactive cost transparency.",
    "cost": "Clarify pricing before service delivery and make billing support easier to reach.",
    "price": "Clarify pricing before service delivery and make billing support easier to reach.",
    "quality": "Audit quality standards and close the loop with customers affected by recurring failures.",
    "treatment": "Review service quality protocols and confirm that follow-up expectations are clear.",
}


@dataclass
class ActionInsight:
    """Decision-support output for one discovered cluster."""

    cluster_id: int
    issue_theme: str
    sentiment_label: str
    sentiment_emoji: str
    priority_label: str
    priority_score: float
    key_insight: str
    recommended_action: str
    keywords: List[str]
    root_cause: str
    cascades: List[str]
    example_feedback: str
    professional_name: str = ""
    ai_enhanced: bool = False
    ai_rationale: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class CXActionIntelligenceAgent:
    """Create practical action guidance from M1-M3 outputs."""

    def build_action_insights(
        self,
        audit_engine,
        causal_engine=None,
        clustering_engine=None,
    ) -> List[ActionInsight]:
        """Build one action insight per cluster from existing pipeline outputs."""
        total_entries = sum(
            len(texts) for texts in getattr(audit_engine, "cluster_texts", {}).values()
        )
        total_entries = max(total_entries, 1)
        insights = []

        cluster_ids = sorted(getattr(audit_engine, "cluster_texts", {}).keys())
        for cluster_id in cluster_ids:
            texts = audit_engine.cluster_texts.get(cluster_id, [])
            sentiment = audit_engine.cluster_sentiment_results.get(cluster_id, {})
            vocabulary = audit_engine.cluster_vocabularies.get(cluster_id, [])
            zone = audit_engine.cluster_zones.get(cluster_id, {})

            keywords = [self._clean_phrase(keyword) for keyword, _ in vocabulary[:6]]
            issue_theme = self._infer_issue_theme(keywords)
            negative_rate = sentiment.get("sentiment_distribution", {}).get(
                "NEGATIVE", 0.0
            )
            priority_score = calculate_priority_score(
                negativity=negative_rate,
                cluster_size=len(texts),
                total_entries=total_entries,
                keyword_scores=vocabulary,
                feedback_texts=texts,
            )
            priority_label = priority_label_for_score(priority_score)
            sentiment_label = sentiment.get("dominant_sentiment", "UNKNOWN")
            sentiment_emoji = sentiment_emoji_for_label(sentiment_label)
            root_cause = self._extract_root_cause(causal_engine, cluster_id, keywords)
            cascades = generate_soft_cascades(
                keywords=keywords,
                causal_engine=causal_engine,
                cluster_id=cluster_id,
            )
            key_insight = generate_insight_summary(
                cluster_id=cluster_id,
                issue_theme=issue_theme,
                sentiment_label=sentiment_label,
                negative_rate=negative_rate,
                cluster_size=len(texts),
                priority_label=priority_label,
            )
            recommended_action = generate_recommendation(
                issue_theme=issue_theme,
                keywords=keywords,
                sentiment_label=sentiment_label,
                priority_label=priority_label,
            )
            professional_name = professional_issue_name(
                issue_theme=issue_theme,
                sentiment_label=sentiment_label,
                priority_label=priority_label,
                keywords=keywords,
            )

            insights.append(
                ActionInsight(
                    cluster_id=int(cluster_id),
                    issue_theme=issue_theme,
                    sentiment_label=sentiment_label,
                    sentiment_emoji=sentiment_emoji,
                    priority_label=priority_label,
                    priority_score=priority_score,
                    key_insight=key_insight,
                    recommended_action=recommended_action,
                    keywords=keywords,
                    root_cause=root_cause,
                    cascades=cascades,
                    example_feedback=self._pick_example_feedback(texts),
                    professional_name=professional_name,
                    metadata={
                        "zone_type": zone.get("zone_type", "UNKNOWN"),
                        "cluster_size": len(texts),
                        "negative_rate": negative_rate,
                        "keyword_density": keyword_frequency_density(
                            vocabulary, texts
                        ),
                        "cluster_share": len(texts) / total_entries,
                    },
                )
            )

        insights.sort(key=lambda item: item.priority_score, reverse=True)
        return insights

    def build_dashboard_dataframe(self, insights: List[ActionInsight]) -> pd.DataFrame:
        """Return the compact dashboard table for Streamlit."""
        return pd.DataFrame(
            [
                {
                    "Signal ID": signal_reference(item.cluster_id),
                    "Cluster": item.cluster_id,
                    "Experience Signal": insight_display_name(item),
                    "Issue (theme)": item.issue_theme.title(),
                    "Sentiment": f"{item.sentiment_emoji} {item.sentiment_label.title()}",
                    "Priority": item.priority_label,
                    "Key Insight": item.key_insight,
                    "Recommended Action": item.recommended_action,
                }
                for item in insights
            ]
        )

    def answer_question(self, question: str, insights: List[ActionInsight]) -> str:
        """Answer common manager questions using the generated action insights."""
        normalized = question.strip().lower()
        if not normalized:
            return "Ask me which issue to fix first, why a cluster is high priority, or what actions to take next."

        if not insights:
            return "I do not have action insights yet. Run the discovery, audit, and reasoning pipeline first."

        referenced_signal = self._find_referenced_signal(normalized, insights)

        if self._is_nli_question(normalized):
            return self._answer_nli_explanation(insights, referenced_signal)

        if any(token in normalized for token in ("evidence", "proof", "example", "examples", "comment", "comments", "said")):
            return self._answer_evidence(insights, referenced_signal)

        if any(phrase in normalized for phrase in ("root cause", "why is", "why are", "why", "cause", "driver", "drivers")):
            return self._answer_root_causes(insights, referenced_signal)

        if any(token in normalized for token in ("risk", "risks", "negative", "worst", "concern", "concerns", "problem", "problems")):
            return self._answer_risks(insights)

        if any(token in normalized for token in ("strength", "strengths", "positive", "opportunity", "opportunities", "working", "good")):
            return self._answer_opportunities(insights)

        if any(token in normalized for token in ("count", "counts", "many", "volume", "metrics", "numbers", "score", "scores")):
            return self._answer_metrics(insights, referenced_signal)

        if any(token in normalized for token in ("compare", "comparison", "versus", "vs", "difference")):
            return self._answer_comparison(insights)

        if any(token in normalized for token in ("owner", "own", "owns", "ownership", "responsible", "assign", "assigned")):
            return self._answer_owners(insights, referenced_signal)

        if any(token in normalized for token in ("first", "priority", "urgent", "fix")):
            return self._answer_priorities(insights)

        if any(token in normalized for token in ("cascade", "connected", "link", "related")):
            return self._answer_cascades(insights)

        if any(
            phrase in normalized
            for phrase in (
                "30-day",
                "30 day",
                "action plan",
                "operational plan",
                "plan",
            )
        ):
            return self._answer_actions(insights)

        if any(token in normalized for token in ("recommend", "action", "next", "do")):
            return self._answer_current_actions(insights)

        if any(token in normalized for token in ("summary", "summarize", "leadership", "executive")):
            return self._answer_summary(insights)

        if "cluster" in normalized:
            cluster_id = self._extract_cluster_id(normalized)
            if cluster_id is not None:
                return self._answer_cluster(cluster_id, insights)

        if referenced_signal is not None:
            return self._answer_signal_question(referenced_signal)

        return self._answer_summary(insights)

    def _is_nli_question(self, normalized: str) -> bool:
        """Detect glossary questions about PX-Intel's NLI support language."""
        if "nli" in normalized:
            return True
        glossary_phrases = (
            "natural language inference",
            "what does support mean",
            "weak support",
            "causal support",
            "issue signals supported",
            "supported issue signals",
            "entailment",
        )
        return any(phrase in normalized for phrase in glossary_phrases)

    def _answer_nli_explanation(
        self,
        insights: List[ActionInsight],
        focus: Optional[ActionInsight] = None,
    ) -> str:
        """Explain NLI support in stakeholder-friendly language."""
        selected = [focus] if focus is not None else insights[:4]
        support_rows = []
        for item in selected:
            support_match = re.search(
                r"(\d+(?:\.\d+)?)%\s+(weak\s+)?NLI support",
                str(item.root_cause),
                flags=re.IGNORECASE,
            )
            if not support_match:
                continue
            support_rows.append(
                {
                    "insight": item,
                    "support": float(support_match.group(1)),
                    "weak": bool(support_match.group(2)),
                }
            )

        lines = [
            "### What NLI means in PX-Intel",
            "",
            (
                "**NLI** means **Natural Language Inference**. PX-Intel uses it in the "
                "causal-reasoning step to test whether a feedback theme or keyword appears "
                "to support a sentiment or root-cause hypothesis."
            ),
            "",
            (
                "When you see wording like **91% NLI support**, read it as model confidence "
                "that the phrase is a plausible driver of the signal. It is **not** saying "
                "that 91% of customers said the same thing, and it is not absolute proof of causality."
            ),
            "",
            "#### How to use it",
            "",
            "- High NLI support means the root-cause hypothesis is stronger and worth operational review.",
            "- Weak NLI support means PX-Intel is flagging a possible cause, but the team should validate it with more evidence.",
            "- Always pair NLI support with feedback volume, negative concentration, representative comments, and the recommended action.",
        ]

        if support_rows:
            lines.extend(["", "#### In the current data"])
            for row in support_rows:
                item = row["insight"]
                qualifier = "weak support" if row["weak"] else "support"
                lines.append(
                    (
                        f"- **{insight_display_name(item, include_reference=True)}** shows "
                        f"**{row['support']:.0f}% NLI {qualifier}** for the root-cause phrase "
                        f"`{item.root_cause}`. Treat that as a confidence signal behind the "
                        f"recommended action, not as a standalone decision."
                    )
                )

        lines.extend(
            [
                "",
                "#### Plain-English translation",
                "",
                (
                    "PX-Intel is saying: “The language in the feedback appears to support this "
                    "root-cause explanation, so use it as evidence for where to investigate first.”"
                ),
            ]
        )
        return "\n".join(lines)

    def _infer_issue_theme(self, keywords: List[str]) -> str:
        """Infer a manager-friendly issue theme from extracted keywords."""
        joined = " ".join(keywords).lower()
        theme_rules = [
            (("wait", "queue", "delay", "appointment", "schedule"), "wait and scheduling"),
            (("staff", "rude", "friendly", "communication", "empathy"), "staff communication"),
            (("clean", "dirty", "room", "facility", "maintenance"), "cleanliness and facilities"),
            (("billing", "cost", "price", "payment", "expensive"), "billing and cost clarity"),
            (("quality", "treatment", "service", "care", "outcome"), "service quality"),
            (("poor", "bad", "good", "great", "excellent", "experience"), "general experience"),
        ]
        for tokens, theme in theme_rules:
            if any(token in joined for token in tokens):
                return theme
        return keywords[0] if keywords else "general experience"

    def _extract_root_cause(
        self, causal_engine, cluster_id: int, keywords: List[str]
    ) -> str:
        """Use M3 validations when available, otherwise fall back to the theme keywords."""
        if causal_engine is not None:
            validations = getattr(causal_engine, "causal_validations", {}).get(
                cluster_id, []
            )
            confirmed = [item for item in validations if item.get("causal_confirmed")]
            if confirmed:
                best = max(confirmed, key=lambda item: item.get("confidence", 0.0))
                return (
                    f"{self._clean_phrase(best.get('keyword', 'issue'))} "
                    f"({best.get('confidence', 0.0):.0%} NLI support)"
                )
            if validations:
                best = max(validations, key=lambda item: item.get("confidence", 0.0))
                return (
                    f"Probable: {self._clean_phrase(best.get('keyword', 'issue'))} "
                    f"({best.get('confidence', 0.0):.0%} weak NLI support)"
                )

        return f"Probable: {keywords[0]}" if keywords else "No clear root cause yet"

    def _pick_example_feedback(self, texts: List[str]) -> str:
        """Choose a readable representative example."""
        if not texts:
            return "No example feedback available."
        text = max(texts, key=len).strip()
        return text[:280] + ("..." if len(text) > 280 else "")

    @staticmethod
    def _clean_phrase(text: str) -> str:
        """Convert model fragments into plain language."""
        return str(text).replace("_", " ").replace("-", " ").strip()

    def _answer_priorities(self, insights: List[ActionInsight]) -> str:
        """Explain the highest-priority signals as a decision brief."""
        lead = insights[0]
        lines = [
            f"### Decision: fix {insight_display_name(lead, include_reference=True)} first",
            "",
            (
                f"PX-Intel ranks this as the first issue to address because it is "
                f"**{lead.priority_label}**, has a priority score of "
                f"**{lead.priority_score:.3f}**, and represents "
                f"**{lead.metadata.get('cluster_size', 0):,} related comments** "
                f"with **{lead.metadata.get('negative_rate', 0):.0%} negative concentration**."
            ),
            "",
            "#### Why it matters",
            "",
            (
                f"{lead.key_insight} The pattern points to **{lead.root_cause}**, "
                "which means the fix should target the operating condition behind the feedback, not only the complaint language."
            ),
            "",
            "#### Immediate action",
            "",
            f"- **Owner:** {self._owner_for(lead)}",
            f"- **Timeline:** {self._action_window_for(lead)}",
            f"- **Move:** {lead.recommended_action}",
            f"- **Success metric:** {self._success_metric_for(lead)}",
            f"- **Risk if ignored:** {self._risk_if_ignored(lead)}",
        ]

        watchlist = insights[1:3]
        if watchlist:
            lines.extend(["", "#### Next watchlist"])
            for item in watchlist:
                lines.append(
                    (
                        f"- **{insight_display_name(item, include_reference=True)}** "
                        f"({item.priority_label}, score {item.priority_score:.3f}): "
                        f"{item.recommended_action}"
                    )
                )
        return "\n".join(lines)

    def _answer_actions(self, insights: List[ActionInsight]) -> str:
        """List recommended actions as an operational action plan."""
        lines = [
            "### Operational action plan",
            "",
            "Use this as the starting work queue for the current feedback cycle.",
            "",
        ]
        for index, item in enumerate(insights[:5], start=1):
            lines.append(
                (
                    f"{index}. **{insight_display_name(item, include_reference=True)}** "
                    f"({item.priority_label})\n"
                    f"   - Owner: {self._owner_for(item)}\n"
                    f"   - Timeline: {self._action_window_for(item)}\n"
                    f"   - Action: {item.recommended_action}\n"
                    f"   - Success metric: {self._success_metric_for(item)}"
                )
            )
        return "\n".join(lines)

    def _answer_current_actions(self, insights: List[ActionInsight]) -> str:
        """Return a direct current action queue instead of a full plan."""
        urgent = [
            item
            for item in insights
            if item.priority_label.startswith("HIGH")
            or item.metadata.get("negative_rate", 0) >= 0.75
        ]
        watchlist = [
            item
            for item in insights
            if item not in urgent
            and (
                item.priority_label.startswith("MEDIUM")
                or item.metadata.get("negative_rate", 0) >= 0.4
            )
        ]
        selected = []
        seen_clusters = set()
        for item in urgent + watchlist + insights:
            if item.cluster_id in seen_clusters:
                continue
            selected.append(item)
            seen_clusters.add(item.cluster_id)
            if len(selected) >= 4:
                break

        lines = [
            "### Current actions to take",
            "",
            (
                "These are the actions that should be taken from the current PX-Intel evidence. "
                "This is a direct action queue, not a long-range operating plan."
            ),
            "",
        ]
        for index, item in enumerate(selected, start=1):
            action_label = (
                "Immediate action"
                if item in urgent
                else "Next action"
                if item in watchlist
                else "Monitor"
            )
            lines.append(
                (
                    f"{index}. **{action_label}: {insight_display_name(item, include_reference=True)}**\n"
                    f"   - Why now: {item.metadata.get('cluster_size', 0):,} comments, "
                    f"{item.metadata.get('negative_rate', 0):.0%} negative concentration, "
                    f"priority score {item.priority_score:.3f}.\n"
                    f"   - Owner: {self._owner_for(item)}\n"
                    f"   - Timeline: {self._action_window_for(item)}\n"
                    f"   - Action: {item.recommended_action}\n"
                    f"   - Success metric: {self._success_metric_for(item)}"
                )
            )

        return "\n".join(lines)

    def _answer_evidence(
        self,
        insights: List[ActionInsight],
        focus: Optional[ActionInsight] = None,
    ) -> str:
        """Show the evidence behind one signal or the top signals."""
        selected = [focus] if focus is not None else insights[:3]
        lines = [
            "### Evidence from the active feedback",
            "",
            "PX-Intel is using the current processed feedback, signal metrics, keywords, and representative comments.",
            "",
        ]
        for item in selected:
            keywords = ", ".join(item.keywords[:6]) if item.keywords else "No keywords"
            lines.append(
                (
                    f"- **{insight_display_name(item, include_reference=True)}**\n"
                    f"  - Evidence base: {item.metadata.get('cluster_size', 0):,} comments, "
                    f"{item.metadata.get('negative_rate', 0):.0%} negative concentration, "
                    f"priority score {item.priority_score:.3f}.\n"
                    f"  - Keywords: {keywords}\n"
                    f"  - Representative feedback: {item.example_feedback}\n"
                    f"  - Interpretation: {item.key_insight}"
                )
            )
        return "\n".join(lines)

    def _answer_root_causes(
        self,
        insights: List[ActionInsight],
        focus: Optional[ActionInsight] = None,
    ) -> str:
        """Explain root causes from the current signals."""
        selected = [focus] if focus is not None else insights[:4]
        lines = [
            "### Root cause readout",
            "",
            "These are the likely drivers PX-Intel found from the active feedback evidence.",
            "",
        ]
        for item in selected:
            lines.append(
                (
                    f"- **{insight_display_name(item, include_reference=True)}**: "
                    f"{item.root_cause}. Recommended response: {item.recommended_action}"
                )
            )
        return "\n".join(lines)

    def _answer_risks(self, insights: List[ActionInsight]) -> str:
        """Summarize the most important risk signals."""
        risk_items = [
            item
            for item in insights
            if item.priority_label.startswith("HIGH")
            or item.metadata.get("negative_rate", 0) >= 0.4
        ]
        risk_items = risk_items[:4] or insights[:3]
        lines = [
            "### Customer risk signals",
            "",
            "These are the signals most likely to require management attention because they combine negative concentration, volume, and priority score.",
            "",
        ]
        for item in risk_items:
            lines.append(
                (
                    f"- **{insight_display_name(item, include_reference=True)}** "
                    f"({item.priority_label}): {item.metadata.get('cluster_size', 0):,} comments, "
                    f"{item.metadata.get('negative_rate', 0):.0%} negative, score {item.priority_score:.3f}. "
                    f"Action: {item.recommended_action}"
                )
            )
        return "\n".join(lines)

    def _answer_opportunities(self, insights: List[ActionInsight]) -> str:
        """Summarize positive or protectable experience patterns."""
        opportunity_items = [
            item
            for item in insights
            if str(item.sentiment_label).upper() == "POSITIVE"
            or item.metadata.get("negative_rate", 0) <= 0.2
        ]
        opportunity_items = sorted(
            opportunity_items,
            key=lambda item: item.metadata.get("cluster_size", 0),
            reverse=True,
        )[:4]
        if not opportunity_items:
            opportunity_items = insights[-3:]
        lines = [
            "### Strengths and opportunities",
            "",
            "These are the patterns PX-Intel would protect, repeat, or use as service-quality examples.",
            "",
        ]
        for item in opportunity_items:
            lines.append(
                (
                    f"- **{insight_display_name(item, include_reference=True)}**: "
                    f"{item.metadata.get('cluster_size', 0):,} comments, "
                    f"{item.metadata.get('negative_rate', 0):.0%} negative concentration. "
                    f"Recommended move: {item.recommended_action}"
                )
            )
        return "\n".join(lines)

    def _answer_metrics(
        self,
        insights: List[ActionInsight],
        focus: Optional[ActionInsight] = None,
    ) -> str:
        """Answer metric and count questions."""
        if focus is not None:
            return "\n".join(
                [
                    f"### Metrics for {insight_display_name(focus, include_reference=True)}",
                    "",
                    f"- Priority: {focus.priority_label}",
                    f"- Priority score: {focus.priority_score:.3f}",
                    f"- Feedback volume: {focus.metadata.get('cluster_size', 0):,} comments",
                    f"- Negative concentration: {focus.metadata.get('negative_rate', 0):.0%}",
                    f"- Signal share: {focus.metadata.get('cluster_share', 0):.0%} of processed feedback",
                    f"- Root cause: {focus.root_cause}",
                ]
            )

        high_count = sum(1 for item in insights if item.priority_label.startswith("HIGH"))
        medium_count = sum(1 for item in insights if item.priority_label.startswith("MEDIUM"))
        low_count = sum(1 for item in insights if item.priority_label.startswith("LOW"))
        total_comments = sum(item.metadata.get("cluster_size", 0) for item in insights)
        avg_negative = (
            sum(item.metadata.get("negative_rate", 0) for item in insights) / len(insights)
            if insights
            else 0
        )
        return "\n".join(
            [
                "### Current PX-Intel metrics",
                "",
                f"- Experience signals: {len(insights)}",
                f"- Feedback represented in signals: {total_comments:,} comments",
                f"- High priority: {high_count}",
                f"- Medium priority: {medium_count}",
                f"- Low priority: {low_count}",
                f"- Average negative concentration across signals: {avg_negative:.0%}",
                f"- Highest priority signal: {insight_display_name(insights[0], include_reference=True)} ({insights[0].priority_score:.3f})",
            ]
        )

    def _answer_comparison(self, insights: List[ActionInsight]) -> str:
        """Compare the top signals."""
        selected = insights[:3]
        lines = [
            "### Signal comparison",
            "",
            "Here is how the highest-priority signals compare in the active dataset.",
            "",
        ]
        for item in selected:
            lines.append(
                (
                    f"- **{insight_display_name(item, include_reference=True)}**: "
                    f"{item.priority_label}, score {item.priority_score:.3f}, "
                    f"{item.metadata.get('cluster_size', 0):,} comments, "
                    f"{item.metadata.get('negative_rate', 0):.0%} negative. "
                    f"Root cause: {item.root_cause}."
                )
            )
        if len(selected) >= 2:
            lines.extend(
                [
                    "",
                    (
                        f"**Decision implication:** start with "
                        f"{insight_display_name(selected[0], include_reference=True)} "
                        "because it carries the highest combined priority score and negative concentration."
                    ),
                ]
            )
        return "\n".join(lines)

    def _answer_owners(
        self,
        insights: List[ActionInsight],
        focus: Optional[ActionInsight] = None,
    ) -> str:
        """Suggest owners for one or more action signals."""
        selected = [focus] if focus is not None else insights[:4]
        lines = [
            "### Suggested ownership",
            "",
            "PX-Intel assigns owners from the issue theme and action type. Adjust these to match your actual operating model.",
            "",
        ]
        for item in selected:
            lines.append(
                (
                    f"- **{insight_display_name(item, include_reference=True)}**: "
                    f"{self._owner_for(item)}. Timeline: {self._action_window_for(item)}. "
                    f"Action: {item.recommended_action}"
                )
            )
        return "\n".join(lines)

    def _answer_signal_question(self, insight: ActionInsight) -> str:
        """Return a general answer for a matched signal or topic."""
        return "\n".join(
            [
                f"### {insight_display_name(insight, include_reference=True)}",
                "",
                f"{insight.key_insight}",
                "",
                "#### What PX-Intel sees",
                "",
                f"- Priority: {insight.priority_label}",
                f"- Score: {insight.priority_score:.3f}",
                f"- Volume: {insight.metadata.get('cluster_size', 0):,} comments",
                f"- Negative concentration: {insight.metadata.get('negative_rate', 0):.0%}",
                f"- Root cause: {insight.root_cause}",
                "",
                "#### Recommended response",
                "",
                f"- Owner: {self._owner_for(insight)}",
                f"- Timeline: {self._action_window_for(insight)}",
                f"- Action: {insight.recommended_action}",
                f"- Success metric: {self._success_metric_for(insight)}",
            ]
        )

    def _answer_cascades(self, insights: List[ActionInsight]) -> str:
        """Explain cascade visibility."""
        lines = [
            "### Cascade risk readout",
            "",
            "These are the likely operational paths where one customer issue can create pressure in another part of the service experience.",
            "",
        ]
        for item in insights[:5]:
            cascades = "; ".join(item.cascades[:3]) or "No cascade path identified yet"
            lines.append(
                (
                    f"- **{insight_display_name(item, include_reference=True)}**: "
                    f"{cascades}. Recommended control: {item.recommended_action}"
                )
            )
        return "\n".join(lines)

    def _answer_summary(self, insights: List[ActionInsight]) -> str:
        """Generate a stakeholder-ready executive summary."""
        high_count = sum(1 for item in insights if item.priority_label.startswith("HIGH"))
        medium_count = sum(
            1 for item in insights if item.priority_label.startswith("MEDIUM")
        )
        top = insights[0]
        return "\n".join(
            [
                "### Executive decision readout",
                "",
                (
                    f"PX-Intel found **{len(insights)} customer-experience signals**: "
                    f"**{high_count} high priority** and **{medium_count} medium priority**. "
                    f"The leading decision item is **{insight_display_name(top, include_reference=True)}**."
                ),
                "",
                "#### What to do first",
                "",
                (
                    f"Fix **{insight_display_name(top, include_reference=True)}** first. "
                    f"It is scored **{top.priority_score:.3f}**, covers "
                    f"**{top.metadata.get('cluster_size', 0):,} comments**, and has "
                    f"**{top.metadata.get('negative_rate', 0):.0%} negative concentration**."
                ),
                "",
                "#### Evidence",
                "",
                f"- Signal insight: {top.key_insight}",
                f"- Root cause: {top.root_cause}",
                f"- Representative feedback: {top.example_feedback}",
                "",
                "#### Recommended move",
                "",
                f"- Owner: {self._owner_for(top)}",
                f"- Timeline: {self._action_window_for(top)}",
                f"- Action: {top.recommended_action}",
                f"- Success metric: {self._success_metric_for(top)}",
                f"- Risk if ignored: {self._risk_if_ignored(top)}",
            ]
        )

    def _answer_cluster(
        self, cluster_id: int, insights: List[ActionInsight]
    ) -> str:
        """Explain a specific cluster."""
        match = next(
            (item for item in insights if item.cluster_id == cluster_id), None
        )
        if match is None:
            available = ", ".join(signal_reference(item.cluster_id) for item in insights)
            return f"I could not find that signal. Available signal IDs: {available}."

        cascades = "; ".join(match.cascades[:3])
        keywords = ", ".join(match.keywords[:6]) if match.keywords else "None"
        return "\n".join(
            [
                f"### Signal brief: {insight_display_name(match, include_reference=True)}",
                "",
                (
                    f"This signal is about **{match.issue_theme}**. It is marked "
                    f"**{match.priority_label}** with a score of **{match.priority_score:.3f}**."
                ),
                "",
                "#### Evidence",
                "",
                f"- Insight: {match.key_insight}",
                f"- Keywords: {keywords}",
                f"- Root cause: {match.root_cause}",
                f"- Cascades: {cascades or 'No cascade path identified yet'}",
                f"- Representative feedback: {match.example_feedback}",
                "",
                "#### Recommended response",
                "",
                f"- Owner: {self._owner_for(match)}",
                f"- Timeline: {self._action_window_for(match)}",
                f"- Action: {match.recommended_action}",
                f"- Success metric: {self._success_metric_for(match)}",
            ]
        )

    def _extract_cluster_id(self, text: str) -> Optional[int]:
        """Extract a cluster number from a question."""
        parts = text.replace("#", " ").split()
        for part in parts:
            if part.isdigit():
                return int(part)
        return None

    def _find_referenced_signal(
        self,
        text: str,
        insights: List[ActionInsight],
    ) -> Optional[ActionInsight]:
        """Find the signal most directly referenced by a free-form question."""
        normalized = str(text or "").lower()
        for item in insights:
            if signal_reference(item.cluster_id).lower() in normalized:
                return item

        cluster_id = self._extract_cluster_id(normalized)
        if cluster_id is not None:
            for item in insights:
                if item.cluster_id == cluster_id:
                    return item

        stop_words = {
            "and",
            "the",
            "what",
            "why",
            "how",
            "are",
            "for",
            "with",
            "that",
            "this",
            "some",
            "need",
            "needs",
            "taken",
        }
        best_item = None
        best_score = 0
        for item in insights:
            terms = set()
            terms.update(str(item.issue_theme).lower().split())
            terms.update(str(insight_display_name(item)).lower().split())
            terms.update(str(keyword).lower() for keyword in item.keywords[:8])
            score = sum(
                1
                for term in terms
                if len(term) > 2 and term not in stop_words and term in normalized
            )
            if score > best_score:
                best_score = score
                best_item = item
        return best_item if best_score >= 2 else None

    def _owner_for(self, insight: ActionInsight) -> str:
        """Suggest a practical owner from the issue theme."""
        theme = str(insight.issue_theme).lower()
        if "wait" in theme or "scheduling" in theme:
            return "Operations lead"
        if "staff" in theme or "communication" in theme:
            return "Customer experience manager"
        if "clean" in theme or "facility" in theme:
            return "Facilities lead"
        if "billing" in theme or "cost" in theme:
            return "Revenue operations lead"
        return "Site manager"

    def _action_window_for(self, insight: ActionInsight) -> str:
        """Suggest action timing from priority and negative concentration."""
        negative_rate = insight.metadata.get("negative_rate", 0)
        if insight.priority_label.startswith("HIGH") or negative_rate >= 0.75:
            return "Immediate: review this week"
        if insight.priority_label.startswith("MEDIUM") or negative_rate >= 0.4:
            return "Next 7 days"
        return "Monitor in the next feedback cycle"

    def _success_metric_for(self, insight: ActionInsight) -> str:
        """Define a success metric grounded in available signal metadata."""
        negative_rate = insight.metadata.get("negative_rate", 0)
        if negative_rate >= 0.4:
            return "Reduce negative feedback concentration for this signal in the next feedback cycle."
        return "Maintain positive concentration while monitoring for new negative comments."

    def _risk_if_ignored(self, insight: ActionInsight) -> str:
        """Describe the operating risk if the signal is not addressed."""
        theme = str(insight.issue_theme).lower()
        if "wait" in theme or "scheduling" in theme:
            return "Wait friction can compound into missed appointments, lower trust, and repeated service recovery."
        if "staff" in theme or "communication" in theme:
            return "Communication gaps can turn fixable issues into escalations and lower customer confidence."
        if "clean" in theme or "facility" in theme:
            return "Facility concerns can weaken perceived service quality even when the core service performs well."
        if "billing" in theme or "cost" in theme:
            return "Cost confusion can create disputes, callbacks, and avoidable dissatisfaction."
        return "The same friction may keep appearing in future feedback and crowd out higher-value improvement work."


def calculate_priority_score(
    negativity: float,
    cluster_size: int,
    total_entries: int,
    keyword_scores: Iterable[Tuple[str, float]],
    feedback_texts: Optional[List[str]] = None,
) -> float:
    """
    Weighted priority score from sentiment negativity, cluster size, and keyword density.

    Score weights:
    - 50% sentiment negativity
    - 30% cluster size share
    - 20% keyword frequency density
    """
    cluster_share = min(cluster_size / max(total_entries, 1), 1.0)
    density = keyword_frequency_density(keyword_scores, feedback_texts or [])
    score = (0.50 * negativity) + (0.30 * cluster_share) + (0.20 * density)
    return round(min(max(score, 0.0), 1.0), 3)


def keyword_density(keyword_scores: Iterable[Tuple[str, float]]) -> float:
    """Summarize KeyBERT keyword confidence as a compact density score."""
    scores = [float(score) for _, score in list(keyword_scores)[:5]]
    if not scores:
        return 0.0
    return min(sum(scores) / len(scores), 1.0)


def keyword_frequency_density(
    keyword_scores: Iterable[Tuple[str, float]], feedback_texts: List[str]
) -> float:
    """Estimate how frequently top keywords appear across a cluster."""
    keywords = [str(keyword).lower() for keyword, _ in list(keyword_scores)[:5]]
    texts = [str(text).lower() for text in feedback_texts]

    if not keywords:
        return 0.0
    if not texts:
        return keyword_density(keyword_scores)

    matched_docs = 0
    for text in texts:
        if any(keyword in text for keyword in keywords):
            matched_docs += 1

    return round(matched_docs / len(texts), 3)


def priority_label_for_score(score: float) -> str:
    """Convert a numeric score to a decision label."""
    if score >= 0.60:
        return "HIGH 🔥"
    if score >= 0.35:
        return "MEDIUM ⚠"
    return "LOW ✅"


def sentiment_emoji_for_label(label: str) -> str:
    """Return a compact sentiment emoji for table display."""
    normalized = str(label).upper()
    if normalized == "NEGATIVE":
        return "🔴"
    if normalized == "POSITIVE":
        return "🟢"
    if normalized == "NEUTRAL":
        return "🟡"
    return "⚪"


def signal_reference(cluster_id: int) -> str:
    """Return a compact professional signal identifier."""
    return f"PX-S{int(cluster_id) + 1:02d}"


def professional_issue_name(
    issue_theme: str,
    sentiment_label: str,
    priority_label: str,
    keywords: Optional[List[str]] = None,
) -> str:
    """Create a customer-facing signal name instead of exposing raw cluster numbers."""
    theme = str(issue_theme or "experience").replace("_", " ").replace("-", " ").strip()
    theme = " ".join(theme.split()).title()
    if theme.lower() in {"poor", "bad", "good", "great", "excellent"}:
        theme = "General Experience"
    sentiment = str(sentiment_label).upper()
    priority = str(priority_label).upper()

    if sentiment == "POSITIVE":
        suffix = "Strength Signal"
    elif priority.startswith("HIGH"):
        suffix = "Recovery Risk"
    elif sentiment == "NEGATIVE":
        suffix = "Friction Signal"
    else:
        suffix = "Watch Signal"

    return f"{theme} {suffix}"


def insight_display_name(insight: ActionInsight, include_reference: bool = False) -> str:
    """Return the preferred display name for an action insight."""
    name = insight.professional_name or professional_issue_name(
        insight.issue_theme,
        insight.sentiment_label,
        insight.priority_label,
        insight.keywords,
    )
    if include_reference:
        return f"{name} ({signal_reference(insight.cluster_id)})"
    return name


def generate_recommendation(
    issue_theme: str,
    keywords: List[str],
    sentiment_label: str,
    priority_label: str,
) -> str:
    """Generate a specific operational recommendation."""
    keyword_text = " ".join(keywords).lower()
    for trigger, recommendation in RECOMMENDATION_RULES.items():
        if trigger in keyword_text or trigger in issue_theme.lower():
            if priority_label.startswith("HIGH"):
                return f"Act this week: {recommendation}"
            if priority_label.startswith("MEDIUM"):
                return f"Plan next cycle: {recommendation}"
            return f"Monitor: {recommendation}"

    if str(sentiment_label).upper() == "NEGATIVE":
        return "Review the top feedback examples and assign an owner for one service recovery action."
    if str(sentiment_label).upper() == "POSITIVE":
        return "Preserve what is working and reuse this cluster as a service-quality reference."
    return "Monitor this theme and collect more examples before changing the process."


def generate_soft_cascades(
    keywords: List[str],
    causal_engine=None,
    cluster_id: Optional[int] = None,
) -> List[str]:
    """Generate soft operational cascades from predefined mappings and M3 similarities."""
    cascades = []
    keyword_text = " ".join(keywords).lower()

    for trigger, links in SOFT_CASCADE_MAP.items():
        if trigger in keyword_text:
            cascades.append(f"{trigger} → " + " → ".join(links))

    if causal_engine is not None and cluster_id is not None:
        for cascade in getattr(causal_engine, "cascade_predictions", {}).get(
            cluster_id, []
        )[:2]:
            cascades.append(
                f"{signal_reference(cluster_id)} -> {signal_reference(cascade['target_cluster'])} "
                f"({cascade['cascade_likelihood']:.0%} shared pattern)"
            )

    return _dedupe(cascades)[:5] or ["No strong cascade found; monitor adjacent service steps."]


def generate_insight_summary(
    cluster_id: int,
    issue_theme: str,
    sentiment_label: str,
    negative_rate: float,
    cluster_size: int,
    priority_label: str,
) -> str:
    """Generate a one-line plain-language insight for managers."""
    signal_name = professional_issue_name(issue_theme, sentiment_label, priority_label)
    if str(sentiment_label).upper() == "NEGATIVE":
        return (
            f"{signal_name} shows recurring {issue_theme} friction "
            f"across {cluster_size} comments ({negative_rate:.0%} negative)."
        )
    if str(sentiment_label).upper() == "POSITIVE":
        return (
            f"{signal_name} highlights a positive {issue_theme} pattern "
            f"that can be protected or replicated."
        )
    return (
        f"{signal_name} is a mixed {issue_theme} signal; "
        f"{priority_label.lower()} follow-up is appropriate."
    )


def _dedupe(values: List[str]) -> List[str]:
    """Preserve order while removing duplicates."""
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
