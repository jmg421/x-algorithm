#!/usr/bin/env python3
"""
Phoenix Score Predictor — Estimate how X's algorithm will rank your post.

Based on the xAI x-algorithm open source release (May 15, 2026).
Weights are estimated from the Rust source code structure and public signals.
The actual params module is not included in the open-source release.

Author: John Muirhead-Gould (@JonIsGold)
Fork: https://github.com/jmg421/x-algorithm
"""

import re
import sys
import json

# =============================================================================
# ESTIMATED WEIGHTS
#
# ⚠️  THESE ARE NOT THE ACTUAL PRODUCTION WEIGHTS.
#
# The actual weights live in the excluded `params` module (not open-sourced).
# These estimates are derived from:
#
# WHAT WE KNOW FROM SOURCE (facts):
# - The 19 scoring signals and their names (weighted_scorer.rs)
# - The formula: Final Score = Σ (weight_i × P(action_i))
# - not_dwelled is a NEW negative signal (only in RankingScorer, not WeightedScorer)
# - share_via_dm and share_via_copy_link are separate from generic share
# - dwell_time is continuous (seconds), separate from binary dwell
# - click_dwell_time is continuous (only in RankingScorer)
# - Video quality view requires min duration threshold
# - Negative scores get offset: (score + neg_sum) / total_sum * OFFSET
# - Author diversity uses exponential decay with a floor
# - OON posts get multiplied by a factor < 1.0
#
# WHAT WE'RE ESTIMATING (informed guesses):
# - The actual numeric weight values
# - Relative magnitudes between signals
#
# Estimation methodology:
# - Old Twitter algorithm (2023) had known ratios (reply ~11x like, retweet ~5x)
# - New signals (share_dm, not_dwelled) estimated from code emphasis
# - Negative weights estimated from the offset formula structure
# =============================================================================

WEIGHTS = {
    # Positive engagement signals
    "favorite":             1.0,    # Like — baseline signal
    "reply":                11.0,   # Reply — high value (conversation)
    "retweet":              5.0,    # Repost — amplification
    "quote":                15.0,   # Quote tweet — highest positive signal
    "share":                4.0,    # Generic share
    "share_via_dm":         8.0,    # DM share — very high (private recommendation)
    "share_via_copy_link":  6.0,    # Copy link — high (off-platform sharing)
    "click":                2.0,    # Click to detail page
    "profile_click":        3.0,    # Click to author profile
    "dwell":                1.5,    # Binary: did they stop scrolling?
    "dwell_time":           0.3,    # Continuous: how long they stopped (per second)
    "click_dwell_time":     0.5,    # Continuous: dwell after clicking
    "photo_expand":         2.0,    # Expanded an image
    "video_quality_view":   3.5,    # Watched video (>min duration)
    "follow_author":        20.0,   # Followed the author — extremely high
    "quoted_click":         2.5,    # Clicked on a quoted post
    "quoted_vqv":           3.0,    # Watched video in quoted post

    # Negative signals (these SUBTRACT from score)
    "not_interested":       -50.0,  # "Not interested" button
    "not_dwelled":          -1.0,   # Scrolled past without stopping (NEW)
    "block_author":         -150.0, # Blocked the author
    "mute_author":          -100.0, # Muted the author
    "report":               -500.0, # Reported the post
}

# Out-of-network multiplier (OON posts get penalized vs in-network)
OON_WEIGHT_FACTOR = 0.6

# Author diversity: exponential decay for repeated authors
AUTHOR_DIVERSITY_DECAY = 0.7
AUTHOR_DIVERSITY_FLOOR = 0.15

# Minimum video duration for VQV eligibility (ms)
MIN_VIDEO_DURATION_MS = 10000


def analyze_post(text: str, has_image=False, has_video=False, video_duration_s=0,
                 is_reply=False, is_quote=False, is_thread=False) -> dict:
    """Analyze a post and predict its scoring potential."""

    analysis = {
        "text": text[:100] + "..." if len(text) > 100 else text,
        "char_count": len(text),
        "word_count": len(text.split()),
        "signals": {},
        "predicted_score": 0.0,
        "recommendations": [],
    }

    # --- Estimate engagement probabilities based on content features ---

    # Dwell time prediction (longer, more complex posts = more dwell)
    word_count = len(text.split())
    has_numbers = bool(re.search(r'\d+', text))
    has_question = '?' in text
    has_list = bool(re.search(r'[\n•\-\d]\s', text))
    has_link = bool(re.search(r'https?://', text))
    has_code = '```' in text or bool(re.search(r'`[^`]+`', text))

    # Base dwell probability (people stop to read)
    p_dwell = 0.3
    if word_count > 50: p_dwell += 0.15
    if word_count > 100: p_dwell += 0.1
    if has_image: p_dwell += 0.2
    if has_video: p_dwell += 0.25
    if has_numbers: p_dwell += 0.05
    if has_list: p_dwell += 0.1
    if has_code: p_dwell += 0.15
    p_dwell = min(p_dwell, 0.95)

    # Not-dwelled (inverse of dwell — scrolled past)
    p_not_dwelled = 1.0 - p_dwell

    # Estimated dwell time (seconds) for those who do stop
    est_dwell_seconds = (word_count / 4.0)  # ~4 words/sec reading speed
    if has_image: est_dwell_seconds += 3
    if has_video: est_dwell_seconds += video_duration_s * 0.6
    if has_code: est_dwell_seconds += 5

    # Like probability
    p_favorite = 0.03
    if has_image: p_favorite += 0.02
    if has_video: p_favorite += 0.015
    if word_count > 30: p_favorite += 0.01
    if has_numbers: p_favorite += 0.01  # data = credibility

    # Reply probability
    p_reply = 0.005
    if has_question: p_reply += 0.015
    if is_thread: p_reply += 0.005
    if word_count > 50: p_reply += 0.005  # more substance = more to respond to

    # Retweet probability
    p_retweet = 0.01
    if has_numbers: p_retweet += 0.005  # shareable data
    if has_image: p_retweet += 0.005

    # Quote probability
    p_quote = 0.003
    if has_numbers: p_quote += 0.002
    if word_count > 50: p_quote += 0.002  # more to comment on

    # Share probabilities
    p_share = 0.005
    p_share_dm = 0.003
    if has_numbers: p_share_dm += 0.002  # people DM data to friends
    if has_code: p_share_dm += 0.003
    p_share_copy_link = 0.004
    if has_link: p_share_copy_link += 0.002

    # Click probability
    p_click = 0.08
    if is_thread: p_click += 0.05
    if has_link: p_click += 0.03

    # Profile click
    p_profile_click = 0.01

    # Photo expand
    p_photo_expand = 0.15 if has_image else 0.0

    # Video quality view
    p_vqv = 0.0
    if has_video and video_duration_s * 1000 > MIN_VIDEO_DURATION_MS:
        p_vqv = 0.25

    # Follow author (very rare per-post)
    p_follow = 0.001

    # Negative signals (assume low for non-spam content)
    p_not_interested = 0.005
    p_block = 0.0005
    p_mute = 0.001
    p_report = 0.0001

    # --- Compute weighted score ---
    signals = {
        "favorite": p_favorite,
        "reply": p_reply,
        "retweet": p_retweet,
        "quote": p_quote,
        "share": p_share,
        "share_via_dm": p_share_dm,
        "share_via_copy_link": p_share_copy_link,
        "click": p_click,
        "profile_click": p_profile_click,
        "dwell": p_dwell,
        "dwell_time": est_dwell_seconds * p_dwell * 0.01,  # normalized
        "click_dwell_time": est_dwell_seconds * p_click * 0.01,
        "photo_expand": p_photo_expand,
        "video_quality_view": p_vqv,
        "follow_author": p_follow,
        "quoted_click": 0.01 if is_quote else 0.0,
        "quoted_vqv": 0.0,
        "not_dwelled": p_not_dwelled,
        "not_interested": p_not_interested,
        "block_author": p_block,
        "mute_author": p_mute,
        "report": p_report,
    }

    weighted_score = 0.0
    score_breakdown = {}

    for signal, probability in signals.items():
        weight = WEIGHTS.get(signal, 0.0)
        contribution = probability * weight
        weighted_score += contribution
        score_breakdown[signal] = {
            "probability": round(probability, 4),
            "weight": weight,
            "contribution": round(contribution, 4),
        }

    analysis["signals"] = score_breakdown
    analysis["predicted_score"] = round(weighted_score, 4)

    # Sort by absolute contribution
    top_positive = sorted(
        [(k, v) for k, v in score_breakdown.items() if v["contribution"] > 0],
        key=lambda x: x[1]["contribution"], reverse=True
    )[:5]
    top_negative = sorted(
        [(k, v) for k, v in score_breakdown.items() if v["contribution"] < 0],
        key=lambda x: x[1]["contribution"]
    )[:3]

    analysis["top_positive_signals"] = [(k, v["contribution"]) for k, v in top_positive]
    analysis["top_negative_signals"] = [(k, v["contribution"]) for k, v in top_negative]

    # --- Recommendations ---
    if not has_image and not has_video:
        analysis["recommendations"].append("Add an image or video — increases dwell time and photo_expand signal")
    if not has_question:
        analysis["recommendations"].append("Add a question — drives replies (11× weight)")
    if word_count < 30:
        analysis["recommendations"].append("Write more — short posts get scrolled past (not_dwelled penalty)")
    if not has_numbers:
        analysis["recommendations"].append("Include data/numbers — increases shares and DM shares (8× weight)")
    if word_count > 200:
        analysis["recommendations"].append("Consider a thread — long single posts may lose readers")
    if p_not_dwelled > 0.5:
        analysis["recommendations"].append("⚠️ High scroll-past risk — make first line a hook")

    return analysis


def print_report(analysis: dict):
    """Print a formatted score report."""
    print("\n" + "=" * 60)
    print("  PHOENIX SCORE PREDICTOR")
    print("  Based on xAI x-algorithm source (May 2026)")
    print("  ⚠️  Weights are ESTIMATES — actual params not open-sourced")
    print("=" * 60)
    print(f"\n  Post: {analysis['text']}")
    print(f"  Words: {analysis['word_count']} | Chars: {analysis['char_count']}")
    print(f"\n  {'─' * 56}")
    print(f"  PREDICTED SCORE: {analysis['predicted_score']:.4f}")
    print(f"  {'─' * 56}")

    print("\n  📈 Top Positive Signals:")
    for signal, contribution in analysis["top_positive_signals"]:
        bar = "█" * int(contribution * 20)
        print(f"    {signal:<22} +{contribution:.4f}  {bar}")

    print("\n  📉 Negative Signals:")
    for signal, contribution in analysis["top_negative_signals"]:
        bar = "▓" * int(abs(contribution) * 20)
        print(f"    {signal:<22} {contribution:.4f}  {bar}")

    if analysis["recommendations"]:
        print("\n  💡 Recommendations:")
        for rec in analysis["recommendations"]:
            print(f"    • {rec}")

    print("\n" + "=" * 60)
    print("  Signal names & formula: from source (weighted_scorer.rs)")
    print("  Weight values: estimated (params module not released)")
    print("  Fork: github.com/jmg421/x-algorithm")
    print("=" * 60 + "\n")


def main():
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        print("Phoenix Score Predictor — xAI x-algorithm (May 2026)")
        print("─" * 50)
        print("Enter your post text (Ctrl+D to finish):\n")
        text = sys.stdin.read().strip()

    if not text:
        print("Usage: python phoenix_score_predictor.py \"your tweet text\"")
        print("       echo \"your tweet\" | python phoenix_score_predictor.py")
        sys.exit(1)

    # Detect content features from text
    has_image = "[image]" in text.lower() or "📸" in text or "🖼" in text
    has_video = "[video]" in text.lower() or "🎥" in text
    is_quote = text.startswith("QT:") or "[quote]" in text.lower()
    is_thread = "🧵" in text or "[thread]" in text.lower()
    has_question = "?" in text

    analysis = analyze_post(
        text,
        has_image=has_image,
        has_video=has_video,
        video_duration_s=30 if has_video else 0,
        is_reply=False,
        is_quote=is_quote,
        is_thread=is_thread,
    )

    print_report(analysis)

    # Also output JSON for programmatic use
    if "--json" in sys.argv:
        print(json.dumps(analysis, indent=2))


if __name__ == "__main__":
    main()
