#!/usr/bin/env python3
"""
App for resume / job description skills matching via LangGraph (Gemini or OpenAI).
Usage: python main.py <resume_path> <job_description_path> [options]
       python main.py --print-graph   (print workflow graph and exit)
Default: comparison in result.json, judge feedback in feedback.json (with --judge).
"""
import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from graph import graph

from utils import _strip_markdown_json

load_dotenv()

def main():
    parser = argparse.ArgumentParser(
        description="Compare resume skills with the job description (LangGraph + Gemini/OpenAI)."
    )
    parser.add_argument(
        "resume",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the resume file (text or content to analyze)",
    )
    parser.add_argument(
        "job_description",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the job description file",
    )
    parser.add_argument(
        "-p", "--provider",
        choices=["gemini", "openai"],
        default="openai",
        help="LLM provider to use (default: gemini)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="File for the comparison result (default: result.json)",
    )
    parser.add_argument(
        "-f", "--feedback",
        type=Path,
        default=None,
        help="File for judge feedback, used only with --judge (default: feedback.json)",
    )
    parser.add_argument(
        "-j", "--judge",
        action="store_true",
        help="Also run the judge step on the matching result",
    )
    parser.add_argument(
        "--print-graph",
        action="store_true",
        help="Render workflow graph (PNG or ASCII) and exit without running",
    )
    args = parser.parse_args()

    # Print graph only, no run
    if args.print_graph:
        drawable = graph.get_graph()
        graph_output = (args.output if args.output is not None else Path("graph.png")).resolve()
        try:
            drawable.draw_mermaid_png(output_file_path=str(graph_output))
            print(f"Graph saved to: {graph_output}")
        except Exception as e:
            print(f"Mermaid PNG failed ({e})")
        sys.exit(0)

    if args.resume is None or args.job_description is None:
        parser.error("resume and job_description are required when not using --print-graph")

    # Check that the chosen provider's API key is set
    if args.provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            print("Error: provider 'openai' requires OPENAI_API_KEY. Set it in .env or in environment variables.", file=sys.stderr)
            sys.exit(1)
    else:
        if not os.getenv("GEMINI_API_KEY"):
            print("Error: provider 'gemini' requires GEMINI_API_KEY. Set it in .env or in environment variables.", file=sys.stderr)
            sys.exit(1)

    resume_path = args.resume
    jd_path = args.job_description

    if not resume_path.exists():
        print(f"File not found: {resume_path}", file=sys.stderr)
        sys.exit(1)
    if not jd_path.exists():
        print(f"File not found: {jd_path}", file=sys.stderr)
        sys.exit(1)

    resume_doc = resume_path.read_text(encoding="utf-8", errors="replace")
    job_description_doc = jd_path.read_text(encoding="utf-8", errors="replace")

    initial_state = {
        "provider": args.provider,
        "run_judge": args.judge,
        "resume_doc": resume_doc,
        "job_description_doc": job_description_doc,
        "extracted_skills_json": "",
        "validated_extracted_skills_json": "",
        "resume_classified_skills_json": "",
        "JD_classified_skills_json": "",
        "comparison_result_json": "",
        "judge_feedback_json": "",
    }

    result = graph.invoke(initial_state)
    comparison = result.get("comparison_result_json") or ""
    comparison = _strip_markdown_json(comparison)
    judge_feedback = result.get("judge_feedback_json") or ""
    judge_feedback = _strip_markdown_json(judge_feedback)

    # Default: comparison in result.json, judge feedback in feedback.json (when --judge)
    comparison_file = args.output if args.output is not None else Path("result.json")
    feedback_file = args.feedback if args.feedback is not None else Path("feedback.json")

    comparison_file.write_text(comparison, encoding="utf-8")
    print(f"Comparison saved to: {comparison_file}")

    if args.judge and judge_feedback:
        feedback_file.write_text(judge_feedback, encoding="utf-8")
        print(f"Judge feedback saved to: {feedback_file}")


if __name__ == "__main__":
    main()
