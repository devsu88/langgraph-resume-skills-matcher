from typing import TypedDict
from langgraph.graph import StateGraph, START, END

from prompts import (
    skill_fetch_prompt,
    extracted_skill_validator,
    skill_type_classification,
    compare_skills,
    judge_comparison_prompt,
)
from api import call_llm


# ----------- State Definition -----------
class SkillExtractionState(TypedDict):
    provider: str
    run_judge: bool
    resume_doc: str
    job_description_doc: str
    extracted_skills_json: str
    validated_extracted_skills_json: str
    resume_classified_skills_json: str
    JD_classified_skills_json: str
    comparison_result_json: str
    judge_feedback_json: str


# ----------- Node 1: Skill Extraction -----------
def skill_extraction_node(state: SkillExtractionState) -> dict:
    resume = state["resume_doc"]
    extract_prompt = skill_fetch_prompt(resume)
    result_json = call_llm(extract_prompt, provider=state["provider"])
    print('SKILL EXTRACTION COMPLETE')
    return {"extracted_skills_json": result_json}


# ----------- Node 2: Skill Validation -----------
def extracted_skill_validator_node(state: SkillExtractionState) -> dict:
    resume = state["resume_doc"]
    extracted_skills = state["extracted_skills_json"]
    validate_prompt = extracted_skill_validator(resume, extracted_skills)
    result_json = call_llm(validate_prompt, provider=state["provider"])
    print('SKILL VALIDATION COMPLETE')
    return {"validated_extracted_skills_json": result_json}


# ----------- Node 3A: Resume Skill Classification -----------
def classify_resume_skills_node(state: SkillExtractionState) -> dict:
    extracted_skills = state["validated_extracted_skills_json"]
    classify_prompt = skill_type_classification(extracted_skills)
    result_json = call_llm(classify_prompt, provider=state["provider"])
    print('RESUME SKILL CLASSIFICATION COMPLETE')
    return {"resume_classified_skills_json": result_json}


# ----------- Node 3B: JD Skill Classification (parallel path) -----------
def classify_JD_skills_node(state: SkillExtractionState) -> dict:
    JD_text = state["job_description_doc"]
    classify_prompt = skill_type_classification(JD_text)
    result_json = call_llm(classify_prompt, provider=state["provider"])
    print('JD SKILL CLASSIFICATION COMPLETE')
    return {"JD_classified_skills_json": result_json}


# ----------- Barrier Node (Synchronization) -----------
def sync_barrier_node(state: SkillExtractionState) -> dict:
    print('SYNCHRONIZATION BARRIER: Both classifications complete')
    return {}


# ----------- Node 4: Compare Skills -----------
def compare_skills_node(state: SkillExtractionState) -> dict:
    candidate_skills = state["resume_classified_skills_json"]
    jd_skills = state["JD_classified_skills_json"]
    comparison_prompt = compare_skills(candidate_skills, jd_skills)
    result_json = call_llm(comparison_prompt, provider=state["provider"])
    print('SKILL COMPARISON COMPLETE')
    return {"comparison_result_json": result_json}


# ----------- Node 5: Judge (feedback sul risultato) -----------
def judge_node(state: SkillExtractionState) -> dict:
    comparison_result = state["comparison_result_json"]
    judge_prompt = judge_comparison_prompt(comparison_result or "{}")
    feedback_json = call_llm(judge_prompt, provider=state["provider"])
    print('JUDGE FEEDBACK COMPLETE')
    return {"judge_feedback_json": feedback_json}


# ----------- Build the Graph -----------
workflow = StateGraph(SkillExtractionState)

workflow.add_node("extract_skills", skill_extraction_node)
workflow.add_node("validate_skills", extracted_skill_validator_node)
workflow.add_node("classify_resume_skills", classify_resume_skills_node)
workflow.add_node("classify_JD_skills", classify_JD_skills_node)
workflow.add_node("sync_barrier", sync_barrier_node)
workflow.add_node("compare_skills", compare_skills_node)
workflow.add_node("judge", judge_node)

# Resume branch: START -> extract -> validate -> classify -> barrier
workflow.add_edge(START, "extract_skills")
workflow.add_edge("extract_skills", "validate_skills")
workflow.add_edge("validate_skills", "classify_resume_skills")
workflow.add_edge("classify_resume_skills", "sync_barrier")

# JD branch (runs in parallel): START -> classify -> barrier
workflow.add_edge(START, "classify_JD_skills")
workflow.add_edge("classify_JD_skills", "sync_barrier")

# After barrier, proceed to comparison
workflow.add_edge("sync_barrier", "compare_skills")


def _after_compare(state: SkillExtractionState):
    return "judge" if state.get("run_judge") else END


workflow.add_conditional_edges("compare_skills", _after_compare)
workflow.add_edge("judge", END)

graph = workflow.compile()
