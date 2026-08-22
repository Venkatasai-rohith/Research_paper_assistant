from schemas import SummaryType

SUMMARY_INSTRUCTIONS = {
    SummaryType.BRIEF: (
        "Generate a concise summary of the research paper in 1-2 paragraphs. "
        "Focus only on the main objective, methodology, and key findings."
    ),

    SummaryType.DETAILED: (
        "Generate a detailed summary of the research paper covering "
        "the objective, problem statement, methodology, experiments, "
        "results, limitations, and future work."
    ),

    SummaryType.STUDENT: (
        "Explain the paper in simple language suitable for a college student. "
        "Avoid unnecessary jargon and include examples where appropriate."
    ),

    SummaryType.TECHNICAL: (
        "Generate a technically detailed summary preserving algorithms, "
        "architectures, mathematical concepts, and implementation details."
    ),

    SummaryType.BULLET: (
        "Summarize the paper as concise bullet points highlighting the "
        "most important information."
    ),
}