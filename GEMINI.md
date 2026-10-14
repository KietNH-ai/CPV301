# Project Rules and Guidelines

## Token Budget & Checkpointing Rule
> [!IMPORTANT]
> **Token Threshold Rule (<20% Remaining Tokens)**:
> Before running out of tokens (specifically when remaining context window is under 20%), the LLM/Agent **MUST** create a comprehensive checkpoint file and save it in the `checkpoint_for_llm` folder at the root of the project.

### Checkpoint Creation Requirements:
1. **Target Directory**: `checkpoint_for_llm/`
2. **File Naming Convention**: `checkpoint_YYYYMMDD_HHMMSS.md` or `checkpoint_<task_name>.md`
3. **Mandatory Checkpoint Contents**:
   - **Summary of Completed Work**: Key findings, scripts run, and issues solved.
   - **Current Task & Progress**: Exactly what was in progress when the checkpoint was created.
   - **Modified & Created Files**: List of all files changed or added with relative paths.
   - **Next Action Steps**: Numbered, step-by-step instructions for the succeeding session or agent to resume without context loss.
   - **Known Issues & Blockers**: Any unresolved bugs, environment constraints, or pending decisions.
