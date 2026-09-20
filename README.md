<<<<<<< HEAD
# AI-Powered Personalized Learning Recommendation System

A recommendation system for corporate training departments. Given an employee's
current skills and a career goal, it recommends a personalized set of Coursera
courses to close the gap, and explains why each one was picked.

## Business scenario

Training departments need customized learning plans for employees. The system
recommends training programs based on skills, professional experience,
education, and career context, aiming to improve course completion and give
learners a personalized path toward their next role.

## Folder structure

```
capstoneProject/
├── data/
│   ├── raw/                     # original Kaggle datasets
│   │   ├── 01_people.csv
│   │   ├── 03_education.csv
│   │   ├── 04_experience.csv
│   │   ├── 05_person_skills.csv
│   │   └── coursera_data.csv
│   ├── cleaned/                 # output of 01_data_cleaning.ipynb
│   │   ├── people_cleaned.csv
│   │   ├── education_cleaned.csv
│   │   ├── experience_cleaned.csv
│   │   ├── person_skills_cleaned.csv
│   │   └── courses_cleaned.csv
│   └── recommendation_ready/    # output of 02_recommendation_data.ipynb
│       ├── user_profiles.csv
│       └── course_profiles.csv
├── 01_data_cleaning.ipynb
├── 02_recommendation_data.ipynb
├── 03_recommendation_engine.ipynb    # skill-gap + TF-IDF + semantic engine
├── 04_agent_workflow.ipynb           # Ollama tool-calling agent
├── requirements.txt
└── README.md
```

All notebooks assume they're run from the `capstoneProject` root, with
`data/` sitting directly alongside them (no extra wrapping folder). This is
where the paths inside them point from. `skill_vocabulary.csv` is no
longer generated at all — it was dropped. `03_vector_store_indexing.ipynb`,
`chroma_db/`, and the old `recommendation_data/` wrapper folder are also
gone; see "Design decisions" below.

## How the pieces fit together

```
raw CSVs
   |
   v
01_data_cleaning.ipynb          -> cleaned/*.csv
   |
   v
02_recommendation_data.ipynb    -> user_profiles.csv, course_profiles.csv
   |
   v
03_recommendation_engine.ipynb  -> given a person_id (or a plain skills list),
   |                               a target-skills list, and optionally a
   |                               career goal and description, returns a
   |                               ranked, diversified, explainable course list
   v
04_agent_workflow.ipynb         -> a local LLM (via Ollama) that talks to the
                                    learner, works out current_skills,
                                    target_skills, skill weights, career goal,
                                    and a learner description, calls the
                                    engine, and turns the result into a
                                    plain-language learning path
```

### 1. Data cleaning (`01_data_cleaning.ipynb`)

Loads the five raw CSVs (people, education, experience, skills, and the
Coursera course catalog), fixes column names and text encoding, drops orphan
records that don't join back to a real person, dedupes courses, and writes
five cleaned CSVs.

### 2. Feature engineering (`02_recommendation_data.ipynb`)

Builds the two files the rest of the system runs on:

- **user_profiles.csv** — one row per person, with their skills, experience,
  education, a combined `profile_text`, and `career_context` (their job
  titles, used as career context rather than a confirmed goal, since the
  dataset has no explicit career-goal field).
- **course_profiles.csv** — one row per course, with normalized skills,
  `skill_text`, `content_text`, and cleaned metadata (ratings, difficulty,
  duration).

### 3. Recommendation engine (`03_recommendation_engine.ipynb`)

Given a learner's current skills and a target-skills list (plus, optionally,
a career goal and a short description of what they're trying to achieve),
it:

1. Canonicalizes every skill through a `skill_aliases` mapping first (so
   "ML", "Machine Learning", and "Machine Learning Algorithms" all collapse
   to one skill before anything is scored).
2. Computes the skill gap (target minus current).
3. Scores every course on three signals: weighted skill-gap coverage,
   TF-IDF similarity, and (if `sentence-transformers` is installed) semantic
   similarity via `BAAI/bge-base-en-v1.5`, built from a richer semantic query
   than bare skill keywords — career goal, skill gap, and description
   together.
4. Filters out courses with no evidence on any signal.
5. Combines the three into one hybrid score (40% coverage, 45% semantic,
   15% TF-IDF by default; if semantic similarity isn't available, its share
   is redistributed across the other two automatically).
6. Applies a small ratings-based adjustment (missing ratings are treated as
   unknown, not zero).
7. Runs a diversity filter so the final list isn't five near-identical
   courses.
8. Returns a ranked result with `rank` and per-signal contribution, so the
   agent can explain *why* each course was picked.

`skill_weights` is optional — leave it out and every gap skill is weighted
equally. The agent is what's meant to supply real weights, based on the
learner's stated goal and timeframe.

### 4. Agent (`04_agent_workflow.ipynb`)

A tool-calling agent, not a fixed pipeline, running against a local Ollama
model. It has three tools:

- `lookup_existing_user` — for a learner already in `user_profiles.csv`.
- `save_new_user_profile` — for a learner who isn't, built from the
  conversation instead.
- `get_recommendations` — calls the engine from step 3, including the
  optional career goal and description.

The agent decides target skills, skill weights, career goal, and a short
learner description itself, calls the engine, and turns the result into a
plain-language learning path — giving a complete first recommendation using
reasonable assumptions rather than interrogating the learner, and only
asking a clarifying question when a brand-new user hasn't given enough to
even build a starting profile.

It also includes two safety nets for running on smaller local models:

- **Fake tool call detection** — some models occasionally type out
  something that *looks* like a tool call as plain text instead of using
  the real mechanism, then invent a result to go with it. This is detected
  and the real tool is run instead.
- **Grounding check on the final answer** — a real test run against Ollama
  showed the model producing plausible-sounding but entirely invented
  course titles and placeholder URLs, despite being told not to. After the
  model's final reply, the agent checks whether it actually mentions the
  real titles from its last `get_recommendations` result; if not, the reply
  is thrown away and rendered directly from the tool's real JSON output
  instead.

## What's done and what's left

Done: data cleaning, user and course profiling, a skill-gap-based
recommendation engine with skill canonicalization, semantic matching,
metadata adjustment, and diversity filtering, and a working agent that has
been run end to end against a real local Ollama model (with a grounding
safety net added after that run surfaced a real hallucination issue).

Left to do: the Streamlit dashboard (learner view and L&D admin view) and
the analytics report.

## Design decisions worth knowing

- **No ChromaDB.** The catalog is 404 courses, small enough that computing
  similarity against all of them directly takes a fraction of a second. A
  vector index earns its cost at a much larger scale than this.
- **No skill_vocabulary.csv.** It was generated early on for reference but
  never used by the engine, so it's been removed from the pipeline.
- **Ollama only, no paid API.** The agent calls a local model through
  Ollama's `/api/chat` endpoint, so there's no API key and no per-call cost.
  In exchange, smaller local models need the extra grounding safeguards
  described above — worth keeping in mind if you swap in a different model.

## Setup

1. Create a virtual environment and install dependencies:

   ```
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. (Optional, for semantic similarity) install `sentence-transformers`. If
   it's not installed, the engine automatically skips that signal and
   redistributes its weight across the other two — nothing breaks.

3. Install and start Ollama, and pull a tool-calling capable model:

   ```
   # install: see https://ollama.com
   ollama pull llama3.1
   ollama serve            # often already running as a background service
   ```

   Check it's up with `curl http://localhost:11434/api/tags`.

## Running the project

Run the notebooks in this order, from the `capstoneProject` root (with `data/` sitting directly inside it):

1. `01_data_cleaning.ipynb`
2. `02_recommendation_data.ipynb`
3. `03_recommendation_engine.ipynb` — try it directly with a `person_id` and
   a target-skills list to see ranked recommendations.
4. `04_agent_workflow.ipynb` — with Ollama running, use the interactive
   `chat()` cell at the end, or the two scripted scenarios earlier in the
   notebook, to talk to a real learner conversation instead of a demo.
=======
# capston-project-9-
>>>>>>> 1bdb5c600706831e555c00ac81197c6cd489f8e8
