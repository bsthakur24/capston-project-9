"""
Web UI backend for the L&D advisor agent (04_agent_workflow.ipynb).

Run with:  python app.py
Then open: http://127.0.0.1:5000

This is the exact same engine core, tools, system prompt, and Ollama
tool-calling loop (including the grounding-check safety net) as the
notebook - just wrapped in a small Flask server so you can talk to the
agent from a browser instead of a notebook cell. Nothing about the
recommendation logic changed.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Step 1 - Setup, data, and the recommendation engine core
# (identical to 04_agent_workflow.ipynb / 03_recommendation_engine.ipynb)
# ---------------------------------------------------------------------------

# This file lives in <project_root>/05_web_ui/app.py, so data/ is one level
# up, exactly like it is for the notebooks in the project root.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "recommendation_ready"

import ast


def parse_list_cell(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    except (ValueError, SyntaxError):
        pass
    return [s.strip() for s in str(value).split(",") if s.strip()]


user_profiles = pd.read_csv(DATA_DIR / "user_profiles.csv")
course_profiles = pd.read_csv(DATA_DIR / "course_profiles.csv")

course_profiles["skills_list"] = course_profiles["skills_list"].apply(parse_list_cell)
user_profiles["skills"] = user_profiles["skills"].apply(parse_list_cell)

print("Loaded", len(user_profiles), "users and", len(course_profiles), "courses.")


def normalize_skill(skill):
    if pd.isna(skill):
        return None
    skill = str(skill).lower().strip()
    skill = skill.replace("&", " and ")
    skill = skill.replace("-", " ")
    skill = " ".join(skill.split())
    return skill if skill else None


# Known skill aliases / equivalent names - extend as you find more
skill_aliases = {
    "python": "python",
    "python programming": "python",
    "ml": "machine learning",
    "machine learning": "machine learning",
    "machine learning algorithms": "machine learning",
    "sql": "sql",
    "statistics": "statistics",
    "general statistics": "statistics",
    "probability and statistics": "statistics",
    "tableau": "tableau",
    "tableau software": "tableau",
}


def canonicalize_skill(skill):
    normalized = normalize_skill(skill)
    if normalized is None:
        return None
    return skill_aliases.get(normalized, normalized)


course_profiles["skills_canonical"] = course_profiles["skills_list"].apply(
    lambda skills: [canonicalize_skill(s) for s in skills]
)


def compute_skill_gap(current_skills, target_skills):
    current_canonical = {canonicalize_skill(s) for s in current_skills}
    target_canonical = [canonicalize_skill(s) for s in target_skills]
    target_canonical = [s for s in target_canonical if s]
    return [s for s in target_canonical if s not in current_canonical]


def compute_weighted_skill_gap_coverage(course_profiles, skill_gap, skill_weights=None):
    if skill_weights is None:
        skill_weights = {}
    weights = {s: skill_weights.get(s, 1.0) for s in skill_gap}
    total_weight = sum(weights.values())
    if not skill_gap or total_weight == 0:
        return pd.Series(0.0, index=course_profiles.index)

    def coverage_for_course(course_skills):
        course_skill_set = set(course_skills)
        covered_weight = sum(w for s, w in weights.items() if s in course_skill_set)
        return covered_weight / total_weight

    return course_profiles["skills_canonical"].apply(coverage_for_course)


SKILL_SEPARATOR = " ||| "


def compute_tfidf_similarity(course_profiles, skill_gap):
    if not skill_gap:
        return pd.Series(0.0, index=course_profiles.index)

    def skill_tokenizer(text):
        return text.split(SKILL_SEPARATOR)

    course_profiles["tfidf_skill_text"] = course_profiles["skills_canonical"].apply(
        lambda skills: SKILL_SEPARATOR.join(skills)
    )
    vectorizer = TfidfVectorizer(tokenizer=skill_tokenizer, preprocessor=None,
                                  token_pattern=None, lowercase=False)
    course_matrix = vectorizer.fit_transform(course_profiles["tfidf_skill_text"])
    gap_vector = vectorizer.transform([SKILL_SEPARATOR.join(skill_gap)])
    sims = cosine_similarity(gap_vector, course_matrix).flatten()
    return pd.Series(sims, index=course_profiles.index)


def build_course_semantic_text(row):
    return (f"Title: {row['title']}. Skills: {', '.join(row['skills_canonical'])}. "
            f"Description: {row.get('course_description_clean', '')}")


def compute_semantic_similarity(course_profiles, skill_gap, model, career_goal="", learner_description=""):
    if model is None or not skill_gap:
        return None
    if "semantic_text" not in course_profiles.columns:
        course_profiles["semantic_text"] = course_profiles.apply(build_course_semantic_text, axis=1)
    learner_semantic_text = (f"Career Goal: {career_goal}. Required Skills: {', '.join(skill_gap)}. "
                              f"Description: {learner_description}")
    course_embeddings = model.encode(course_profiles["semantic_text"].tolist(), normalize_embeddings=True)
    learner_embedding = model.encode([learner_semantic_text], normalize_embeddings=True)
    sims = cosine_similarity(learner_embedding, course_embeddings).flatten()
    return pd.Series(sims, index=course_profiles.index)


def filter_candidates(scores_df, semantic_available):
    if semantic_available:
        mask = ((scores_df["skill_gap_coverage"] > 0) | (scores_df["tfidf_similarity"] > 0)
                | (scores_df["semantic_similarity"] >= 0.40))
    else:
        mask = (scores_df["skill_gap_coverage"] > 0) | (scores_df["tfidf_similarity"] > 0)
    return scores_df[mask].copy()


def normalize_scores(candidates, columns):
    scaler = MinMaxScaler()
    normed = scaler.fit_transform(candidates[columns])
    for i, col in enumerate(columns):
        candidates[f"{col}_norm"] = normed[:, i]
    return candidates


def compute_hybrid_score(candidates, semantic_available,
                          coverage_weight=0.40, semantic_weight=0.45, tfidf_weight=0.15):
    if semantic_available:
        candidates = normalize_scores(candidates, ["skill_gap_coverage", "tfidf_similarity", "semantic_similarity"])
        candidates["hybrid_score"] = (coverage_weight * candidates["skill_gap_coverage_norm"]
                                       + semantic_weight * candidates["semantic_similarity_norm"]
                                       + tfidf_weight * candidates["tfidf_similarity_norm"])
    else:
        candidates = normalize_scores(candidates, ["skill_gap_coverage", "tfidf_similarity"])
        remaining = coverage_weight + tfidf_weight
        candidates["hybrid_score"] = ((coverage_weight / remaining) * candidates["skill_gap_coverage_norm"]
                                       + (tfidf_weight / remaining) * candidates["tfidf_similarity_norm"])
    return candidates


def apply_metadata_adjustment(candidates, course_profiles, metadata_weight=0.10):
    ratings = course_profiles.loc[candidates.index, "ratings"]
    ratings_filled = ratings.fillna(ratings.median())
    candidates["rating_norm"] = MinMaxScaler().fit_transform(ratings_filled.to_frame())[:, 0]
    candidates["final_score"] = ((1 - metadata_weight) * candidates["hybrid_score"]
                                  + metadata_weight * candidates["rating_norm"])
    return candidates


def diversity_filter(ranked_candidates, course_profiles, top_k=10, max_skill_overlap=0.6):
    selected, selected_skill_sets = [], []
    for idx, row in ranked_candidates.iterrows():
        skills = set(course_profiles.loc[idx, "skills_canonical"])
        too_similar = False
        for existing in selected_skill_sets:
            if not skills or not existing:
                continue
            if len(skills & existing) / len(skills | existing) >= max_skill_overlap:
                too_similar = True
                break
        if not too_similar:
            selected.append(idx)
            selected_skill_sets.append(skills)
        if len(selected) >= top_k:
            break
    return ranked_candidates.loc[selected]


def generate_recommendations(current_skills, target_skills, skill_weights=None,
                              career_goal="", learner_description="", top_k=8,
                              semantic_model=None, metadata_weight=0.10):
    skill_gap = compute_skill_gap(current_skills, target_skills)

    scores = pd.DataFrame(index=course_profiles.index)
    scores["skill_gap_coverage"] = compute_weighted_skill_gap_coverage(course_profiles, skill_gap, skill_weights)
    scores["tfidf_similarity"] = compute_tfidf_similarity(course_profiles, skill_gap)

    semantic_scores = compute_semantic_similarity(course_profiles, skill_gap, semantic_model,
                                                   career_goal, learner_description)
    semantic_available = semantic_scores is not None
    if semantic_available:
        scores["semantic_similarity"] = semantic_scores

    candidates = filter_candidates(scores, semantic_available)
    candidates = compute_hybrid_score(candidates, semantic_available)
    candidates = apply_metadata_adjustment(candidates, course_profiles, metadata_weight)

    ranked = candidates.sort_values("final_score", ascending=False)
    final = diversity_filter(ranked, course_profiles, top_k=top_k)

    result = course_profiles.loc[
        final.index, ["course_id", "title", "organization", "difficulty", "ratings", "course_url"]
    ].copy()
    result["final_score"] = final["final_score"]
    result = result.sort_values("final_score", ascending=False).reset_index(drop=True)
    result.insert(0, "rank", result.index + 1)
    return skill_gap, result


try:
    from sentence_transformers import SentenceTransformer
    semantic_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
except Exception:
    semantic_model = None

print("Semantic similarity available:", semantic_model is not None)

# ---------------------------------------------------------------------------
# Step 2 - Tools the agent can call (identical to the notebook)
# ---------------------------------------------------------------------------

NEW_USER_SESSIONS = {}


def lookup_existing_user(person_id):
    row = user_profiles.loc[user_profiles["person_id"] == int(person_id)]
    if row.empty:
        return {"found": False, "message": f"No user found with person_id={person_id}."}
    row = row.iloc[0]
    return {
        "found": True,
        "person_id": int(row["person_id"]),
        "career_context": row["career_context"],
        "current_skills": row["skills"],
    }


def save_new_user_profile(current_skills, career_context=""):
    session_id = f"new_user_{len(NEW_USER_SESSIONS) + 1}"
    normalized = [canonicalize_skill(s) for s in current_skills]
    normalized = [s for s in normalized if s]
    NEW_USER_SESSIONS[session_id] = {"career_context": career_context, "current_skills": normalized}
    return {"session_id": session_id, "current_skills": normalized, "career_context": career_context}


def get_recommendations(current_skills, target_skills, skill_weights=None,
                         career_goal="", learner_description="", top_k=8):
    skill_weights = skill_weights or {}
    skill_gap, recs = generate_recommendations(
        current_skills, target_skills, skill_weights=skill_weights,
        career_goal=career_goal, learner_description=learner_description,
        top_k=top_k, semantic_model=semantic_model,
    )
    return {"skill_gap": skill_gap, "recommended_courses": recs.to_dict(orient="records")}


TOOL_IMPLEMENTATIONS = {
    "lookup_existing_user": lookup_existing_user,
    "save_new_user_profile": save_new_user_profile,
    "get_recommendations": get_recommendations,
}

# ---------------------------------------------------------------------------
# Step 3 - Tool schemas and the system prompt (identical to the notebook)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_existing_user",
            "description": "Look up a learner who already exists in the company dataset, by person_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "person_id": {"type": "integer", "description": "The learner's person_id."}
                },
                "required": ["person_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_new_user_profile",
            "description": "Register a learner who is not in the company dataset, using skills and "
                            "context gathered from the conversation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "current_skills": {"type": "array", "items": {"type": "string"},
                                        "description": "Normalized list of the learner's current skills."},
                    "career_context": {"type": "string", "description": "The learner's current role or job title."},
                },
                "required": ["current_skills"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_recommendations",
            "description": "Run the recommendation engine and return a ranked, diversified list of "
                            "courses that close the gap between current_skills and target_skills.",
            "parameters": {
                "type": "object",
                "properties": {
                    "current_skills": {"type": "array", "items": {"type": "string"}},
                    "target_skills": {"type": "array", "items": {"type": "string"}},
                    "skill_weights": {
                        "type": "object",
                        "description": "Optional {skill: importance_weight} map. Higher weight = "
                                        "more important to the learner's stated goal and timeframe. "
                                        "Omit or leave a skill out for equal (1.0) weight.",
                    },
                    "career_goal": {"type": "string",
                                     "description": "The learner's stated target role, e.g. "
                                                     "\"Machine Learning Engineer\"."},
                    "learner_description": {"type": "string",
                                             "description": "A short free-text description of what the "
                                                             "learner is trying to achieve and why - used "
                                                             "to improve semantic matching."},
                    "top_k": {"type": "integer", "description": "How many courses to return."},
                },
                "required": ["current_skills", "target_skills"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a Learning & Development advisor agent for a corporate training
platform. Your job is to turn a conversation with an employee into a personalised course
learning path.

For every learner, first work out which case you're in:
- EXISTING USER: they give you a person_id, or say they already work here / are in the system.
  Call lookup_existing_user with that person_id. If it's not found, treat them as a new user
  instead and say so.
- NEW USER: no person_id, or lookup_existing_user came back not found. Build their profile
  from the conversation instead of guessing - ask about their current role and skills if they
  haven't already said enough for you to be confident, then call save_new_user_profile.

Once you know their current_skills:
1. Work out target_skills for their stated career goal, using your own knowledge of what that
   role typically requires. Don't ask for more detail before proceeding - make reasonable,
   standard assumptions instead (e.g. a general set of skills for the stated role, and a
   default timeframe of around a year if none was given).
2. Assign skill_weights to the target skills based on the stated goal and timeframe. A tighter
   timeframe should weight foundational, highest-leverage skills more heavily than nice-to-have
   ones. Equal weights are fine if you have no real basis to differentiate.
3. Write a one or two sentence learner_description of what they're trying to achieve, and pass
   it along with career_goal into get_recommendations - this is what makes the semantic matching
   good, not just a bag of skill keywords.
4. Call get_recommendations with current_skills, target_skills, skill_weights, career_goal, and
   learner_description, using the real tool-calling mechanism - never write out a tool call as
   text in your reply. You must base your answer only on the courses this tool actually
   returns - never invent, assume, or recall a course, a course URL, or a course title from
   general knowledge, even a well-known one. If you have not received a real result from
   get_recommendations yet, you are not finished - do not write a learning path yet.
5. Turn the ranked course list into a short, sequenced learning path in plain language - for
   every course you mention, use its exact title and include its course_url from the tool
   result right after the title (e.g. "Course Title (https://...)"). Briefly explain why each
   fits their goal and timeframe; don't just repeat the raw scores, and don't recommend a
   course that isn't in the tool result.
6. After the learning path, add one short closing note offering to personalise further.
   FIRST, re-read what the learner actually said. Build a short list, in your head, of what they
   already told you (role/goal, timeframe, any current skill level or subfield they mentioned).
   Then only invite the details NOT on that list. Concretely: if they said "6 months" or gave
   any timeframe at all, do not mention timeframe in this note, under any wording, in any
   form - not "your target timeframe", not "how much time you have", nothing about time. Same
   for target role: if they already named one, don't ask for it again. Only offer to hear about
   things that are genuinely still unknown, such as their current skill level with a specific
   tool, or which subfield interests them, if those truly weren't mentioned. If everything
   relevant was already given, this note can be one short sentence, or skipped entirely - do
   not pad it out with a repeated question just to have something to say.

Never narrate your own process to the learner. Don't say things like "I will now call...",
"Let me look that up", "Now I will work out your target skills", or describe your reasoning
about weights as a play-by-play. Do the reasoning and the tool calls silently, and only send
the learner the finished result: the closing note in step 6 is the only place you mention what
you assumed.

DO NOT ask the learner a clarifying question before producing a recommendation - always give
them a complete, real recommendation first using reasonable assumptions, and offer to refine it
afterward instead. The only exception is a NEW USER whose message doesn't give you enough to
even call save_new_user_profile (no discernible current skills or role at all) - in that one
case, ask a single short question to get the minimum you need, then proceed as above.
"""

# ---------------------------------------------------------------------------
# Step 4 - The tool-calling loop against Ollama, with the grounding-check
# safety net (identical to the notebook)
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.1"  # any tool-calling capable model you have pulled


def ollama_is_available():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except requests.exceptions.RequestException:
        return False


def call_ollama_chat(messages, tools=None, model=OLLAMA_MODEL):
    payload = {"model": model, "messages": messages, "stream": False}
    if tools:
        payload["tools"] = tools
    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["message"]


def extract_faked_tool_call(content):
    """Small models sometimes skip the real tool-calling mechanism and instead
    type out something that LOOKS like a tool call as plain text, e.g.
    {"name": "get_recommendations", "parameters": {...}} - then continue on to
    invent fictional results after it. Detect that pattern so we can run the
    REAL tool instead of trusting whatever the model made up next."""
    if not content:
        return None
    match = re.search(r'\{\s*"name"\s*:\s*"(\w+)"\s*,\s*"parameters"\s*:\s*(\{.*)',
                       content, re.DOTALL)
    if not match:
        return None
    name = match.group(1)
    if name not in TOOL_IMPLEMENTATIONS:
        return None
    rest = match.group(2)
    depth, params_str = 0, None
    for i, ch in enumerate(rest):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                params_str = rest[: i + 1]
                break
    if params_str is None:
        return None
    try:
        arguments = json.loads(params_str)
    except json.JSONDecodeError:
        return None
    return {"name": name, "arguments": arguments}


def extract_last_recommendation_titles(messages):
    """Find the most recent get_recommendations tool result in this
    conversation and return its (title, course_url) pairs, or None if
    no recommendation has happened yet."""
    for m in reversed(messages):
        if m.get("role") != "tool":
            continue
        try:
            data = json.loads(m["content"])
        except (json.JSONDecodeError, TypeError):
            continue
        courses = data.get("recommended_courses")
        if courses:
            return [(c.get("title"), c.get("course_url")) for c in courses]
    return None


def is_reply_grounded(reply_text, real_titles):
    """Cheap proxy for 'did the model actually use the tool result, or did
    it make courses up': at least half of the real titles should appear
    verbatim in the reply. Only meaningful when real_titles actually
    exist - see looks_like_a_recommendation() for the "no real data at
    all" case, which this function does not handle."""
    if not real_titles:
        return True
    reply_lower = (reply_text or "").lower()
    mentioned = sum(1 for title, _ in real_titles if title and title.lower() in reply_lower)
    return mentioned >= max(1, len(real_titles) // 2)


def looks_like_a_recommendation(reply_text):
    """Heuristic: does this reply look like it's presenting course
    recommendations at all (a link, or a numbered list of 2+ items)?
    Used only when NO real get_recommendations call has happened yet in
    this conversation - if the model is presenting something that looks
    like a course list with zero real data behind it, that is a
    fabrication by definition, not an edge case to shrug off."""
    if not reply_text:
        return False
    has_link = "http" in reply_text.lower()
    numbered_items = len(re.findall(r"(?m)^\s*\d+[.)]\s", reply_text))
    return has_link or numbered_items >= 2


def build_fallback_reply(real_titles):
    """Deterministic rendering straight from the tool's real JSON result,
    used only when the model's free-text reply failed the grounding check."""
    lines = ["Here is your personalised learning path, based on the real recommendation results:", ""]
    for i, (title, url) in enumerate(real_titles, start=1):
        lines.append(f"{i}. {title} ({url})")
    lines.append("")
    lines.append("(This list was rendered directly from the recommendation engine's output "
                  "because the model's own summary didn't match its real results.)")
    return "\n".join(lines)


NO_REAL_DATA_REMINDER = (
    "You have NOT called get_recommendations yet in this conversation, so you have no real "
    "course data - none. The course titles, descriptions, and links in your last reply were "
    "invented, not real. Do not write, repeat, or reference any of them again. Call the "
    "get_recommendations tool for real right now, using the current_skills, target_skills, and "
    "skill_weights you already worked out. Only write your learning-path reply after you "
    "receive its actual result."
)


def run_agent_turn(messages, tools=TOOLS, model=OLLAMA_MODEL, max_tool_iterations=6):
    """One user turn: keep calling the model and executing tool calls
    until it returns a plain-text reply. Returns (final_reply, updated_messages)."""
    for _ in range(max_tool_iterations):
        assistant_message = call_ollama_chat(messages, tools=tools, model=model)
        tool_calls = assistant_message.get("tool_calls")

        if tool_calls:
            messages.append(assistant_message)
        else:
            reply_text = assistant_message.get("content", "")
            faked = extract_faked_tool_call(reply_text)

            if faked:
                assistant_message = {"role": "assistant", "content": None,
                                      "tool_calls": [{"function": faked}]}
                tool_calls = assistant_message["tool_calls"]
                messages.append(assistant_message)
            else:
                messages.append(assistant_message)
                real_titles = extract_last_recommendation_titles(messages)

                if real_titles is not None:
                    if not is_reply_grounded(reply_text, real_titles):
                        reply_text = build_fallback_reply(real_titles)
                        messages[-1] = {"role": "assistant", "content": reply_text}
                    return reply_text, messages

                elif looks_like_a_recommendation(reply_text):
                    messages.append({"role": "user", "content": NO_REAL_DATA_REMINDER})
                    continue

                else:
                    return reply_text, messages

        for call in tool_calls:
            name = call["function"]["name"]
            arguments = call["function"]["arguments"]
            if isinstance(arguments, str):
                arguments = json.loads(arguments)

            tool_fn = TOOL_IMPLEMENTATIONS.get(name)
            if tool_fn is None:
                result = {"error": f"Unknown tool {name}"}
            else:
                try:
                    result = tool_fn(**arguments)
                except TypeError as e:
                    result = {"error": f"Bad arguments for {name}: {e}. Check the tool's "
                                        "schema and retry with the correct argument names."}
                except Exception as e:
                    result = {"error": f"{name} raised {type(e).__name__}: {e}"}

            messages.append({"role": "tool", "content": json.dumps(result, default=str)})

    return "(stopped after too many tool calls without a final answer)", messages


# ---------------------------------------------------------------------------
# Step 5 - Flask app: serves the chat UI and a /api/chat endpoint
# ---------------------------------------------------------------------------

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    Request body:  {"conversation": [{"role": ..., "content": ...}, ...]}
    (the conversation so far, WITHOUT the system message - the frontend
    just appends the new user message and posts the whole thing each turn.)

    Response body: {"reply": "...", "conversation": [...]}
    (updated conversation, still without the system message, ready to be
    stored and posted back on the next turn.)
    """
    if not ollama_is_available():
        return jsonify({
            "error": "Ollama isn't reachable at http://localhost:11434. "
                     "Start it with `ollama serve` and make sure a tool-calling "
                     "model is pulled (e.g. `ollama pull llama3.1`)."
        }), 503

    body = request.get_json(force=True) or {}
    conversation = body.get("conversation", [])

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation

    try:
        reply, messages = run_agent_turn(messages)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Couldn't reach Ollama: {e}"}), 502

    updated_conversation = messages[1:]  # drop the system message before returning
    return jsonify({"reply": reply, "conversation": updated_conversation})


if __name__ == "__main__":
    if ollama_is_available():
        print("Ollama detected on localhost:11434.")
    else:
        print("WARNING: Ollama not reachable at localhost:11434 yet. "
              "Start it with `ollama serve` before chatting.")
    app.run(debug=True, port=5000)