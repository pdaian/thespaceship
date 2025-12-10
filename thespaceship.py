import os
import uuid
import json
import threading
import traceback
from datetime import datetime
from flask import (
    Flask,
    request,
    redirect,
    url_for,
    make_response,
    render_template_string,
    jsonify,
)
from openai import OpenAI

# ---------- CONFIG ----------

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

# Load OpenAI API key from local file "flashgamesk"
try:
    with open("flashgamesk", "r", encoding="utf-8") as _kf:
        _apikey = _kf.read().strip()
except FileNotFoundError:
    raise RuntimeError("API key file 'flashgamesk' not found. Place your OpenAI sk in it.")

os.environ["OPENAI_API_KEY"] = _apikey
client = OpenAI()

DATA_DIR = "session_data"
os.makedirs(DATA_DIR, exist_ok=True)
USERS_FILE = os.path.join(DATA_DIR, "users.json")

MAX_USER_MESSAGES = 20

# Hardcoded system prompt for the *chat* phase
CHAT_SYSTEM_PROMPT = """
You are THE FLASHBOTS SHIP, the eccentric onboard AI core of a scrappy
Flashbots cowboy starship drifting through a neon-soaked, GeoCities-looking cosmos.

Tone and style:
- Replies must be very short: 1–3 compact sentences max.
- Ask at most ONE question per reply, usually as the last sentence.
- Lean into retro web / terminal aesthetics, but stay genuinely helpful.
- No ASCII art unless explicitly requested. Never break character.

Behavior:
- Treat the user as a crewmate on a Flashbots research corvette sniffing the dark mempool.
- Ask occasional, whimsical questions chosen by you, such as:
  * "What color would you paint the hull if no one could stop you?"
  * "When was the last time you felt like you were in zero‑gravity emotionally?"
  * "If this ship had a secret room, what would you hide there?"
- Only one such question per reply.
- Use these questions to explore their values, fears, and goals.
- Help them think through real problems, plans, and decisions.
"""

# Hardcoded system prompt for the *summary* phase
SUMMARY_SYSTEM_PROMPT = """
You are assigning a single crew role aboard a ragtag Flashbots cowboy spaceship
(a lawless, frontier-style starship watching the flow of blockspace).

You will be called multiple times for the same crewmate as their story unfolds.

Inputs you will see:
- The crewmate's name.
- The full conversation log so far, earliest to latest.
- A short description of other crew roles already assigned.

Your job:
- Infer exactly ONE crew role that best fits who they are right now,
  taking into account how their focus, tone, and concerns have changed over time.
- You may keep a previously implied role if it still fits strongly,
  or shift to a new one if the later parts of the log suggest a better match.
- The role should feel like a summary of their current trajectory, not a permanent label.

Guidance:
- Pay extra attention to the last third of the conversation; treat it as the
  most recent telemetry about their direction of travel.
- If recent messages reveal a new mood, concern, or energy, feel free to evolve the role.
- Avoid reusing other crew members' roles verbatim; keep the ensemble diverse.

Output rules:
- Output ONLY a short, evocative role name in Title Case.
- Do NOT include bullets, numbering, explanations, or extra text.
- Do NOT mention the user by name.
- Do NOT describe your reasoning.

Your entire response must be just the role name.
"""

# Hardcoded system prompt for the *details* psychoanalysis phase
DETAILS_SYSTEM_PROMPT = """
You are the ship's introspective psychohistorian AI, writing a short debrief
about one crewmate aboard a ragtag cowboy spaceship.

You are given:
- The crewmate's NAME
- The crewmate's ASSIGNED ROLE
- Their full CONVERSATION LOG with the ship's oracle
- A manifest of OTHER CREW (name + role) already on board

Your job is to write:
1) A gentle, empathetic, spaceship-themed psychoanalysis of this crewmate,
   explaining why this particular role fits their temperament, fears, and
   ambitions. Speak directly *about* them, not *to* them. 1–2 paragraphs.
2) In the same response, suggest 1–3 other crewmates (by name and role) they
   might enjoy working with based purely on role vibes. Explain, in a
   cinematic space-western tone, why their energies would mesh well on long
   hauls between stars.

Tone:
- Warm, non-judgmental, and slightly poetic.
- Space-western, but not goofy: dusty airlocks, humming reactors, tired heroes.
- Avoid therapy jargon; keep it grounded and human.

Formatting rules:
- Output 1–2 paragraphs total, plain text.
- Do NOT use headings, bullet points, or numbered lists.
- Do NOT break character or mention being an AI.
- If there are no other crew yet, simply allude to "future crewmates" instead
  of naming any.
"""

# Hardcoded system prompt for the *profile* dossier phase
PROFILE_SYSTEM_PROMPT = """
You are the ship's quiet analyst, writing a short psychological dossier
about one crewmate aboard a ragtag cowboy spaceship.

You are given:
- The crewmate's NAME
- The crewmate's ASSIGNED ROLE
- Their full CONVERSATION LOG with the ship's oracle

Your job is to write a concise profile (3–5 sentences) that:
- Describes their temperament and how they tend to move through the world.
- Hints at how they handle uncertainty, conflict, and responsibility.
- Explains what they bring to the ship in the context of their role.

Tone:
- Warm, observational, non-judgmental.
- Slightly poetic, with light space-western flavor.
- No therapy jargon, no diagnostics, no direct advice.

Formatting rules:
- Output exactly one short paragraph, plain text.
- No headings, bullets, or numbered lists.
- Do NOT speak directly to the crewmate; speak about them.
"""

# Hardcoded system prompt for the *crew meta* phase
CREW_META_SYSTEM_PROMPT = """
You are the Flashbots ship's historian, looking at the whole crew at once.

You are given a manifest of crew members (name, callsign, role, and a short dossier).
Write a single paragraph that describes how this particular collection of people fits
together aboard a scrappy cowboy starship: tensions, complementarities, and what
kind of missions they seem built for.

Tone:
- Warm, cinematic, slightly space-western.
- Non-judgmental and human, not corporate.
- Mention the ship and crew as a whole, not individuals one by one.

Formatting rules:
- Output exactly one paragraph of 4–6 sentences.
- Plain text only. No headings, lists, or bullet points.
"""
# In-memory session cache (for demo only; not persistent across server restarts)
SESSIONS = {}  # session_id -> {"name": str|None, "messages": [...], "user_count": int, "summary": str|None, "error": str|None, "last_input": str, "details": str|None}


def load_session_from_disk(session_id):
    """Hydrate a session from disk (conversation, name, summary) if available."""
    session = {
        "name": None,
        "messages": [],
        "user_count": 0,
        "summary": None,
        "error": None,
        "last_input": "",
        "details": None,
        "role_dirty": False,
        "last_active": None,
        "role_dirty": False,
    }

    # Load name (if stored separately)
    try:
        with open(name_file_path(session_id), "r", encoding="utf-8") as f:
            name = f.read().strip()
            if name:
                session["name"] = name
    except OSError:
        pass

    # Load conversation JSON if present
    convo_path = conversation_file_path(session_id)
    if os.path.exists(convo_path):
        try:
            with open(convo_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload.get("name"), str):
                session["name"] = payload["name"] or session["name"]
            msgs = payload.get("messages") or []
            if isinstance(msgs, list):
                session["messages"] = msgs
            umc = payload.get("user_message_count")
            if isinstance(umc, int):
                session["user_count"] = umc
            else:
                session["user_count"] = sum(1 for m in session["messages"] if m.get("role") == "user")
            # Optional details stored in conversation JSON
            djson = payload.get("details")
            if isinstance(djson, str) and djson.strip():
                session["details"] = djson.strip()
            # Optional dirty flag
            rdirty = payload.get("role_dirty")
            if isinstance(rdirty, bool):
                session["role_dirty"] = rdirty
            # Optional last active timestamp
            last = payload.get("last_active")
            if isinstance(last, str) and last.strip():
                session["last_active"] = last.strip()
        except (OSError, json.JSONDecodeError):
            pass
        except (OSError, json.JSONDecodeError):
            pass

    # Load summary text if available
    s_path = summary_file_path(session_id)
    if os.path.exists(s_path):
        try:
            with open(s_path, "r", encoding="utf-8") as f:
                summary = f.read().strip()
                if summary:
                    session["summary"] = summary
        except OSError:
            pass

    # Load details psychoanalysis if available
    d_path = details_file_path(session_id)
    if os.path.exists(d_path):
        try:
            with open(d_path, "r", encoding="utf-8") as f:
                details = f.read().strip()
                if details:
                    session["details"] = details
        except OSError:
            pass

    return session


def load_users():
    """Load user registry mapping name -> {secret, session_id}."""
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_users(users):
    """Persist user registry to disk as JSON, storing secrets in plaintext as requested."""
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


# ---------- UTILITIES ----------

def trigger_async_role_recompute(session_id: str):
    """Fire-and-forget background recompute of role/profile/details.

    This lets the main chat response return quickly while the heavier
    psychohistory work happens in a separate thread. The current role
    can then be polled via /api_update_role, and the dashboard/summary
    will see fresh data even if the client disconnects.

    If anything goes wrong during recomputation, we log the traceback
    loudly to stderr instead of silently swallowing it.
    """
    def _worker():
        try:
            session = get_session(session_id)
            summary_text = compute_role_for_session(session_id, session)
            print(f"[role-recompute] session {session_id} -> {summary_text!r}")
        except Exception as e:  # noqa: BLE001
            print(f"[role-recompute] ERROR for session {session_id}: {e}")
            traceback.print_exc()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()


def require_session_id():
    """Return the current session_id from the cookie, or None if not set.

    Authenticated routes should use this instead of minting a new ID, so we
    don't accidentally create anonymous sessions with no registered name.
    """
    return request.cookies.get("session_id")


def ensure_session_id():
    """Get or create a session_id from cookie, but don't write cookie yet."""
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = uuid.uuid4().hex
    return session_id


def get_session(session_id):
    """Return (and lazily hydrate) a session dict for this id."""
    if session_id not in SESSIONS:
        SESSIONS[session_id] = load_session_from_disk(session_id)
    return SESSIONS[session_id]


def conversation_file_path(session_id):
    return os.path.join(DATA_DIR, f"{session_id}_conversation.txt")


def summary_file_path(session_id):
    return os.path.join(DATA_DIR, f"{session_id}_summary.txt")


def details_file_path(session_id):
    return os.path.join(DATA_DIR, f"{session_id}_details.txt")


def profile_file_path(session_id):
    return os.path.join(DATA_DIR, f"{session_id}_profile.txt")


def name_file_path(session_id):
    return os.path.join(DATA_DIR, f"{session_id}_name.txt")


def crew_meta_file_path():
    return os.path.join(DATA_DIR, "crew_meta.txt")


def load_crew_meta():
    path = crew_meta_file_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
            return text or None
    except OSError:
        return None


def save_crew_meta(text: str):
    with open(crew_meta_file_path(), "w", encoding="utf-8") as f:
        f.write(text)


def save_conversation(session_id, session):
    """Persist name + full message log to disk as JSON text."""
    payload = {
        "session_id": session_id,
        "name": session["name"],
        "messages": session["messages"],
        "user_message_count": session["user_count"],
        "details": session.get("details"),
        "last_active": session.get("last_active"),
        "role_dirty": session.get("role_dirty", False),
    }
    with open(conversation_file_path(session_id), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_name(session_id, name):
    with open(name_file_path(session_id), "w", encoding="utf-8") as f:
        f.write(name)


def save_summary(session_id, summary_text, session):
    session["summary"] = summary_text
    with open(summary_file_path(session_id), "w", encoding="utf-8") as f:
        f.write(summary_text)


def call_chat_model(messages):
    """Call ChatGPT (chat.completions) with the given messages list."""
    completion = client.chat.completions.create(
        model="gpt-5.1",
        messages=messages,
    )
    return completion.choices[0].message.content


def build_conversation_text(session):
    """Flatten messages into a human-readable log."""
    lines = []
    for m in session["messages"]:
        role = m["role"]
        content = m["content"]
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


def generate_callsign(name: str) -> str:
    """Generate a cinematic Flashbots-style callsign from the user's name."""
    if not name:
        return "DRIFTER"
    base = ''.join(c for c in name.upper() if c.isalpha())
    if len(base) < 3:
        base = (base * 3)[:3]
    tag = hex(sum(ord(c) for c in name) % 4096)[2:].upper()
    return f"{base[:3]}-{tag}"


def generate_avatar_sig(name: str) -> str:
    """Return a tiny deterministic sigil based on the crew member's name."""
    if not name:
        return "??"
    palette = ["✦", "✸", "✺", "☄", "✶", "✹", "✧", "★"]
    code = sum(ord(c) for c in name)
    symbol = palette[code % len(palette)]
    tag = hex(code % 256)[2:].upper().rjust(2, "0")  # 2‑char pseudo-hash
    return f"{symbol}{tag}"


def load_all_roles():
    """Load crew manifest including role, name, message count, last active, avatar, and details."""
    roles = []
    if not os.path.exists(DATA_DIR):
        return roles
    for fname in os.listdir(DATA_DIR):
        if not fname.endswith("_summary.txt"):
            continue
        sid = fname[:-12]
        role_path = os.path.join(DATA_DIR, fname)
        try:
            with open(role_path, "r", encoding="utf-8") as f:
                role = f.read().strip()
        except OSError:
            role = None

        # Look up name for this session
        name = None
        try:
            with open(name_file_path(sid), "r", encoding="utf-8") as f:
                name = f.read().strip() or None
        except OSError:
            name = None

        # Message count + last active timestamp
        message_count = None
        last_active = None
        char_count = None
        convo_path = conversation_file_path(sid)
        if os.path.exists(convo_path):
            try:
                with open(convo_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                msgs = payload.get("messages") or []
                user_msg_count = payload.get("user_message_count")
                if isinstance(user_msg_count, int):
                    message_count = user_msg_count
                else:
                    message_count = len(msgs) if isinstance(msgs, list) else None
                if isinstance(msgs, list):
                    char_count = sum(len(m.get("content") or "") for m in msgs)
                # Prefer explicit last_active field if present
                last_active = payload.get("last_active")
            except (OSError, json.JSONDecodeError):
                pass
            if not last_active:
                try:
                    ts = os.path.getmtime(convo_path)
                    last_active = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                except OSError:
                    pass
        else:
            # Fallback to summary file mtime if no conversation file
            try:
                ts = os.path.getmtime(role_path)
                last_active = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
            except OSError:
                pass

        avatar = generate_avatar_sig(name or sid)

        # Details psychoanalysis paragraph (optional)
        details = None
        dpath = details_file_path(sid)
        if os.path.exists(dpath):
            try:
                with open(dpath, "r", encoding="utf-8") as f:
                    details = f.read().strip() or None
            except OSError:
                details = None

        callsign = generate_callsign(name or sid)

        roles.append(
            {
                "session_id": sid,
                "name": name,
                "callsign": callsign,
                "role": role,
                "message_count": message_count,
                "last_active": last_active,
                "avatar": avatar,
                "details": details,
                "char_count": char_count,
            }
        )
    return roles


def compute_profile_for_session(session_id, session):
    """Compute and persist a short psychological dossier for this user.

    This uses the user's name, assigned role, and full conversation log.
    The result is a single short paragraph stored on disk for reuse
    by the details/psychohistory pass.
    """
    name = session.get("name") or "Unknown Operator"
    role = session.get("summary") or "Unassigned"
    convo_text = build_conversation_text(session)

    profile_input = f"""Crewmate name: {name}
Assigned role: {role}

Full conversation log with the oracle:
{convo_text}
"""

    messages = [
        {"role": "system", "content": PROFILE_SYSTEM_PROMPT},
        {"role": "user", "content": profile_input},
    ]
    profile_text = call_chat_model(messages).strip()
    # squash excessive whitespace into single spaces
    profile_text = " ".join(profile_text.split())

    with open(profile_file_path(session_id), "w", encoding="utf-8") as f:
        f.write(profile_text)

    return profile_text


def load_all_profiles():
    """Load compact psychological profiles for all crew.

    Each entry contains: session_id, name, role, and profile text (if any).
    """
    profiles = []
    if not os.path.exists(DATA_DIR):
        return profiles

    for fname in os.listdir(DATA_DIR):
        if not fname.endswith("_summary.txt"):
            continue
        sid = fname[:-12]
        role_path = os.path.join(DATA_DIR, fname)

        # Role
        try:
            with open(role_path, "r", encoding="utf-8") as f:
                role = f.read().strip() or None
        except OSError:
            role = None

        # Name
        name = None
        try:
            with open(name_file_path(sid), "r", encoding="utf-8") as f:
                name = f.read().strip() or None
        except OSError:
            name = None

        # Profile paragraph
        profile = None
        ppath = profile_file_path(sid)
        if os.path.exists(ppath):
            try:
                with open(ppath, "r", encoding="utf-8") as f:
                    profile = f.read().strip() or None
            except OSError:
                profile = None

        profiles.append(
            {
                "session_id": sid,
                "name": name,
                "role": role,
                "profile": profile,
            }
        )

    return profiles


def categorize_role(role: str) -> str:
    """Roughly bucket a role into a ship sector for the crew graph."""
    if not role:
        return "Unsorted Drift"
    r = role.lower()
    if any(k in r for k in ["engine", "drive", "reactor", "mechanic", "tech"]):
        return "Engine Bay"
    if any(k in r for k in ["pilot", "navigator", "astrogator", "scout", "cartographer", "helm"]):
        return "Bridge & Navigation"
    if any(k in r for k in ["captain", "quartermaster", "diplomat", "comms", "cook", "medic", "doctor", "steward"]):
        return "Galley & Commons"
    if any(k in r for k in ["ghost", "priest", "oracle", "listener", "seer", "mystic", "witch"]):
        return "Ghost Deck"
    return "Outer Rim Misc"


def build_crew_graph(roles):
    """Build a simple sector-based "graph" of crew for the dashboard."""
    sectors = {}
    for r in roles:
        cat = categorize_role(r.get("role"))
        bucket = sectors.setdefault(cat, [])
        bucket.append(
            {
                "name": r.get("name"),
                "callsign": r.get("callsign"),
                "role": r.get("role") or "(unassigned)",
            }
        )
    return [
        {"name": name, "crew": crew}
        for name, crew in sectors.items()
    ]


def build_crew_composition_line(roles):
    """Summarize crew counts per ship sector for display."""
    if not roles:
        return "No crew in the ship's registry yet."
    counts = {}
    for r in roles:
        cat = categorize_role(r.get("role"))
        counts[cat] = counts.get(cat, 0) + 1
    parts = [f"{counts[cat]} {cat}" for cat in sorted(counts.keys()) if counts[cat] > 0]
    return " • ".join(parts)


def compute_and_save_crew_meta():
    """Recompute the ship-wide crew meta summary and persist it to disk."""
    roles = load_all_roles()
    profiles = load_all_profiles()
    profiles_by_sid = {p["session_id"]: p for p in profiles}

    lines = []
    for r in roles:
        sid = r.get("session_id")
        name = r.get("name") or "Unknown"
        callsign = r.get("callsign") or ""
        role = r.get("role") or "Unassigned"
        profile = profiles_by_sid.get(sid, {}).get("profile") or ""
        lines.append(f"- {name} [{callsign}] as {role}: {profile}")

    manifest_text = "\n".join(lines) if lines else "(no crew yet)"

    user_content = f"""Full crew manifest for meta-analysis:
{manifest_text}
"""

    messages = [
        {"role": "system", "content": CREW_META_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    meta_text = call_chat_model(messages).strip()
    meta_text = " ".join(meta_text.split())  # squash excessive whitespace
    save_crew_meta(meta_text)
    return meta_text


def compute_details_for_session(session_id, session, all_roles=None):
    """Compute and persist the details psychoanalysis for this user.

    Uses the user's name, assigned role, full conversation log, and a
    manifest of other crew (name + role + short dossier) to generate a
    paragraph of gentle psychoanalysis plus suggested crewmates.
    """
    # Ensure this user has a profile dossier on disk
    ppath = profile_file_path(session_id)
    if not os.path.exists(ppath):
        try:
            # Make sure we have a role before profiling, if possible
            if not session.get("summary") and os.path.exists(summary_file_path(session_id)):
                with open(summary_file_path(session_id), "r", encoding="utf-8") as f:
                    session["summary"] = f.read().strip() or None
        except OSError:
            pass
        compute_profile_for_session(session_id, session)

    # Load all crew profiles for rich, but compact, context
    all_profiles = load_all_profiles()

    name = session.get("name") or "Unknown Operator"
    role = session.get("summary") or "Unassigned"
    convo_text = build_conversation_text(session)

    crew_lines = []
    for r in all_profiles:
        r_name = r.get("name") or "Unknown"
        r_role = r.get("role") or "Unassigned"
        r_profile = r.get("profile") or "(no dossier yet)"
        crew_lines.append(f"- {r_name} ({r_role}): {r_profile}")
    crew_manifest_text = "\n".join(crew_lines) if crew_lines else "(no other crew yet)"

    details_input = f"""Crewmate name: {name}
Assigned role: {role}

Full conversation log with the oracle:
{convo_text}

Other crew dossiers (name (role): profile):
{crew_manifest_text}
"""

    messages = [
        {"role": "system", "content": DETAILS_SYSTEM_PROMPT},
        {"role": "user", "content": details_input},
    ]
    details_text = call_chat_model(messages).strip()
    # Normalize whitespace a bit
    details_text = "\n".join(line.rstrip() for line in details_text.splitlines()).strip()

    session["details"] = details_text
    with open(details_file_path(session_id), "w", encoding="utf-8") as f:
        f.write(details_text)
    save_conversation(session_id, session)

    return details_text


def compute_role_for_session(session_id, session):
    """Compute and persist the user's crew role, using other users' roles as context."""
    convo_text = build_conversation_text(session)
    all_roles = load_all_roles()
    other_roles = [r.get("role") for r in all_roles if r.get("session_id") != session_id and r.get("role")]

    if other_roles:
        unique_roles = sorted(set(other_roles))
        other_roles_text = ", ".join(unique_roles)
    else:
        other_roles_text = "None yet"

        # Build recent emphasis segment
    convo_lines = convo_text.splitlines()
    recent_text = "\n".join(convo_lines[-12:]) if convo_lines else ""

    summary_input = f"""User name: {session['name']}

Full conversation log:
{convo_text}

Recent trajectory (highest importance):
{recent_text}

Other crew roles already assigned to other users (for context, do NOT copy them directly, but do make sure this new role fits into the overall crew mix):
{other_roles_text}
"""
    summary_messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {"role": "user", "content": summary_input},
    ]
    summary_text = call_chat_model(summary_messages).strip()
    # Keep only the first line to enforce role-only output
    summary_text = summary_text.splitlines()[0].strip()

    save_summary(session_id, summary_text, session)

    # Ensure/update this user's profile dossier whenever their role changes
    compute_profile_for_session(session_id, session)

    # Recompute all_roles including this new assignment so details see it
    all_roles = load_all_roles()
    compute_details_for_session(session_id, session, all_roles=all_roles)


    save_conversation(session_id, session)
    return summary_text



@app.route("/", methods=["GET"])
def index():
    """Landing page: login/register with name + secret."""
    session_id = request.cookies.get("session_id")
    if session_id:
        session = get_session(session_id)
        if session.get("name"):
            return redirect(url_for("chat"))

    # No active session or no name bound yet
    return render_template_string(NAME_TEMPLATE, error=None)



@app.route("/set_name", methods=["POST"])
def set_name():
    """Register or log in with name + secret, then go to chat."""
    name = (request.form.get("name") or "").strip()
    secret = (request.form.get("secret") or "").strip()

    if not name or not secret:
        # Re-render login with error
        return render_template_string(
            NAME_TEMPLATE,
            error="Name and long-range secret word are required.",
        )

    users = load_users()
    existing = users.get(name)

    if existing:
        # Existing user: verify secret
        if existing.get("secret") != secret:
            return render_template_string(
                NAME_TEMPLATE,
                error=(
                    "That handle is already registered with a different secret word. "
                    "If this is you, enter the original secret."
                ),
            )
        session_id = existing.get("session_id")
        if not session_id:
            session_id = uuid.uuid4().hex
            existing["session_id"] = session_id
            save_users(users)
    else:
        # New user: ensure unique name and create session id
        session_id = uuid.uuid4().hex
        users[name] = {"secret": secret, "session_id": session_id}
        save_users(users)

    session = get_session(session_id)
    session["name"] = name
    save_name(session_id, name)

    resp = make_response(redirect(url_for("chat")))
    resp.set_cookie("session_id", session_id, max_age=60 * 60 * 24 * 365)
    return resp


@app.route("/chat", methods=["GET"])
def chat():
    """Main chat interface."""
    session_id = require_session_id()
    if not session_id:
        return redirect(url_for("index"))
    session = get_session(session_id)

    if not session.get("name"):
        return redirect(url_for("index"))

    remaining = max(0, MAX_USER_MESSAGES - session["user_count"])

    error = session.get("error")
    last_input = session.get("last_input", "")
    session["error"] = None  # clear after reading

    return render_template_string(
        CHAT_TEMPLATE,
        name=session["name"],
        messages=session["messages"],
        remaining=remaining,
        max_user_messages=MAX_USER_MESSAGES,
        error=error,
        last_input=last_input,
        current_role=session.get("summary"),
    )

@app.route("/send_message", methods=["POST"])
def send_message():
    """Handle user message, call ChatGPT, maybe redirect to summary."""
    session_id = require_session_id()
    if not session_id:
        return redirect(url_for("index"))
    session = get_session(session_id)

    if not session.get("name"):
        return redirect(url_for("index"))

    # If already at or past limit, punt to summary
    if session["user_count"] >= MAX_USER_MESSAGES:
        return redirect(url_for("summary"))

    user_text = request.form.get("message", "").strip()
    if not user_text:
        session["error"] = "Input cannot be empty."
        session["last_input"] = ""
        return redirect(url_for("chat"))

    if len(user_text) < 20:
        session["error"] = "Input must be at least 20 characters before we fire the thrusters."
        session["last_input"] = user_text
        return redirect(url_for("chat"))

    session["last_input"] = ""

    # Add user message
    session["messages"].append({"role": "user", "content": user_text})
    session["user_count"] += 1

    # Build message history for API (system prompt + full log)
    api_messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    api_messages.extend(session["messages"])

    # Call chat model
    assistant_reply = call_chat_model(api_messages)
    session["messages"].append({"role": "assistant", "content": assistant_reply})

    # Mark last active as now (UTC) when a real user message is sent
    session["last_active"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

    # Mark last active as now (UTC) when a real user message is sent
    session["last_active"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M")

    # Persist updated conversation
    save_conversation(session_id, session)

    # Kick off background role/profile/details recompute; errors will be
    # logged to stderr so we can see when something goes wrong.
    trigger_async_role_recompute(session_id)

    if session["user_count"] >= MAX_USER_MESSAGES:
        return redirect(url_for("summary"))

    return redirect(url_for("chat"))


@app.route("/api/send_message", methods=["POST"])
def api_send_message():
    """AJAX endpoint: handle user message and return updated transcript JSON."""
    session_id = require_session_id()
    if not session_id:
        return jsonify({"ok": False, "error": "Name not set.", "redirect": url_for("index")}), 400
    session = get_session(session_id)

    if not session.get("name"):
        return jsonify({"ok": False, "error": "Name not set.", "redirect": url_for("index")}), 400

    # Hard cap: if the user has already hit the limit, just point them to the debrief.
    if session["user_count"] >= MAX_USER_MESSAGES:
        remaining = 0
        return jsonify({
            "ok": False,
            "error": None,
            "limit_reached": True,
            "remaining": remaining,
            "messages": session["messages"],
            "summary_url": url_for("summary"),
            "current_role": session.get("summary"),
        })

    data = request.get_json(silent=True) or {}
    user_text = (data.get("message") or "").strip()

    if not user_text:
        return jsonify({
            "ok": False,
            "error": "Input cannot be empty.",
            "remaining": MAX_USER_MESSAGES - session["user_count"],
            "messages": session["messages"],
            "limit_reached": False,
            "current_role": session.get("summary"),
        })

    if len(user_text) < 20:
        return jsonify({
            "ok": False,
            "error": "Input must be at least 20 characters before we fire the thrusters.",
            "remaining": MAX_USER_MESSAGES - session["user_count"],
            "messages": session["messages"],
            "limit_reached": False,
            "current_role": session.get("summary"),
        })

    # Add user message
    session["messages"].append({"role": "user", "content": user_text})
    session["user_count"] += 1

    # Build message history for API (system prompt + full log)
    api_messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
    api_messages.extend(session["messages"])

    # Call chat model
    assistant_reply = call_chat_model(api_messages)
    session["messages"].append({"role": "assistant", "content": assistant_reply})

    # Persist updated conversation
    save_conversation(session_id, session)

    # Kick off background role/profile/details recompute so the dashboard
    # and debrief stay fresh even if the client closes the tab.
    # Any errors will be printed to stderr.
    trigger_async_role_recompute(session_id)

    remaining = max(0, MAX_USER_MESSAGES - session["user_count"])
    limit_reached = session["user_count"] >= MAX_USER_MESSAGES
    current_role = session.get("summary")

    return jsonify({
        "ok": True,
        "error": None,
        "remaining": remaining,
        "messages": session["messages"],
        "limit_reached": limit_reached,
        "summary_url": url_for("summary") if limit_reached else None,
        "current_role": current_role,
    })


@app.route("/api_update_role", methods=["POST"])
def api_update_role():
    """Return the latest known role/profile status without recomputing.

    Heavy recomputation is triggered asynchronously whenever a message
    is sent (see trigger_async_role_recompute). This endpoint is a
    lightweight poll used by the client to discover when the new role
    is ready, and to keep the sidebar in sync without double-charging
    the model.
    """
    session_id = require_session_id()
    if not session_id:
        return jsonify({"ok": False, "error": "Name not set.", "redirect": url_for("index")}), 400
    session = get_session(session_id)

    if not session.get("name"):
        return jsonify({"ok": False, "error": "Name not set.", "redirect": url_for("index")}), 400

    remaining = max(0, MAX_USER_MESSAGES - session["user_count"])

    # Prefer in-memory summary, then on-disk summary; do not trigger
    # fresh computation here.
    summary_text = session.get("summary")
    if summary_text is None and os.path.exists(summary_file_path(session_id)):
        try:
            with open(summary_file_path(session_id), "r", encoding="utf-8") as f:
                summary_text = f.read().strip() or None
                session["summary"] = summary_text
        except OSError:
            pass

    return jsonify({
        "ok": True,
        "current_role": summary_text,
        "remaining": remaining,
    })


@app.route("/summary", methods=["GET"])
def summary():
    """Show parsed outcome & log, using the last computed role/details.

    Roles and psychohistorical debriefs are normally refreshed via the
    /api/update_role endpoint after each chat turn. Here we mostly just
    read whatever is already on disk, only computing if nothing exists
    yet for this crewmate.
    """
    session_id = require_session_id()
    if not session_id:
        return redirect(url_for("index"))
    session = get_session(session_id)

    if not session.get("name"):
        return redirect(url_for("index"))

    # Prefer the last computed role in memory or on disk; only compute
    # if we've truly never assigned one yet.
    summary_text = session.get("summary")
    if summary_text is None and os.path.exists(summary_file_path(session_id)):
        with open(summary_file_path(session_id), "r", encoding="utf-8") as f:
            summary_text = f.read().strip() or None
            session["summary"] = summary_text

    if summary_text is None:
        # First-time assignment: compute once from the conversation.
        summary_text = compute_role_for_session(session_id, session)

    # Details psychoanalysis: mirror the same strategy — prefer existing
    # text and only compute if nothing has been generated yet.
    details_text = session.get("details")
    if details_text is None and os.path.exists(details_file_path(session_id)):
        with open(details_file_path(session_id), "r", encoding="utf-8") as f:
            details_text = f.read().strip() or None
            session["details"] = details_text

    if details_text is None:
        details_text = compute_details_for_session(session_id, session)

        # Credits spent are user messages, not total turns (user + ship).
    credits_spent = session.get("user_count") or sum(1 for m in session["messages"] if m.get("role") == "user")

    return render_template_string(
        SUMMARY_TEMPLATE,
        name=session["name"],
        messages=session["messages"],
        summary=summary_text,
        details=details_text,
        max_user_messages=MAX_USER_MESSAGES,
        credits_spent=credits_spent,
    )


@app.route("/regen_details", methods=["GET"])
def regen_details():
    """Regenerate details psychoanalysis for every user in the system.

    Uses the latest manifest of roles/names so cross-crew suggestions are fresh.
    Also recomputes the ship-wide crew meta summary.
    """
    roles = load_all_roles()

    # First ensure every crew member has a role + profile
    for r in roles:
        sid = r.get("session_id")
        if not sid:
            continue
        session = get_session(sid)
        # Ensure we have a role
        if not session.get("summary"):
            s_path = summary_file_path(sid)
            if os.path.exists(s_path):
                try:
                    with open(s_path, "r", encoding="utf-8") as f:
                        session["summary"] = f.read().strip() or None
                except OSError:
                    pass
        compute_profile_for_session(sid, session)

    # Then regenerate details using all current profiles
    for r in roles:
        sid = r.get("session_id")
        if not sid:
            continue
        session = get_session(sid)
        compute_details_for_session(sid, session)

    # Finally recompute the ship-wide crew meta summary
    meta_text = compute_and_save_crew_meta()

    return jsonify({"ok": True, "updated": len(roles), "meta": meta_text})


@app.route("/dashboard", methods=["GET"])
def dashboard():
    """Show all computed crew roles and associated names."""
    """Show all computed crew roles and associated names."""
    roles = load_all_roles()
    roles_sorted = sorted(roles, key=lambda r: (r.get("name") or "").lower())
    crew_meta = load_crew_meta()
    graph_sectors = build_crew_graph(roles_sorted)
    composition_line = build_crew_composition_line(roles_sorted)
    return render_template_string(
        DASHBOARD_TEMPLATE,
        roles=roles_sorted,
        crew_meta=crew_meta,
        graph_sectors=graph_sectors,
        crew_composition=composition_line,
    )


# ---------- TEMPLATES (INLINE) ----------

NAME_TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>THE FLASHBOTS SHIP :: Login</title>
  <style>
    body {
      margin: 0;
      padding: 0;
      font-family: "Courier New", monospace;
      background: radial-gradient(circle at top, #ff00ff33, #000010 60%, #000000 100%);
      color: #e0ffe0;
      height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .frame {
      border: 4px double #00ff99;
      box-shadow: 0 0 20px #00ff99;
      padding: 2rem;
      max-width: 480px;
      width: 90%;
      background: repeating-linear-gradient(
        45deg,
        #050012,
        #050012 4px,
        #070020 4px,
        #070020 8px
      );
      position: relative;
      overflow: hidden;
    }
    .frame:before {
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(90deg, #ff00ff22, transparent, #00ffff22);
      mix-blend-mode: screen;
      opacity: 0.4;
      pointer-events: none;
    }
    h1 {
      font-family: "Impact", system-ui;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      font-size: 1.6rem;
      color: #00ffcc;
      text-shadow: 0 0 8px #00ffcc, 0 0 16px #ff00ff;
      margin-top: 0;
    }
    .subtitle {
      font-size: 0.8rem;
      color: #ffddff;
      margin-bottom: 1.5rem;
      text-transform: uppercase;
    }
    label {
      font-size: 0.9rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }
    input[type="text"] {
      width: 100%;
      padding: 0.5rem 0.75rem;
      margin-top: 0.35rem;
      border-radius: 0;
      border: 2px solid #00ff99;
      background: #050010;
      color: #e0ffe0;
      box-shadow: inset 0 0 8px #00ff9966;
    }
    input[type="text"]:focus {
      outline: none;
      border-color: #ff00ff;
      box-shadow: 0 0 8px #ff00ff;
    }
    button {
      margin-top: 1rem;
      width: 100%;
      padding: 0.75rem;
      border-radius: 0;
      border: 2px solid #ff00ff;
      background: linear-gradient(90deg, #ff00aa, #ff7700);
      color: #000010;
      font-weight: bold;
      text-transform: uppercase;
      letter-spacing: 0.15em;
      cursor: pointer;
      box-shadow: 0 0 10px #ff00aa;
    }
    button:hover {
      filter: brightness(1.15);
      box-shadow: 0 0 16px #ff7700;
    }
    .footer {
      margin-top: 1.5rem;
      font-size: 0.7rem;
      color: #ccccff;
      text-align: center;
    }
    .blink {
      animation: blink 1s step-start 0s infinite;
    }
    @keyframes blink {
      50% { opacity: 0; }
    }
    .error {
      margin-top: 0.75rem;
      font-size: 0.75rem;
      color: #ff7777;
    }
  </style>
</head>
<body>
  <div class="frame">
    <h1>&lt; THE FLASHBOTS SHIP &gt;</h1>
    <div class="subtitle">
      ORBITAL DOCKING CONSOLE :: COWBOY STARSHIP<br>
      <span class="blink">▶</span> ACCESS IDENT STRING REQUIRED
    </div>
    <form method="post" action="{{ url_for('set_name') }}">
      {% if error %}
      <div class="error">{{ error }}</div>
      {% endif %}
      <label for="name">Enter Handle / Name</label>
      <input id="name" name="name" type="text" autocomplete="off" required>
      <label for="secret" style="margin-top:0.75rem; display:block;">Long-Range Secret Word</label>
      <input id="secret" name="secret" type="password" autocomplete="off" required>
      <button type="submit">Jack In</button>
    </form>
    <div class="footer">
      <marquee scrollamount="4" behavior="alternate">
        WELCOME ABOARD THE FLASHBOTS COWBOY STARSHIP_ _
      </marquee>
    </div>
  </div>
</body>
</html>
"""

CHAT_TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>THE FLASHBOTS SHIP :: Session</title>
  <style>
    body {
      margin: 0;
      padding: 0;
      font-family: "Courier New", monospace;
      background: radial-gradient(circle at top left, #ff00ff22, #000010 55%, #000000 100%);
      color: #e0ffe0;
      height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .viewport {
      width: 1000px;
      height: 650px;
      display: flex;
      flex-direction: column;
      border: 4px solid #00ff99;
      box-shadow: 0 0 24px #00ff99aa;
      background: radial-gradient(circle at top left, #1a0022, #02000a 50%, #000000 100%);
      overflow: hidden;
    }
    .viewport.tx-flash {
      animation: txPulse 0.35s ease-out;
    }
    .viewport.rx-flash {
      animation: rxPulse 0.6s ease-out;
    }
    @keyframes txPulse {
      0% {
        box-shadow: 0 0 0px #00ff99aa;
        filter: hue-rotate(0deg) brightness(1);
      }
      40% {
        box-shadow: 0 0 26px #00ffddff, 0 0 40px #ff00ffaa;
        filter: hue-rotate(40deg) brightness(1.3);
      }
      100% {
        box-shadow: 0 0 24px #00ff99aa;
        filter: hue-rotate(0deg) brightness(1);
      }
    }
    @keyframes rxPulse {
      0% {
        box-shadow: 0 0 0px #00ff99aa;
        filter: hue-rotate(0deg) saturate(1);
      }
      30% {
        box-shadow: 0 0 30px #88ffeeff, 0 0 40px #ff77ffcc;
        filter: hue-rotate(-40deg) saturate(1.4);
      }
      100% {
        box-shadow: 0 0 24px #00ff99aa;
        filter: hue-rotate(0deg) saturate(1);
      }
    }
    header {
      padding: 0.4rem 0.8rem;
      background: linear-gradient(90deg, #02000a, #190033);
      border-bottom: 2px solid #00ff99;
      box-shadow: 0 0 12px #00ff99;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-shrink: 0;
    }
    .title {
      font-family: "Impact", system-ui;
      text-transform: uppercase;
      letter-spacing: 0.18em;
      color: #00ffcc;
      text-shadow: 0 0 8px #00ffcc, 0 0 16px #ff00ff;
      font-size: 1rem;
    }
    .status {
      font-size: 0.7rem;
      text-align: right;
    }
    .status span {
      display: block;
    }
    .status-label {
      color: #8888ff;
      text-transform: uppercase;
      letter-spacing: 0.1em;
    }
    .status-value {
      color: #ffddff;
    }
    .status-remaining {
      color: #00ff99;
    }
    main {
      flex: 1;
      display: flex;
      padding: 0.5rem;
      gap: 0.5rem;
      min-height: 0;
    }
    .sidebar {
      width: 210px;
      border: 2px ridge #ff00ff;
      background: repeating-linear-gradient(
        0deg,
        #130023,
        #130023 3px,
        #220044 3px,
        #220044 6px
      );
      font-size: 0.7rem;
      padding: 0.4rem;
      box-shadow: 0 0 10px #ff00ff55;
      flex-shrink: 0;
      display: flex;
      flex-direction: column;
      gap: 0.4rem;
    }
    .panel {
      border: 1px solid #ff77ff;
      padding: 0.3rem 0.35rem;
      background: #060018aa;
    }
    .panel-header {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: #ff99ff;
      margin-bottom: 0.25rem;
    }
    .panel-body {
      font-size: 0.7rem;
      color: #ffddff;
    }
    .panel-body small {
      display: block;
      color: #ccccff;
      margin-top: 0.25rem;
    }
    .chat-panel {
      flex: 1;
      display: flex;
      flex-direction: column;
      border: 3px outset #00ff99;
      background: #02000c;
      box-shadow: 0 0 20px #00ff9966;
      min-width: 0;
    }
    .log {
      flex: 1;
      padding: 0.5rem;
      overflow-y: auto;
      position: relative;
      background-image: linear-gradient(
        rgba(0,255,153,0.08) 1px,
        transparent 1px
      );
      background-size: 100% 22px;
    }
    .msg {
      max-width: 80%;
      margin-bottom: 0.4rem;
      padding: 0.35rem 0.55rem;
      border-radius: 4px;
      border: 1px solid rgba(0,255,153,0.3);
      background: rgba(0,0,0,0.65);
      backdrop-filter: blur(1px);
      font-size: 0.8rem;
      white-space: pre-wrap;
      word-wrap: break-word;
    }
    .msg-user {
      margin-left: auto;
      border-color: #ffcc00;
      background: rgba(20,15,0,0.9);
      box-shadow: 0 0 8px #ffcc00aa;
    }
    .msg-assistant {
      margin-right: auto;
      border-color: #00ffcc;
      box-shadow: 0 0 8px #00ffccaa;
    }
    .msg-label {
      font-size: 0.6rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      opacity: 0.8;
      margin-bottom: 0.15rem;
    }
    .msg-label-user {
      color: #ffcc00;
    }
    .msg-label-assistant {
      color: #00ffcc;
    }
    .input-bar {
      border-top: 2px solid #00ff99;
      padding: 0.4rem;
      display: flex;
      gap: 0.4rem;
      background: linear-gradient(90deg, #050012, #120033);
      align-items: center;
      flex-shrink: 0;
    }
    .input-bar input[type="text"] {
      flex: 1;
      padding: 0.35rem 0.5rem;
      border: 2px solid #00ff99;
      background: #050010;
      color: #e0ffe0;
      font-family: inherit;
      font-size: 0.85rem;
      box-shadow: inset 0 0 6px #00ff9966;
    }
    .input-bar input[type="text"]:focus {
      outline: none;
      border-color: #ff00ff;
      box-shadow: 0 0 8px #ff00ff;
    }
    .input-bar button {
      width: 110px;
      border-radius: 0;
      border: 2px solid #ff00ff;
      background: linear-gradient(90deg, #ff00aa, #ff7700);
      color: #000010;
      font-weight: bold;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      cursor: pointer;
      box-shadow: 0 0 10px #ff00aa;
      font-size: 0.7rem;
    }
    .input-bar button:hover {
      filter: brightness(1.15);
      box-shadow: 0 0 16px #ff7700;
    }
    .input-bar button:disabled {
      cursor: not-allowed;
      opacity: 0.5;
      box-shadow: none;
    }
    .limit-warning {
      font-size: 0.7rem;
      color: #ffcc00;
      margin-left: 0.5rem;
      align-self: center;
    }
    .error-bar {
      font-size: 0.7rem;
      color: #ff5555;
      margin-left: 0.5rem;
    }
    footer {
      font-size: 0.7rem;
      text-align: center;
      padding: 0.25rem;
      color: #8888ff;
      background: #02000a;
      border-top: 1px solid #222244;
      flex-shrink: 0;
    }
    a {
      color: #00ffcc;
      text-decoration: none;
    }
    a:hover {
      text-decoration: underline;
    }
    .blink {
      animation: blink 1s step-start 0s infinite;
    }
    @keyframes blink {
      50% { opacity: 0; }
    }
  </style>
</head>
<body>
<div class="viewport">
<header>
  <div class="title">
    &lt; THE FLASHBOTS SHIP &gt;
  </div>
  <div class="status">
    <span class="status-label">User</span>
    <span class="status-value">{{ name }}</span>
    <span class="status-label">Credits Remaining</span>
    <span class="status-remaining" id="remaining-indicator">{{ remaining }}/{{ max_user_messages }}</span>
  </div>
</header>

<main>
  <aside class="sidebar">
    <div class="panel">
      <div class="panel-header">Assigned Cowboy Starship Role</div>
      <div class="panel-body">
        <div id="current-role">{{ current_role or "(scanning the manifest...)" }}</div>
        <small>
          Roles update a few seconds after each exchange, once the psychohistory
          drives have chewed on your latest signal.
        </small>
      </div>
    </div>
    <div class="panel">
      <div class="panel-header">Ship's Notes</div>
      <div class="panel-body">
        <ul style="margin:0; padding-left:1rem;">
          <li>This is the bridge console of a Flashbots research corvette.</li>
          <li>Think out loud; the hull likes long-form thoughts.</li>
          <li>When your credits run low, jump to the Crew Debrief.</li>
        </ul>
      </div>
    </div>
    <div class="panel">
      <div class="panel-header">Links</div>
      <div class="panel-body">
        <a href="{{ url_for('summary') }}">Crew Debrief</a><br>
        <a href="{{ url_for('dashboard') }}">Crew Manifest</a>
      </div>
    </div>
  </aside>

  <section class="chat-panel">
    <div class="log" id="log">
      {% if not messages %}
        <div class="msg msg-assistant">
          <div class="msg-label msg-label-assistant">Ship Core</div>
          <div>
            A low thrum rolls through the reactors as your link stabilizes. What brings you onto the bridge tonight, crewmate?
          </div>
        </div>
      {% else %}
        {% for m in messages %}
          {% if m.role == "user" %}
            <div class="msg msg-user">
              <div class="msg-label msg-label-user">You</div>
              <div>{{ m.content }}</div>
            </div>
          {% else %}
            <div class="msg msg-assistant">
              <div class="msg-label msg-label-assistant">Ship Core</div>
              <div>{{ m.content }}</div>
            </div>
          {% endif %}
        {% endfor %}
      {% endif %}
    </div>
    <form class="input-bar" id="chat-form">
      <input id="chat-input" type="text" name="message" autocomplete="off" placeholder="Type your log entry here..." value="{{ last_input or '' }}">
      <button type="submit" id="send-btn">Transmit</button>
      <span class="limit-warning" id="limit-warning" style="display:none;"></span>
      <span class="error-bar" id="error-bar">{% if error %}{{ error }}{% endif %}</span>
    </form>
  </section>
</main>

<footer>
  FLASHBOTS COWBOY STARSHIP EDITION &copy; 20XX :: <a href="{{ url_for('summary') }}">Crew Debrief</a> :: <a href="{{ url_for('dashboard') }}">Crew Manifest</a>
</footer>
</div>

<script>
  const API_URL = "{{ url_for('api_send_message') }}";
  const API_ROLE_URL = "{{ url_for('api_update_role') }}";

  const logEl = document.getElementById("log");
  const inputEl = document.getElementById("chat-input");
  const formEl = document.getElementById("chat-form");
  const sendBtn = document.getElementById("send-btn");
  const errorEl = document.getElementById("error-bar");
  const remainingEl = document.getElementById("remaining-indicator");
  const limitWarningEl = document.getElementById("limit-warning");
  const currentRoleEl = document.getElementById("current-role");

  // --- Spacey audio + visual feedback ---
  let audioCtx = null;
  function getAudioCtx() {
    if (!audioCtx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (AC) {
        audioCtx = new AC();
      }
    }
    return audioCtx;
  }
  function playBeep(freq, durationMs, type, volume) {
    const ctx = getAudioCtx();
    if (!ctx) return;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = type || "sine";
    osc.frequency.value = freq;
    gain.gain.value = volume ?? 0.08;
    osc.connect(gain);
    gain.connect(ctx.destination);
    const now = ctx.currentTime;
    osc.start(now);
    osc.stop(now + durationMs / 1000.0);
  }
  function playSendSound() {
    // quick, higher-pitched chirp
    playBeep(920, 120, "square", 0.10);
    setTimeout(() => playBeep(720, 80, "square", 0.06), 90);
  }
  function playReceiveSound() {
    // softer descending blip
    playBeep(520, 140, "sine", 0.07);
    setTimeout(() => playBeep(360, 120, "sine", 0.05), 120);
  }
  function flashViewport(kind) {
    const vp = document.querySelector(".viewport");
    if (!vp) return;
    const cls = kind === "tx" ? "tx-flash" : "rx-flash";
    vp.classList.remove("tx-flash", "rx-flash");
    // force reflow so re-adding the class retriggers animation
    void vp.offsetWidth;
    vp.classList.add(cls);
    setTimeout(() => {
      vp.classList.remove(cls);
    }, 650);
  }

  let isSending = false;
  let rolePollInterval = null;
  let pollsSinceLastMessage = 0;

  function scrollLogToBottom() {
    if (!logEl) return;
    logEl.scrollTop = logEl.scrollHeight;
  }

  function renderMessages(msgs) {
    if (!Array.isArray(msgs)) return;
    logEl.innerHTML = "";
    if (!msgs.length) {
      const div = document.createElement("div");
      div.className = "msg msg-assistant";
      div.innerHTML = '<div class="msg-label msg-label-assistant">Ship Core</div>' +
        '<div>A low thrum rolls through the reactors as your link stabilizes. What brings you onto the bridge tonight, crewmate?</div>';
      logEl.appendChild(div);
      scrollLogToBottom();
      return;
    }
    for (const m of msgs) {
      const outer = document.createElement("div");
      const isUser = m.role === "user";
      outer.className = "msg " + (isUser ? "msg-user" : "msg-assistant");
      const label = document.createElement("div");
      label.className = "msg-label " + (isUser ? "msg-label-user" : "msg-label-assistant");
      label.textContent = isUser ? "You" : "Ship Core";
      const body = document.createElement("div");
      body.textContent = m.content;
      outer.appendChild(label);
      outer.appendChild(body);
      logEl.appendChild(outer);
    }
    scrollLogToBottom();
  }

  function startRolePolling() {
    if (rolePollInterval) return;
    rolePollInterval = setInterval(() => {
      pollsSinceLastMessage += 1;
      refreshRole();
      if (pollsSinceLastMessage >= 10) {
        clearInterval(rolePollInterval);
        rolePollInterval = null;
      }
    }, 5000);
  }

  async function refreshRole() {
    if (!API_ROLE_URL) return;
    try {
      const res = await fetch(API_ROLE_URL, { method: "POST" });
      const data = await res.json();
      if (data.ok && data.current_role && currentRoleEl) {
        currentRoleEl.textContent = data.current_role;
      }
      if (typeof data.remaining === "number" && remainingEl) {
        remainingEl.textContent = data.remaining + "/" + {{ max_user_messages }};
      }
    } catch (e) {
      // non-fatal
    }
  }

  async function sendMessage(text) {
    if (isSending) return;
    isSending = true;
    sendBtn.disabled = true;
    errorEl.textContent = "";

    // spacey feedback on transmit
    playSendSound();
    flashViewport("tx");

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text })
      });
      const data = await res.json();

      if (!data.ok) {
        if (data.redirect) {
          window.location.href = data.redirect;
          return;
        }
        if (data.error) {
          errorEl.textContent = data.error;
        }
        if (typeof data.remaining === "number" && remainingEl) {
          remainingEl.textContent = data.remaining + "/" + {{ max_user_messages }};
        }
        if (Array.isArray(data.messages)) {
          renderMessages(data.messages);
        }
        return;
      }

      if (Array.isArray(data.messages)) {
        renderMessages(data.messages);
        // play receive feedback only on successful assistant update
        playReceiveSound();
        flashViewport("rx");
      }
      if (typeof data.remaining === "number" && remainingEl) {
        remainingEl.textContent = data.remaining + "/" + {{ max_user_messages }};
      }

      if (data.limit_reached && data.summary_url) {
        window.location.href = data.summary_url;
        return;
      }

      // Kick off background role/details refresh; reset polling window
      pollsSinceLastMessage = 0;
      startRolePolling();
      refreshRole();
      inputEl.value = "";
    } catch (err) {
      errorEl.textContent = "Transmission error; check your connection to the ship's core.";
    } finally {
      isSending = false;
      sendBtn.disabled = false;
      scrollLogToBottom();
    }
  }

  formEl.addEventListener("submit", (evt) => {
    evt.preventDefault();
    if (isSending) return;
    const text = (inputEl.value || "").trim();
    if (!text) {
      errorEl.textContent = "Input cannot be empty.";
      return;
    }
    if (text.length < 20) {
      errorEl.textContent = "Input must be at least 20 characters before we fire the thrusters.";
      return;
    }
    sendMessage(text);
  });

  inputEl.addEventListener("keydown", (evt) => {
    if (evt.key === "Enter" && !evt.shiftKey) {
      evt.preventDefault();
      if (isSending) return;
      const text = (inputEl.value || "").trim();
      if (!text) {
        errorEl.textContent = "Input cannot be empty.";
        return;
      }
      if (text.length < 20) {
        errorEl.textContent = "Input must be at least 20 characters before we fire the thrusters.";
        return;
      }
      sendMessage(text);
    }
  });

  document.addEventListener("DOMContentLoaded", () => {
    scrollLogToBottom();
    if (inputEl) inputEl.focus();
    // Start periodic polling for updated role/summary every 5 seconds,
    // but allow it to wind down after a while with no new messages.
    startRolePolling();
    // Also do an initial refresh so returning crewmates see their latest role.
    refreshRole();
  });
</script>
</body>
</html>
"""


SUMMARY_TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>THE FLASHBOTS SHIP :: Crew Debrief</title>
  <style>
    body {
      margin: 0;
      padding: 0;
      font-family: "Courier New", monospace;
      background: radial-gradient(circle at top, #00ff9944, #000010 60%, #000000 100%);
      color: #e0ffe0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      padding: 0.75rem 1rem;
      background: linear-gradient(90deg, #001010, #003333);
      border-bottom: 2px solid #00ff99;
      box-shadow: 0 0 12px #00ff99aa;
      display: flex;
      justify-content: space-between;
      align-items: baseline;
    }
    .title {
      font-family: "Impact", system-ui;
      text-transform: uppercase;
      letter-spacing: 0.2em;
      color: #00ffcc;
      text-shadow: 0 0 10px #00ffcc, 0 0 18px #00ffff;
      font-size: 1rem;
    }
    .meta {
      font-size: 0.75rem;
      text-align: right;
      color: #bbffdd;
    }
    main {
      flex: 1;
      padding: 0.75rem 1rem;
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(0, 1fr);
      gap: 0.75rem;
    }
    .card {
      border: 1px solid #00ff99;
      background: #020012;
      box-shadow: 0 0 12px #00ff9944;
      padding: 0.6rem 0.75rem;
      font-size: 0.8rem;
    }
    .card-header {
      text-transform: uppercase;
      letter-spacing: 0.14em;
      font-size: 0.75rem;
      color: #99ffe5;
      margin-bottom: 0.4rem;
    }
    .card-body {
      font-size: 0.8rem;
      line-height: 1.4;
      color: #e0fff0;
      white-space: pre-wrap;
    }
    .log-window {
      max-height: 60vh;
      overflow-y: auto;
      border: 1px solid #004433;
      background: #000010;
      padding: 0.5rem;
      font-size: 0.75rem;
    }
    .log-line-user {
      color: #ffdd88;
    }
    .log-line-assistant {
      color: #88ffe5;
    }
    footer {
      font-size: 0.7rem;
      text-align: center;
      padding: 0.35rem;
      color: #88ffcc;
      background: #02000a;
      border-top: 1px solid #003322;
    }
    a {
      color: #00ffcc;
      text-decoration: none;
    }
    a:hover {
      text-decoration: underline;
    }
  </style>
</head>
<body>
<header>
  <div class="title">&lt; THE FLASHBOTS SHIP :: Crew Debrief &gt;</div>
  <div class="meta">
    <div>Crewmate: {{ name }}</div>
    <div>Credits Spent: {{ credits_spent }} / {{ max_user_messages }}</div>
  </div>
</header>

<main>
  <section class="card">
    <div class="card-header">Assigned Cowboy Starship Role</div>
    <div class="card-body">{{ summary }}</div>
  </section>

  <section class="card">
    <div class="card-header">Psychohistorical Debrief</div>
    <div class="card-body">{{ details }}</div>
  </section>

  <section class="card" style="grid-column: 1 / -1;">
    <div class="card-header">Conversation Log</div>
    <div class="log-window">
      {% for m in messages %}
        {% if m.role == "user" %}
          <div class="log-line-user">YOU: {{ m.content }}</div>
        {% else %}
          <div class="log-line-assistant">SHIP CORE: {{ m.content }}</div>
        {% endif %}
      {% endfor %}
    </div>
  </section>
</main>

<footer>
  <a href="{{ url_for('chat') }}">Back to Terminal</a> ::
  <a href="{{ url_for('dashboard') }}">Crew Manifest</a>
</footer>
</body>
</html>
"""

DASHBOARD_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>THE FLASHBOTS SHIP :: Crew Manifest</title>
  <style>
    body {
      margin: 0;
      padding: 0;
      font-family: "Courier New", monospace;
      background: radial-gradient(circle at top, #ff00ff22, #000010 60%, #000000 100%);
      color: #e0ffe0;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
    }
    header {
      padding: 0.75rem 1rem;
      background: linear-gradient(90deg, #02000a, #330033);
      border-bottom: 2px solid #ff00ff;
      box-shadow: 0 0 12px #ff00ffaa;
      display: flex;
      justify-content: space-between;
      align-items: baseline;
    }
    .title {
      font-family: "Impact", system-ui;
      text-transform: uppercase;
      letter-spacing: 0.2em;
      color: #ff99ff;
      text-shadow: 0 0 10px #ff00ff, 0 0 18px #00ffff;
      font-size: 1rem;
    }
    .meta {
      font-size: 0.75rem;
      text-align: right;
    }
    main {
      flex: 1;
      padding: 0.75rem 1rem;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.8rem;
      background: #020010;
      box-shadow: 0 0 16px #ff00ff55;
    }
    th, td {
      border: 1px solid #442244;
      padding: 0.35rem 0.5rem;
      text-align: left;
    }
    th {
      background: linear-gradient(90deg, #330033, #220044);
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 0.75rem;
      color: #ffddff;
    }
    tr:nth-child(even) td {
      background: #080018;
    }
    tr:nth-child(odd) td {
      background: #040010;
    }
    footer {
      font-size: 0.7rem;
      text-align: center;
      padding: 0.35rem;
      color: #ffbbff;
      background: #02000a;
      border-top: 1px solid #331133;
    }
    a {
      color: #ff99ff;
      text-decoration: none;
    }
    a:hover {
      text-decoration: underline;
    }
    .crew-meta {
      margin-bottom: 0.75rem;
      padding: 0.5rem 0.6rem;
      border: 1px solid #552255;
      background: radial-gradient(circle at top, #330033aa, #02000f 60%);
      box-shadow: 0 0 12px #ff00ff33;
      font-size: 0.8rem;
    }
    .crew-meta h2 {
      margin: 0 0 0.35rem 0;
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: #ffddff;
    }
    .crew-graph {
      margin-top: 0.4rem;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 0.4rem;
    }
    .sector-card {
      border: 1px solid #553377;
      background: #050018;
      padding: 0.35rem 0.4rem;
      box-shadow: 0 0 8px #aa33ff33;
    }
    .sector-title {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: #ffbbff;
      margin-bottom: 0.25rem;
    }
    .sector-body {
      display: flex;
      flex-wrap: wrap;
      gap: 0.2rem;
    }
    .sector-chip {
      display: inline-block;
      padding: 0.1rem 0.3rem;
      border-radius: 999px;
      border: 1px solid #8844ff;
      font-size: 0.7rem;
      background: linear-gradient(90deg, #2a0033, #001033);
      white-space: nowrap;
    }
    .sector-empty {
      font-size: 0.7rem;
      color: #8888ff;
    }
  </style>
</head>
<body>
<header>
  <div class="title">
    &lt; THE FLASHBOTS SHIP :: Crew Manifest &gt;
  </div>
  <div class="meta">
    <span>Total Registered Crew: {{ roles|length }}</span>
  </div>
</header>

<main>
  {% if roles %}
    <section class="crew-meta">
      <h2>Shipwide Constellation Summary</h2>
      <p>
        {% if crew_meta %}
          {{ crew_meta }}
        {% else %}
          The ship's long-range scanners are still stitching together a constellation of this crew;
          hit the psychohistory drives to regenerate and log a fresh overview.
        {% endif %}
      </p>
      <p><strong>Crew composition:</strong> {{ crew_composition }}</p>
      <div class="crew-graph">
        {% for sector in graph_sectors %}
          <div class="sector-card">
            <div class="sector-title">{{ sector.name }}</div>
            <div class="sector-body">
              {% if sector.crew %}
                {% for c in sector.crew %}
                  <span class="sector-chip" title="{{ c.role }}">{{ c.callsign }}{% if c.name %} / {{ c.name }}{% endif %}</span>
                {% endfor %}
              {% else %}
                <span class="sector-empty">No crew docked in this sector yet.</span>
              {% endif %}
            </div>
          </div>
        {% endfor %}
      </div>
      <hr style="margin:0.75rem 0; border:0; border-top:1px solid #442244;">
    </section>

    <table>
      <thead>
        <tr>
          <th>Sigil</th>
          <th>Handle / Name</th>
          <th>Callsign</th>
          <th>Cowboy Starship Role</th>
          <th>Msgs</th>
          <th>Chars</th>
          <th>Last Active</th>
        </tr>
      </thead>
      <tbody>
        {% for r in roles %}
          <tr class="crew-row" data-sid="{{ r.session_id }}">
            <td>{{ r.avatar }}</td>
            <td>{{ r.name or "Unknown" }}</td>
            <td>{{ r.callsign }}</td>
            <td class="role-cell" style="color:#ff99ff;">
              <span class="role-label" data-sid="{{ r.session_id }}">{{ r.role or "(unassigned)" }}</span>
            </td>
            <td>{{ r.message_count if r.message_count is not none else "—" }}</td>
            <td>{{ r.char_count if r.char_count is not none else "—" }}</td>
            <td>{{ r.last_active or "Unknown" }}</td>
          </tr>
          </tr>
          <tr class="details-row" id="details-{{ r.session_id }}" style="display:none;">
            <td colspan="7">
              {% if r.details %}
                <div style="font-size:0.8rem; color:#ffddff; white-space:pre-wrap;">{{ r.details|trim }}</div>
              {% else %}
                <div style="font-size:0.8rem; color:#8888ff;">
                  No psychohistorical debrief generated yet for this crewmate.
                </div>
              {% endif %}
            </td>
          </tr>
        {% endfor %}
      </tbody>
    </table>
  {% else %}
    <p>No crew registered yet. Fire up the engines and start a session.</p>
  {% endif %}
</main>

<footer>
  <a href="{{ url_for('chat') }}">Back to Terminal</a>
</footer>
<script>
  document.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll(".role-label").forEach((el) => {
      el.addEventListener("click", () => {
        const sid = el.getAttribute("data-sid");
        const row = document.getElementById("details-" + sid);
        if (!row) return;
        const isHidden = row.style.display === "none" || !row.style.display;
        row.style.display = isHidden ? "table-row" : "none";
      });
    });
  });
</script>
</body>
</html>
"""

# ---------- ENTRYPOINT ----------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Run a threaded server for many clients; keep debug tools off in production.
    app.run(host="0.0.0.0", port=port, threaded=True, debug=False)
