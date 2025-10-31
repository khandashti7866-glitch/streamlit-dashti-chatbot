# app.py
import os
import time
import random
import html
from typing import List, Dict

import streamlit as st
from openai import OpenAI  # required by your spec; only used if API key is present

# -------------------------
# Page config & styling
# -------------------------
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto",
)

# Custom CSS for a modern chat look (high-DPI friendly)
st.markdown(
    """
    <style>
    /* Background & card */
    .stApp {
        background: linear-gradient(180deg, #f7fbff 0%, #ffffff 100%);
        color: #0f172a;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    /* Chat area */
    .chat-container {
        max-width: 1600px;
        margin: 0 auto;
        padding: 18px;
    }
    /* Bubble styling */
    .user-bubble {
        background: linear-gradient(90deg, #0ea5e9, #6366f1);
        color: white;
        padding: 14px;
        border-radius: 16px;
        display: inline-block;
        max-width: 78%;
        word-wrap: break-word;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }
    .ai-bubble {
        background: #f1f5f9;
        color: #0f172a;
        padding: 14px;
        border-radius: 16px;
        display: inline-block;
        max-width: 80%;
        word-wrap: break-word;
        box-shadow: 0 6px 18px rgba(2,6,23,0.06);
    }
    .meta {
        font-size: 12px;
        color: #64748b;
        margin-bottom: 6px;
    }
    .chat-row { margin: 12px 4px; }
    /* Make text area and controls more visible in 4k */
    .stTextInput>div>div>input { height: 44px; font-size: 16px; padding: 12px; }
    .stButton>button { padding: 10px 14px; font-size: 15px; }
    /* Code block tweaks */
    .stMarkdown pre { background: #0f172a; color: #e6eef8; padding: 12px; border-radius: 8px; overflow: auto; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Utilities & Local AI simulator
# -------------------------
def init_state():
    """Initialize session state for messages and UI settings."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "system", "content": "You are an intelligent, helpful AI assistant. Keep answers friendly, concise, and clear."}
        ]
    if "model_style" not in st.session_state:
        st.session_state["model_style"] = "General"

def append_message(role: str, content: str):
    st.session_state["messages"].append({"role": role, "content": content})

def format_timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def local_simulated_response(user_text: str, history: List[Dict], style: str) -> str:
    """
    Rule-based, context-aware fallback generator.
    It uses heuristics to produce helpful outputs when no OpenAI key is available.
    """
    user_text_l = user_text.lower().strip()
    # Look for code requests
    if any(word in user_text_l for word in ["code", "implement", "function", "script", "program", "example"]):
        lang = "python"
        if "javascript" in user_text_l or "js " in user_text_l:
            lang = "javascript"
        code_examples = {
            "python": (
                "```python\n"
                "def greet(name: str) -> str:\n"
                "    \"\"\"Return a greeting for name.\"\"\"\n"
                "    return f\"Hello, {name}! How can I help you today?\"\n\n"
                "if __name__ == '__main__':\n"
                "    print(greet('World'))\n"
                "```"
            ),
            "javascript": (
                "```javascript\n"
                "function greet(name) {\n"
                "  return `Hello, ${name}! How can I help you today?`;\n"
                "}\n\n"
                "console.log(greet('World'));\n"
                "```"
            ),
        }
        intro = "Here's a compact example to get you started:\n\n"
        return intro + code_examples[lang]

    # Explanation request
    if any(word in user_text_l for word in ["explain", "what is", "why", "how does", "describe"]):
        # Attempt a short friendly explanation and a one-line summary
        explanation = ("Sure — here's a clear, friendly explanation:\n\n"
                       + summarize_text(user_text, history))
        return explanation

    # Summarize or simplify
    if any(word in user_text_l for word in ["summarize", "summary", "simplify", "tl;dr"]):
        return summarize_text(user_text, history)

    # Translate
    if user_text_l.startswith("translate ") or "translate to" in user_text_l:
        # naive translate simulation: mark it as translated
        return "Translation (simulated):\n\n" + user_text.replace("translate", "").strip()

    # Fallback: context-aware friendly reply
    examples = [
        "That sounds interesting — here's a clear answer: ",
        "Good question. In short: ",
        "Here's how I would approach that: ",
    ]
    suffixes = [
        "If you'd like, I can expand with examples or code.",
        "Want a shorter TL;DR or a longer explanation?",
        "I can also provide step-by-step instructions if you'd like."
    ]

    # build a synthetic answer referencing the user's words
    brief = f"{random.choice(examples)}{user_text.strip()[:120]}..."
    # include a helpful tip based on chosen style
    if style == "Coding":
        brief += "\n\nTip: break large problems into smaller functions and add tests."
    elif style == "Education":
        brief += "\n\nStudy tip: try explaining this idea to someone else in simple terms."
    else:
        brief += "\n\nQuick advice: ask follow-up questions to dig deeper."

    return brief + "\n\n" + random.choice(suffixes)

def summarize_text(user_text: str, history: List[Dict]) -> str:
    """Naive summarizer that extracts key sentence and makes a short explanation."""
    # If the user provided an explicit 'summarize X' instruction — summarize that part
    text = user_text
    parts = text.split(".")
    if len(parts) > 0:
        core = parts[0].strip()
    else:
        core = text.strip()
    if len(core) < 6 and len(history) > 1:
        # summarize the last user message if the user gave a terse command
        last_user = next((m for m in reversed(history) if m["role"] == "user"), None)
        core = last_user["content"] if last_user else core
    # Create a small summary:
    summary = f"{core}\n\nIn short: {core[:200]} — the main point is to focus on the essential ideas and practical next steps."
    return summary

# -------------------------
# Optional: OpenAI client wrapper (only used if API key present)
# -------------------------
def openai_response(user_text: str, history: List[Dict], model_style: str) -> str:
    """
    Try to use the OpenAI client if an API key is present.
    If anything fails or no key, raise an exception to fallback.
    """
    # Look for API key
    api_key = os.environ.get("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None) if "secrets" in dir(st) else None
    if not api_key:
        raise RuntimeError("No OpenAI API key available.")

    client = OpenAI(api_key=api_key)
    # Build conversation in simple format
    messages = []
    for m in history:
        if m["role"] == "system":
            messages.append({"role": "system", "content": m["content"]})
        elif m["role"] == "user":
            messages.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            messages.append({"role": "assistant", "content": m["content"]})
    # append current user
    messages.append({"role": "user", "content": user_text})

    # Choose an appropriate model mapping (this is conservative)
    model_name = "gpt-4o-mini" if model_style == "General" else "gpt-4o-mini"  # placeholder
    # Call OpenAI (synchronous)
    resp = client.chat.create(model=model_name, messages=messages, temperature=0.2)
    # The response structure may vary; try to extract content
    try:
        return resp.choices[0].message["content"]
    except Exception:
        # some clients return differently
        return str(resp)

# -------------------------
# Main UI
# -------------------------
def main():
    init_state()

    # Sidebar
    with st.sidebar:
        st.markdown("## 🤖 AI Chat Assistant")
        st.markdown("A local ChatGPT-like chat interface built with Streamlit. Works without an API key using a simulated AI. Set `OPENAI_API_KEY` to use the OpenAI API.")
        st.write("---")
        model_choice = st.selectbox("Model style", ["General", "Coding", "Education"], index=["General", "Coding", "Education"].index(st.session_state["model_style"]))
        st.session_state["model_style"] = model_choice
        st.write("")
        if st.button("Clear chat"):
            st.session_state["messages"] = [{"role": "system", "content": "You are an intelligent, helpful AI assistant. Keep answers friendly, concise, and clear."}]
            st.experimental_rerun()
        st.write("---")
        st.markdown("**Tips**")
        st.markdown("- Ask coding questions with `code` or `implement` to get examples.\n- Use `translate`, `summarize`, or `explain` in your prompt for specialized behaviors.")
        st.write("")
        st.markdown("Made for local, high-fidelity demos. Adjust window size for best 4K experience.")

    # Main chat area
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    st.title("AI Chat Assistant")
    st.caption("Type commands or natural language prompts — the assistant responds like ChatGPT (simulated if no API key).")

    chat_placeholder = st.container()

    # Display messages
    with chat_placeholder:
        for i, msg in enumerate(st.session_state["messages"]):
            # skip system messages when rendering chat bubbles
            if msg["role"] == "system":
                continue
            is_user = msg["role"] == "user"
            key = f"msg_{i}_{msg['role']}"
            with st.container():
                # meta row
                cols = st.columns([0.02, 0.98]) if is_user else st.columns([0.98, 0.02])
                if is_user:
                    with cols[1]:
                        st.markdown(f'<div class="meta">You • {format_timestamp()}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="chat-row"><div class="user-bubble">{html.escape(msg["content"]).replace("\\n","<br/>")}</div></div>', unsafe_allow_html=True)
                else:
                    with cols[0]:
                        st.markdown(f'<div class="meta">AI • {format_timestamp()}</div>', unsafe_allow_html=True)
                        # allow markdown rendering for ai content (code blocks, lists)
                        st.markdown(f'<div class="chat-row"><div class="ai-bubble">{msg["content"].replace("\\n","<br/>")}</div></div>', unsafe_allow_html=True)

    # Input area (at bottom) using st.chat_input
    user_input = st.chat_input("Type your command, question, or description here... (e.g., 'Explain black holes in simple terms')")
    if user_input:
        # append user message and display immediately
        append_message("user", user_input)
        # Optimistic display of user message (force re-render)
        st.experimental_rerun()

    # If the last message is a user message and there's no assistant reply yet, generate one
    if len(st.session_state["messages"]) >= 2 and st.session_state["messages"][-1]["role"] == "user":
        last_user = st.session_state["messages"][-1]["content"]
        # only generate if there's no immediate assistant after it
        if not (len(st.session_state["messages"]) >= 2 and st.session_state["messages"][-1]["role"] == "assistant"):
            # show a streaming "thinking" indicator
            with st.spinner("Generating response..."):
                try:
                    # try OpenAI if available
                    try:
                        ai_text = openai_response(last_user, st.session_state["messages"], st.session_state["model_style"])
                    except Exception:
                        ai_text = local_simulated_response(last_user, st.session_state["messages"], st.session_state["model_style"])
                except Exception as e:
                    ai_text = "Sorry — I hit an error while generating a reply. " + str(e)

            append_message("assistant", ai_text)
            st.experimental_rerun()

    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
