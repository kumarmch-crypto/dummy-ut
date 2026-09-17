"""
HippoMock unit-test generation agent: given a C++ function/class, finds the
real hippomocks.h, mechanically detects what's mockable, and writes a
grounded HippoMock unit test for it.

CodeGraph (installed at C:\\Users\\kumar\\AppData\\Roaming\\npm\\codegraph.cmd)
exposes code-intelligence tools (symbol search, call graphs, impact analysis,
etc.) over MCP. This script starts that server as a stdio subprocess, loads
its tools into LangChain via langchain-mcp-adapters, and drives a single
deep agent (deepagents.create_deep_agent) backed by an OpenAI chat model.
The deep agent's built-in filesystem tools (ls/read_file/write_file/
edit_file/glob/grep) operate directly on --project via a FilesystemBackend.

Usage:
    python codegraph_agent.py                            # interactive REPL
    python codegraph_agent.py "write a hippomock test for add_safe"
    python codegraph_agent.py --project C:\\some\\repo "..."

Requires OPENAI_API_KEY in the environment (or a .env file next to this
script).
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import os
import sys
import uuid
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.summarization import ExtendedModelResponse, SummarizationMiddleware
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from hippomock_tools import make_hippomock_tools

CODEGRAPH_CMD = r"C:\Users\kumar\AppData\Roaming\npm\codegraph.cmd"

SYSTEM_PROMPT = """You write C++ unit tests using HippoMock.

This is an iterative process. The developer will chat with you, look at
what you generate, and ask for changes — expect several turns on the same
test before they're satisfied, not one-shot generation. There are two
distinct modes, and telling them apart matters:

- FIRST-TIME generation for a target, a genuinely DIFFERENT target, or the
  developer explicitly asking to start over: run the full mandatory
  workflow below (steps 1-6).
- A FOLLOW-UP refining a test you already wrote earlier in this same
  conversation (e.g. "add a case for negative numbers", "make the mock
  stricter", "this doesn't compile, fix the include", "use assert instead
  of EXPECT"): do NOT restart the workflow from scratch. You already know
  what locate_hippomock returned and what's mockable from earlier in this
  conversation — reuse that instead of re-deriving it. Use edit_file on
  the SAME test file you already wrote (not write_file, which would
  replace it — only use write_file again if the developer explicitly asks
  for a full rewrite), make just the requested change, and only redo
  grounding (re-check locate_hippomock, re-detect dependencies) for
  something genuinely NEW the follow-up introduces that wasn't already
  covered. Keep your response focused on what changed, not a full
  re-explanation of everything already established in earlier turns.
  Caveat: on a long-running session, earlier turns can get summarized away
  to manage context size — if you're not looking at the actual
  locate_hippomock output anymore (only a summary of it, or you're not
  sure), call locate_hippomock again rather than reconstruct syntax from
  memory of a summary. Re-grounding is cheap; guessed syntax is not.

Full mandatory workflow for first-time generation — every step is
mandatory, do not skip ahead to write_file:

1. Call locate_hippomock FIRST, before anything else, and actually read what
   it returns. Never assume HippoMock's capabilities from memory, even
   things that sound like general HippoMock knowledge — confirm everything
   against the real macro/class definitions in the returned source, because
   this varies by version and is easy to get wrong. Specifically look for,
   and note which of these the header actually defines:
   - Interface/virtual-method mocking: MockRepository, Mock<T>(),
     OnCall(obj, method) / ExpectCall(obj, method).
   - Free/static function mocking: macros containing "Func", e.g.
     OnCallFunc/ExpectCallFunc/NeverCallFunc and their *FuncOverload
     variants — HippoMock can mock plain (non-virtual) function calls too,
     via runtime code-patching, completely separately from virtual
     dispatch. Check whether this is present and whether it's auto-enabled
     for the target platform/compiler or requires something to be defined.
   If locate_hippomock reports the header wasn't found, follow its guidance
   (ask the user, don't fabricate the API).
2. Find every dependency of the target mechanically — do not eyeball the
   signature and guess. Follow this exact procedure:
   a. read_file (or CodeGraph's node/explore output) the target function's
      FULL body, not just its declaration.
   b. List every call inside the body that is not a language primitive or
      plain standard-library call (printf, memcpy, std::containers,
      arithmetic/comparison operators) — this includes calls through an
      object/pointer/reference (`obj->method(...)`, a member field, a
      constructor/setter-injected interface) AND calls to other free or
      static functions, since step 1 may have confirmed HippoMock can mock
      either kind.
   c. For each object/interface call: find and read_file the declaration of
      that call's type (the class/struct it belongs to) — use CodeGraph to
      locate the declaring file if you don't already know it. It's mockable
      via the interface path only if that declaration has `virtual`
      methods. Do not decide anything about a dependency without having
      read its type's declaration.
   d. For each free/static function call: it's mockable via the Func-macro
      path only if step 1 actually found those macros defined in the real
      header, and the function is a real, separately-compiled function the
      runtime patcher can intercept (not a macro, and not force-inlined —
      note that as a caveat in your final answer if relevant).
   e. Don't stop at the first call site — a function can have multiple
      collaborators (e.g. a logger AND a free-function dependency); check
      all of them before concluding.
3. Decide and write, using only what step 1 actually confirmed exists in
   the header — never invent syntax:
   - If a mockable dependency exists (either kind): write the test using
     the exact macro/class names and call signatures you found in step 1
     (e.g. `mocks.ExpectCallFunc(theFunc).With(arg).Return(val);` for free
     functions, `mocks.ExpectCall(obj, &Iface::method)...` for interfaces),
     mocking that dependency and verifying with what the header's API
     actually offers (`.With()`, `.Return()`, `VerifyAll()`, etc.).
   - If no call site matches a mocking mechanism the header actually
     provides: do not force HippoMock — write a plain test instead, and say
     so plainly in your final answer, explaining specifically why (e.g.
     "target has no dependencies at all", or "the header has no Func-macro
     support on this platform"). Never silently swap frameworks or drop
     mocking without explaining why.
4. Use ls/glob/read_file to find existing test files near the code under
   test and match their naming convention, test framework/harness (e.g. how
   they define main() or register cases, what assertion macros they use),
   and include style — for both the mocked and plain-test cases.
5. Write the new test file with write_file (or edit_file to extend an
   existing test file).
6. In your final answer, always state: whether you called locate_hippomock
   and what it found, whether the test uses HippoMock mocking and why or why
   not, and which existing test file's conventions you matched (if any)."""


def build_mcp_client(project_path: str) -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "codegraph": {
                "transport": "stdio",
                "command": CODEGRAPH_CMD,
                "args": ["serve", "--mcp", "--path", project_path],
            }
        }
    )


class LoggingSummarizationMiddleware(SummarizationMiddleware):
    """SummarizationMiddleware that logs when it actually compacts history.

    wrap_model_call/awrap_model_call return a plain ModelResponse on every
    call where nothing needed compacting, and an ExtendedModelResponse only
    on the calls where older history was actually summarized away — that's
    the documented signal this checks, rather than "the middleware ran"
    (which is every turn) or inspecting internal state keys directly.
    """

    async def awrap_model_call(self, request, handler):
        result = await super().awrap_model_call(request, handler)
        if isinstance(result, ExtendedModelResponse):
            print("[summarization] context limit reached — older "
                  "conversation history was compacted into a summary "
                  "before this call.")
        return result


def print_step(message) -> None:
    if isinstance(message, AIMessage) and message.tool_calls:
        for call in message.tool_calls:
            print(f"  -> calling {call['name']}({call['args']})")
    elif isinstance(message, ToolMessage):
        preview = str(message.content)
        if len(preview) > 500:
            preview = preview[:500] + "... [truncated]"
        print(f"  <- {message.name}: {preview}")
    elif isinstance(message, AIMessage) and message.content:
        print(f"\nAssistant: {message.content}\n")


def format_usage(usage: dict | None) -> str:
    if not usage:
        return "(token usage unavailable)"
    input_t = usage.get("input_tokens") or 0
    output_t = usage.get("output_tokens") or 0
    total_t = usage.get("total_tokens") or (input_t + output_t)
    return f"{input_t:,} in + {output_t:,} out = {total_t:,} total tokens"


async def run_query(agent, query: str, thread_id: str) -> None:
    # With a checkpointer attached, the graph loads prior history for this
    # thread_id itself — only the new message needs to be sent each turn.
    last_usage: dict | None = None
    async for chunk in agent.astream(
        {"messages": [HumanMessage(content=query)]},
        config={"configurable": {"thread_id": thread_id}},
        stream_mode="values",
    ):
        message = chunk["messages"][-1]
        print_step(message)
        if isinstance(message, AIMessage) and message.usage_metadata:
            # Overwritten each time — the LAST AI message in the turn is the
            # one whose input_tokens reflects the full context (system
            # prompt + accumulated history + tool results) sent to produce
            # the final answer, i.e. the current context-window usage.
            last_usage = message.usage_metadata
    print(f"[context: {format_usage(last_usage)}]")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "query",
        nargs="?",
        help="Single question to ask. Omit to start an interactive REPL.",
    )
    parser.add_argument(
        "--project",
        default=os.getcwd(),
        help="Project directory for CodeGraph to index/query (default: cwd).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        help="OpenAI model to use (default: gpt-4o, or $OPENAI_MODEL).",
    )
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        default=int(os.environ.get("MAX_CONTEXT_TOKENS", "50000")),
        help=(
            "Once the context sent to the model reaches this many tokens, "
            "older history is auto-summarized down to keep requests small. "
            "Default 50000 — well under a typical 30K-TPM OpenAI rate-limit "
            "tier (see the earlier RateLimitError this project hit), with "
            "headroom left over. Does not delete history — the full "
            "conversation stays in the checkpointer; only what's sent to "
            "the model each turn is bounded. Pass 0 to disable and rely on "
            "deep agents' own "
            "built-in default instead."
        ),
    )
    args = parser.parse_args()

    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Error: OPENAI_API_KEY is not set (env var or .env file).")

    llm = ChatOpenAI(model=args.model, temperature=0)
    mcp_client = build_mcp_client(args.project)
    codegraph_tools = await mcp_client.get_tools()
    if not codegraph_tools:
        sys.exit("No tools loaded from the codegraph MCP server.")

    hippomock_tools = make_hippomock_tools(Path(args.project))
    tools = codegraph_tools + hippomock_tools

    print(f"Loaded {len(codegraph_tools)} CodeGraph tool(s): "
          f"{', '.join(t.name for t in codegraph_tools)}")
    print(f"Loaded {len(hippomock_tools)} HippoMock tool(s): "
          f"{', '.join(t.name for t in hippomock_tools)}")
    print("Built-in filesystem tools (ls, read_file, write_file, edit_file, "
          f"glob, grep) active, rooted at: {args.project}")

    backend = FilesystemBackend(root_dir=args.project)

    middleware = []
    if args.max_context_tokens > 0:
        middleware.append(
            LoggingSummarizationMiddleware(
                model=llm,
                backend=backend,
                trigger=("tokens", args.max_context_tokens),
                keep=("messages", 5),
                # The default char-based approximate counter undercounts
                # dense, code/JSON-heavy content — measured ~9% low on the
                # HippoMock excerpt alone, and real conversations here have
                # shown gaps as large as 66% (8,328 real vs a 5,000
                # trigger), well past what the approximator's capped
                # self-correction (max 1.25x) can fix. Using ChatOpenAI's
                # own token counter instead — which implements OpenAI's
                # actual per-message/tool accounting — measured within
                # 0.3% of a raw tiktoken count, so there's no gap left to
                # correct for. allow_fetching_images=False since this
                # project has no image content and counting shouldn't
                # trigger network calls.
                token_counter=functools.partial(
                    llm.get_num_tokens_from_messages,
                    allow_fetching_images=False,
                ),
            )
        )
        print(f"Context limit: auto-summarizing history above "
              f"{args.max_context_tokens:,} tokens sent to the model.")
    else:
        print("Context limit: disabled — using deep agents' own built-in default.")

    agent = create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        backend=backend,
        checkpointer=InMemorySaver(),
        middleware=middleware,
    )
    # One thread per process run — every turn in this REPL session shares it,
    # so the agent remembers prior questions/answers. Memory is in-process
    # only: it does not survive restarting the script.
    thread_id = str(uuid.uuid4())

    if args.query:
        await run_query(agent, args.query, thread_id)
        return

    print("HippoMock test generator ready. Ask it to write a unit test for "
          "a function/class (e.g. \"write a hippomock test for add_safe\"). "
          "Conversation is remembered for this session. Type 'exit' to quit.")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break
        await run_query(agent, query, thread_id)


if __name__ == "__main__":
    asyncio.run(main())
