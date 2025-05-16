#!/usr/bin/env python
"""
Chat-agent minimale che usa OpenAI GPT-4o + rag_tools.py
– gestisce tool-call JSON corretti –
"""
from inspect import signature, _empty
import json, os, sys, importlib, readline
import inspect
import openai
from dotenv import load_dotenv
load_dotenv()

# importa rag_tools
sys.path.append("/Users/stefanoroybisignano/Workspace/BisiAgent007")
rag_tools = importlib.import_module("rag_tools")

# mapping funzione-nome → callable
TOOLS = {
    "codebase_search": rag_tools.codebase_search,
    "read_file":       rag_tools.read_file,
    "list_dir":        rag_tools.list_dir,
    "grep_search":     rag_tools.grep_search,
    "file_search":     rag_tools.file_search,
    "run_terminal_cmd":rag_tools.run_terminal_cmd,
    "edit_file":       rag_tools.edit_file,
    "delete_file":     rag_tools.delete_file,
    "diff_history":    rag_tools.diff_history,
    "web_search":      rag_tools.web_search,
}

# ------------------------------------------------------------------ helper: build OpenAI tool schema

def build_schema(fn_name, fn_obj):
    """Costruisce lo schema OpenAI leggendo la signature della funzione."""
    sig = signature(fn_obj)
    props, required = {}, []
    for name, param in sig.parameters.items():
        if name in {"self", "cls"}:
            continue
        # tipo JSON di default → string
        json_type = "string"
        if param.annotation in (int, "int"):
            json_type = "integer"
        elif param.annotation in (bool, "bool"):
            json_type = "boolean"
        elif param.annotation in (list, "list"):
            json_type = "array"
        props[name] = {
            "type": json_type,
            "description": f"{name} parameter for {fn_name}"
        }
        if param.default is _empty:
            required.append(name)

    return {
        "type": "function",
        "function": {
            "name": fn_name,
            "description": f"Proxy to rag_tools.{fn_name}",
            "parameters": {
                "type": "object",
                "properties": props,
                "required": required,
            },
        },
    }


OPENAI_TOOLS = [build_schema(n, f) for n, f in TOOLS.items()]

# ------------------------------------------------------------------ prompt & history
SYSTEM_PROMPT = open("/Users/stefanoroybisignano/Desktop/Projects/BisiAgent007/system_prompt.txt").read()
history = [{"role": "system", "content": SYSTEM_PROMPT}]

# ------------------------------------------------------------------ executor

def call_tool(tool_call):
    """
    Esegue la tool-call restituita da OpenAI e la converte in un messaggio
    da inserire nella history. Gestisce sia il formato nuovo (ChatCompletionMessageToolCall)
    sia il vecchio dizionario. Aggiunge tool_call_id al risultato.
    """
    # ------ estrai nome, id e arguments ------
    if hasattr(tool_call, "function"):          # nuovo SDK
        name = tool_call.function.name
        raw_args = tool_call.function.arguments
        tc_id = tool_call.id
    else:                                       # vecchio formato dict
        name = tool_call["name"]
        raw_args = tool_call.get("arguments", "{}")
        tc_id = tool_call.get("id")

    # decodifica arguments
    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args

    # esegui il vero strumento
    result = TOOLS[name](**args)

    # ritorna il messaggio tool con tool_call_id
    return {
        "role": "tool",
        "name": name,
        "tool_call_id": tc_id,
        "content": json.dumps(result, ensure_ascii=False),
    }

def chat_loop():
    while True:
        try:
            user_msg = input("👤  ")
        except (EOFError, KeyboardInterrupt):
            break
        history.append({"role": "user", "content": user_msg})

        while True:
            resp = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=history,
                tools=OPENAI_TOOLS,
                tool_choice="auto",
            )
            msg = resp.choices[0].message

            # se il modello chiede di usare un tool
            if msg.tool_calls:
                # 1) aggiungo il messaggio assistant con tool_calls (includendo gli id)
                history.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tc.to_dict() for tc in msg.tool_calls]
                })
                # 2) eseguo ciascun tool e lo appendo
                for tc in msg.tool_calls:
                    history.append(call_tool(tc))
                # 3) rilancio la completion con i risultati dei tool
                continue

            # risposta finale user-visibile
            print("🤖", msg.content.strip())
            history.append({"role": "assistant", "content": msg.content})
            break

# ------------------------------------------------------------------ main
if __name__ == "__main__":
    os.chdir("/Users/stefanoroybisignano/Desktop/Projects/mlops_finetuning_framework/src")
    chat_loop()
