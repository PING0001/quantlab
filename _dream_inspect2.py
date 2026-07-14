"""Read-only deep inspection of mimocode.db trajectory for dream consolidation."""
import sqlite3
import json

DB = r"C:\Users\cui\.local\share\mimocode\mimocode.db"
conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
c = conn.cursor()

PROJECT_ID = "3911cfa2-9a2c-44d2-a2d2-9fb838af162e"
CURRENT_SESSION = "ses_0b039a409ffeoLDJjtEvDR9MmL"

# Get all sessions for this project
c.execute("SELECT id, title, time_created FROM session WHERE project_id=? ORDER BY time_created DESC", (PROJECT_ID,))
sessions = c.fetchall()
print("=== PROJECT SESSIONS ===")
for s in sessions:
    print(f"  {s[0]}: {s[1]} (created: {s[2]})")

# Get messages for each session (excluding the current dream session)
for sid, title, _ in sessions:
    if sid == CURRENT_SESSION:
        continue
    print(f"\n=== MESSAGES for session: {title} ({sid}) ===")
    c.execute("""
        SELECT m.id, m.agent_id, json_extract(m.data, '$.role') as role,
               m.time_created, m.data
        FROM message m
        WHERE m.session_id = ?
        ORDER BY m.time_created
    """, (sid,))
    msgs = c.fetchall()
    print(f"  Total messages: {len(msgs)}")
    for mid, agent_id, role, ts, data in msgs:
        data_json = json.loads(data) if data else {}
        content_preview = ""
        if 'content' in data_json:
            if isinstance(data_json['content'], str):
                content_preview = data_json['content'][:200]
            elif isinstance(data_json['content'], list):
                for block in data_json['content']:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        content_preview = block['text'][:200]
                        break
        print(f"  [{role}] agent={agent_id or 'main'}: {content_preview}")

# Get parts for each session
for sid, title, _ in sessions:
    if sid == CURRENT_SESSION:
        continue
    print(f"\n=== PARTS for session: {title} ({sid}) ===")
    c.execute("""
        SELECT p.id, p.message_id, json_extract(p.data, '$.type') as part_type,
               substr(p.data, 1, 500) as preview
        FROM part p
        WHERE p.session_id = ?
        ORDER BY p.time_created
    """, (sid,))
    parts = c.fetchall()
    print(f"  Total parts: {len(parts)}")
    for pid, mid, ptype, preview in parts:
        print(f"  [{ptype}] msg={mid[:20]}... : {preview[:300]}")

# Check memory_fts entries
print("\n=== MEMORY FTS ENTRIES ===")
c.execute("SELECT id, path, scope, scope_id, type, substr(body, 1, 200) FROM memory_fts")
mem_rows = c.fetchall()
print(f"  Total memory entries: {len(mem_rows)}")
for mid, path, scope, scope_id, mtype, body in mem_rows:
    print(f"  [{scope}:{scope_id}] type={mtype} path={path}: {body[:150]}")

# Check tasks
print("\n=== TASKS ===")
c.execute("SELECT id, session_id, status, substr(summary, 1, 200), created_at FROM task ORDER BY created_at DESC")
tasks = c.fetchall()
print(f"  Total tasks: {len(tasks)}")
for tid, sid, status, summary, created in tasks:
    print(f"  [{status}] {tid} session={sid[:20]}... : {summary}")

# Check task events
print("\n=== TASK EVENTS ===")
c.execute("SELECT task_id, kind, substr(summary, 1, 200), at FROM task_event ORDER BY at DESC LIMIT 30")
events = c.fetchall()
for tid, kind, summary, at in events:
    print(f"  [{kind}] {tid[:20]}... at={at}: {summary}")

# Check actor_registry for subagents
print("\n=== ACTOR REGISTRY ===")
c.execute("SELECT session_id, actor_id, agent, status, description, turn_count FROM actor_registry ORDER BY time_created DESC")
actors = c.fetchall()
print(f"  Total actors: {len(actors)}")
for sid, aid, agent, status, desc, turns in actors:
    print(f"  [{status}] {aid[:20]}... agent={agent} turns={turns} desc={desc[:80]}")

conn.close()
