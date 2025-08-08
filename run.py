#!/usr/bin/env python3
"""
Unified CLI for the GitHub Duplicate Issue Finder service.
This consolidates commands into clear groups without duplication.
"""

import subprocess
import os
from pathlib import Path
from typing import Optional
import typer
from enum import Enum

# Create the main Typer app
__version__ = "0.2.0"


class Operation(str, Enum):
    summarize = "summarize"
    create_questions = "create-questions"
    create_questions_and_concepts = "create-questions-and-concepts"


class TableName(str, Enum):
    issues = "issues"
    discourse_posts = "discourse_posts"
    metabase_docs = "metabase_docs"


app = typer.Typer(
    name="GitHub Duplicate Issue Finder",
    help="Unified service runner",
    add_completion=False,
)


def run_command(cmd: list[str], description: str) -> bool:
    print(f"\n🚀 {description}")
    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️  {description} interrupted by user")
        return False


def change_to_project_root():
    project_root = Path(__file__).parent
    os.chdir(project_root)


# Sub-apps matching requested groups
api_app = typer.Typer(help="Start API or manage API keys")
workers_app = typer.Typer(help="Start workers (GitHub, Discourse, LLM summaries, embeddings, batch monitor)")
populate_app = typer.Typer(help="Populate database from sources (GitHub, Discourse, Docs/Glossary)")
batch_app = typer.Typer(help="Batch processing: create, status, pending, files management")
db_app = typer.Typer(help="Database setup and table recreation")
keywords_app = typer.Typer(help="Manage keyword definitions")
synonyms_app = typer.Typer(help="Manage synonyms")

# Mount sub-apps
app.add_typer(api_app, name="api")
app.add_typer(workers_app, name="workers")
app.add_typer(populate_app, name="populate")
app.add_typer(batch_app, name="batch")
app.add_typer(db_app, name="db")
app.add_typer(keywords_app, name="keywords")
app.add_typer(synonyms_app, name="synonyms")


# ---------- API ----------
@api_app.command("start")
def api_start(
    host: str = typer.Option("0.0.0.0", help="Host"),
    port: int = typer.Option(8000, help="Port"),
    reload: bool = typer.Option(True, help="Enable auto-reload"),
    log_level: str = typer.Option("info", help="Log level"),
):
    """Start the FastAPI server via uvicorn."""
    change_to_project_root()
    cmd = [
        "python",
        "-m",
        "uvicorn",
        "src.api:app",
        "--host",
        host,
        "--port",
        str(port),
        "--log-level",
        log_level,
    ]
    if reload:
        cmd.append("--reload")
    run_command(cmd, "Starting API server")


@api_app.command("add-key")
def api_add_key(description: str = typer.Argument(..., help="API key description")):
    """Create and store a new API key."""
    change_to_project_root()
    cmd = ["python", "scripts/manage_db.py", "--add-api-key", description]
    run_command(cmd, "Adding API key")


@app.command()
def version():
    typer.echo(__version__)


# ---------- WORKERS ----------
@workers_app.command("github")
def workers_github():
    change_to_project_root()
    run_command(["python", "scripts/monitor_worker.py", "github_issues"], "Starting GitHub worker")


@workers_app.command("discourse")
def workers_discourse():
    change_to_project_root()
    run_command(["python", "scripts/monitor_worker.py", "discourse_posts"], "Starting Discourse worker")


@workers_app.command("llm")
def workers_llm():
    change_to_project_root()
    run_command(["python", "scripts/monitor_worker.py", "llm_summaries"], "Starting LLM summaries worker")


@workers_app.command("embeddings")
def workers_embeddings():
    change_to_project_root()
    run_command(["python", "scripts/monitor_worker.py", "embeddings"], "Starting embeddings worker")


@workers_app.command("batch-monitor")
def workers_batch_monitor():
    change_to_project_root()
    run_command(["python", "scripts/batch_monitor_worker.py"], "Starting batch monitor worker")


# ---------- POPULATE ----------
@populate_app.command("github")
def populate_github():
    """Fetch GitHub issues and store with embeddings."""
    change_to_project_root()
    run_command(["python", "scripts/populate_database.py", "--fetch-and-embed"], "Populating from GitHub")


@populate_app.command("discourse")
def populate_discourse():
    """Fetch Discourse posts and store with embeddings."""
    change_to_project_root()
    run_command(["python", "scripts/populate_database.py", "--fetch-discourse"], "Populating from Discourse")


@populate_app.command("docs")
def populate_docs():
    """Crawl documentation and store in DB."""
    change_to_project_root()
    os.chdir("js")
    run_command(["node", "run_crawler.js", "docs"], "Running docs crawler")


@populate_app.command("glossary")
def populate_glossary():
    """Crawl glossary and store in DB."""
    change_to_project_root()
    os.chdir("js")
    run_command(["node", "run_crawler.js", "glossary"], "Running glossary crawler")


# ---------- BATCH PROCESSING ----------
@batch_app.command("create")
def batch_create(
    operation: Operation = typer.Argument(..., help="summarize | create-questions | create-questions-and-concepts"),
    table: TableName = typer.Argument(..., help="issues | discourse_posts | metabase_docs"),
):
    change_to_project_root()
    cmd = ["python", "scripts/batch_manager.py", "create", operation.value, table.value]
    run_command(cmd, f"Submitting batch for {operation.value} on {table.value}")


@batch_app.command("status")
def batch_status(batch_id: Optional[str] = typer.Option(None, help="Specific batch ID")):
    change_to_project_root()
    cmd = ["python", "scripts/batch_manager.py", "status"]
    if batch_id:
        cmd.extend(["--batch-id", batch_id])
    run_command(cmd, "Checking batch status")


@batch_app.command("pending")
def batch_pending():
    change_to_project_root()
    run_command(["python", "scripts/batch_manager.py", "pending"], "Listing pending items")


@batch_app.command("backfill-all")
def batch_backfill_all():
    change_to_project_root()
    run_command(["python", "scripts/batch_manager.py", "backfill-all"], "Backfilling all operations")


# Files management (remote and local)
batch_files_app = typer.Typer(help="Manage remote batch files (list/delete)")
local_files_app = typer.Typer(help="Manage local batch files (list/delete)")
batch_app.add_typer(batch_files_app, name="files")
batch_app.add_typer(local_files_app, name="local")


@batch_files_app.command("list")
def batch_files_list(
    purpose: Optional[str] = typer.Option(None, help="Filter by purpose"),
    details: bool = typer.Option(False, help="Show details"),
):
    change_to_project_root()
    cmd = ["python", "scripts/manage_batch_files.py", "--list"]
    if purpose:
        cmd.extend(["--purpose", purpose])
    if details:
        cmd.append("--details")
    run_command(cmd, "Listing remote batch files")


@batch_files_app.command("delete-pattern")
def batch_files_delete_pattern(pattern: str = typer.Argument(..., help="Substring to match")):
    change_to_project_root()
    cmd = ["python", "scripts/manage_batch_files.py", "--delete-pattern", pattern]
    run_command(cmd, f"Deleting remote files matching: {pattern}")


@batch_files_app.command("delete-older-than")
def batch_files_delete_old(days: int = typer.Argument(..., help="Age in days")):
    change_to_project_root()
    cmd = ["python", "scripts/manage_batch_files.py", "--delete-older-than", str(days)]
    run_command(cmd, f"Deleting remote files older than {days} days")


@batch_files_app.command("delete-all-batch")
def batch_files_delete_all_batch():
    change_to_project_root()
    run_command(["python", "scripts/manage_batch_files.py", "--delete-all-batch"], "Deleting all remote batch-purpose files")


@batch_files_app.command("test-connection")
def batch_files_test_connection():
    change_to_project_root()
    run_command(["python", "scripts/manage_batch_files.py", "--test-connection"], "Testing files API connection")


@local_files_app.command("list")
def local_files_list(pattern: Optional[str] = typer.Option(None, help="Filter by pattern")):
    change_to_project_root()
    cmd = ["python", "scripts/manage_local_batch_files.py", "--list"]
    if pattern:
        cmd.extend(["--pattern", pattern])
    run_command(cmd, "Listing local batch files")


@local_files_app.command("delete-pattern")
def local_files_delete_pattern(pattern: str = typer.Argument(..., help="Substring to match")):
    change_to_project_root()
    cmd = ["python", "scripts/manage_local_batch_files.py", "--delete-pattern", pattern]
    run_command(cmd, f"Deleting local files matching: {pattern}")


@local_files_app.command("delete-older-than")
def local_files_delete_old(days: int = typer.Argument(..., help="Age in days")):
    change_to_project_root()
    cmd = ["python", "scripts/manage_local_batch_files.py", "--delete-older-than", str(days)]
    run_command(cmd, f"Deleting local files older than {days} days")


@local_files_app.command("status")
def local_files_status():
    change_to_project_root()
    run_command(["python", "scripts/manage_local_batch_files.py", "--status"], "Showing batch process status from DB")


@local_files_app.command("clean-orphans")
def local_files_clean_orphans():
    change_to_project_root()
    run_command(["python", "scripts/manage_local_batch_files.py", "--clean-orphans"], "Cleaning orphan local files")


# ---------- DATABASE ----------
@db_app.command("enable-vector")
def db_enable_vector():
    change_to_project_root()
    run_command(["python", "scripts/manage_db.py", "--enable-vector"], "Enabling pgvector extension")


@db_app.command("recreate")
def db_recreate(
    all_tables: bool = typer.Option(False, "--all", help="Drop and recreate all tables"),
    issues: bool = typer.Option(False, help="Recreate issues"),
    discourse: bool = typer.Option(False, help="Recreate discourse_posts"),
    metabase_docs: bool = typer.Option(False, help="Recreate metabase_docs"),
    questions: bool = typer.Option(False, help="Recreate questions"),
    chat_sessions: bool = typer.Option(False, help="Recreate chat_sessions"),
    chat_session_entities: bool = typer.Option(False, help="Recreate chat_session_entities"),
    keyword_definitions: bool = typer.Option(False, help="Recreate keyword_definitions"),
    synonyms: bool = typer.Option(False, help="Recreate synonyms"),
    batches: bool = typer.Option(False, help="Recreate batch_processes"),
):
    change_to_project_root()
    cmd = ["python", "scripts/manage_db.py"]
    if all_tables:
        cmd.append("--recreate")
    if issues:
        cmd.append("--recreate-issues")
    if discourse:
        cmd.append("--recreate-discourse")
    if metabase_docs:
        cmd.append("--recreate-metabase-docs")
    if questions:
        cmd.append("--recreate-questions")
    if chat_sessions:
        cmd.append("--recreate-chat-sessions")
    if chat_session_entities:
        cmd.append("--recreate-chat-session-entities")
    if keyword_definitions:
        cmd.append("--recreate-keyword-definitions")
    if synonyms:
        cmd.append("--recreate-synonyms")
    if batches:
        cmd.append("--recreate-batch-processes")
    if len(cmd) == 2:
        typer.echo("No tables selected. Use --all or specific flags.")
        raise typer.Exit(1)
    run_command(cmd, "Recreating selected tables")


# ---------- KEYWORDS ----------
@keywords_app.command("add")
def keywords_add(
    keyword: str = typer.Argument(..., help="Keyword"),
    definition: str = typer.Argument(..., help="Definition"),
    category: Optional[str] = typer.Option(None, help="Category"),
):
    change_to_project_root()
    cmd = ["python", "scripts/manage_keywords.py", "--add", keyword, definition]
    if category:
        cmd.extend(["--category", category])
    run_command(cmd, f"Adding keyword: {keyword}")


@keywords_app.command("update")
def keywords_update(
    keyword: str = typer.Argument(...),
    definition: str = typer.Argument(...),
    category: Optional[str] = typer.Option(None),
):
    change_to_project_root()
    cmd = ["python", "scripts/manage_keywords.py", "--update", keyword, definition]
    if category:
        cmd.extend(["--category", category])
    run_command(cmd, f"Updating keyword: {keyword}")


@keywords_app.command("delete")
def keywords_delete(keyword: str = typer.Argument(...)):
    change_to_project_root()
    run_command(["python", "scripts/manage_keywords.py", "--delete", keyword], f"Deleting keyword: {keyword}")


@keywords_app.command("toggle")
def keywords_toggle(keyword: str = typer.Argument(...)):
    change_to_project_root()
    run_command(["python", "scripts/manage_keywords.py", "--toggle", keyword], f"Toggling keyword: {keyword}")


@keywords_app.command("list")
def keywords_list(
    category: Optional[str] = typer.Option(None),
    show_inactive: bool = typer.Option(False),
):
    change_to_project_root()
    cmd = ["python", "scripts/manage_keywords.py", "--list"]
    if category:
        cmd.extend(["--category", category])
    if show_inactive:
        cmd.append("--show-inactive")
    run_command(cmd, "Listing keywords")


@keywords_app.command("search")
def keywords_search(term: str = typer.Argument(...)):
    change_to_project_root()
    run_command(["python", "scripts/manage_keywords.py", "--search", term], f"Searching: {term}")


@keywords_app.command("stats")
def keywords_stats():
    change_to_project_root()
    run_command(["python", "scripts/manage_keywords.py", "--stats"], "Keyword stats")


# ---------- SYNONYMS ----------
@synonyms_app.command("add")
def synonyms_add(
    word: str = typer.Argument(..., help="Word"),
    synonym_of: str = typer.Argument(..., help="Canonical keyword"),
):
    """Add a synonym relation (word -> synonym_of)."""
    change_to_project_root()
    try:
        from src.db import SessionLocal
        from src.models import Synonym
        db = SessionLocal()
        db.add(Synonym(word=word, synonym_of=synonym_of))
        db.commit()
        print(f"✓ Added synonym: {word} -> {synonym_of}")
    except Exception as e:
        print(f"✗ Failed to add synonym: {e}")
    finally:
        try:
            db.close()
        except Exception:
            pass


@synonyms_app.command("delete")
def synonyms_delete(
    word: str = typer.Argument(..., help="Word"),
    synonym_of: str = typer.Argument(..., help="Canonical keyword"),
):
    """Delete a synonym relation."""
    change_to_project_root()
    try:
        from src.db import SessionLocal
        from src.models import Synonym
        db = SessionLocal()
        deleted = db.query(Synonym).filter(Synonym.word == word, Synonym.synonym_of == synonym_of).delete()
        db.commit()
        if deleted:
            print(f"✓ Deleted synonym: {word} -> {synonym_of}")
        else:
            print("No matching synonym found")
    except Exception as e:
        print(f"✗ Failed to delete synonym: {e}")
    finally:
        try:
            db.close()
        except Exception:
            pass


@synonyms_app.command("list")
def synonyms_list(of: Optional[str] = typer.Option(None, "--of", help="Filter by canonical keyword")):
    """List synonyms, optionally filtered by canonical keyword."""
    change_to_project_root()
    try:
        from src.db import SessionLocal
        from src.models import Synonym
        db = SessionLocal()
        q = db.query(Synonym)
        if of:
            q = q.filter(Synonym.synonym_of == of)
        rows = q.order_by(Synonym.synonym_of, Synonym.word).all()
        if not rows:
            print("No synonyms found")
            return
        current = None
        for syn in rows:
            if syn.synonym_of != current:
                current = syn.synonym_of
                print(f"\n{current}:")
            print(f"  - {syn.word}")
    except Exception as e:
        print(f"✗ Failed to list synonyms: {e}")
    finally:
        try:
            db.close()
        except Exception:
            pass


if __name__ == "__main__":
    app()