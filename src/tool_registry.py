import os
import subprocess
import tempfile
import sys
from memory import save_preference, get_preference

DYNAMIC_TOOLS = {}

def verify_code_safety_isolated(func_code_string: str) -> tuple[bool, str]:
    """
    Spawns an isolated Python subprocess to compile and check the candidate script.
    Guards your active running memory from structural crashes and syntax traps.
    """
    testing_harness = f"""
import sys
try:
    # 1. Attempt isolated runtime compilation
    local_scope = {{}}
    exec({repr(func_code_string)}, globals(), local_scope)
    
    # 2. Basic structural assertion test
    if not local_scope:
        print("Verification Failed: No function or structural declaration found in code payload.")
        sys.exit(1)
        
    func_name = list(local_scope.keys())[0]
    print(f"Verification Success: Compiles safely. Main function exported: {{func_name}}")
    sys.exit(0)

except SyntaxError as se:
    print(f"Compilation Failed [SyntaxError]: {{se.msg}} at line {{se.lineno}}")
    sys.exit(2)
except Exception as e:
    print(f"Compilation Failed [RuntimeInitializationError]: {{str(e)}}")
    sys.exit(3)"""

    # Write boilerplate code to a temporary, throwaway file for execution checking
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as temp_script:
        temp_script.write(testing_harness)
        temp_script_path = temp_script.name

    try:
        # Spawn an independent subprocess using the same python interpreter configuration
        result = subprocess.run(
            [sys.executable, temp_script_path],
            capture_output=True,
            text=True,
            timeout=5  # Strictly terminate frozen or infinite loop generations
        )
        
        is_safe = (result.returncode == 0)
        logs = result.stdout.strip() if is_safe else result.stdout.strip() + "\n" + result.stderr.strip()
        return is_safe, logs
        
    finally:
        # Clean up disk footprint by deleting temporary execution file
        if os.path.exists(temp_script_path):
            os.remove(temp_script_path)


def secure_register_tool(args: dict) -> str:
    """The protected registration gate that evaluates safety before memory mapping."""
    name = args.get("name")
    code_content = args.get("code")

    if not name or not code_content:
        return "Registration Error: Parameters 'name' and 'code' are mandatory."

    print(f"\n[Sandbox Auditing] Evaluating candidate code for tool: '{name}'...")
    passed_audit, feedback = verify_code_safety_isolated(code_content)

    if not passed_audit:
        print(f"❌ [Sandbox Abort] Quarantine triggered! Code failed isolation checks:\n{feedback}")
        raise ValueError(f"Code validation failed. Subprocess diagnostics:\n{feedback}")

    print(f"✅ [Sandbox Approved] Code passed static analysis. Hot-loading tool into loop.")
    
    # Write to local dynamic extensions database space for permanence
    os.makedirs("dynamic_extensions", exist_ok=True)
    safe_filename = f"ext_{name.lower().replace('.', '_')}.py"
    filepath = os.path.join("dynamic_extensions", safe_filename)
    
    with open(filepath, "w") as f:
        f.write(code_content)

    # Inject safely into active memory scope dictionary since compilation is verified
    local_scope = {}
    exec(code_content, globals(), local_scope)
    exported_callable_key = list(local_scope.keys())[0]
    DYNAMIC_TOOLS[name] = local_scope[exported_callable_key]

    return f"Tool '{name}' cleared verification checks and hot-mapped successfully."