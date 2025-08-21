import re, yaml, pathlib

env = yaml.safe_load(open("environment.yml"))
pins = []

for dep in env.get("dependencies", []):
    # conda pin like "name=1.2.3"
    if isinstance(dep, str):
        m = re.match(r"^([A-Za-z0-9_.-]+)=(\d+(?:\.\d+)*)$", dep)
        if m:
            name, ver = m.groups()
            pins.append(f"{name.replace('_','-')}=={ver}")
    # pip subsection in environment.yml (if any)
    elif isinstance(dep, dict) and "pip" in dep:
        pins.extend(dep["pip"])

pathlib.Path("constraints.txt").write_text("\n".join(pins) + "\n")
print(f"Wrote constraints.txt with {len(pins)} pins")