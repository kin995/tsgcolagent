from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path


DEFAULT_EXTERNAL_MAVEN_HOMES = [
    Path(r"D:\wn\mavenEvo\apache-maven-3.9.1"),
]


@dataclass(frozen=True)
class BuildToolResolution:
    command: str | None
    available: bool
    source: str
    checked: list[str]
    message: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_maven_command(project_root: str | Path | None = None) -> BuildToolResolution:
    checked: list[str] = []
    root = Path(project_root).resolve() if project_root else None
    if root is not None:
        wrapper = root / ("mvnw.cmd" if os.name == "nt" else "mvnw")
        checked.append(str(wrapper))
        if wrapper.exists():
            return BuildToolResolution(str(wrapper), True, "project_wrapper", checked, "project Maven wrapper found")

    for env_name in ("MAVEN_CMD", "MVN_CMD"):
        configured = os.environ.get(env_name)
        if configured:
            configured = configured.strip().strip('"')
            checked.append(f"{env_name}={configured}")
            if Path(configured).exists() or shutil.which(configured):
                return BuildToolResolution(configured, True, env_name, checked, f"{env_name} points to Maven")
            return BuildToolResolution(None, False, env_name, checked, f"{env_name} is set but not executable: {configured}")

    executable_names = ["mvn.cmd", "mvn.bat", "mvn"] if os.name == "nt" else ["mvn"]
    for home in DEFAULT_EXTERNAL_MAVEN_HOMES:
        for executable in executable_names:
            candidate = home / "bin" / executable
            checked.append(str(candidate))
            if candidate.exists():
                return BuildToolResolution(str(candidate), True, "default_external_maven", checked, "default external Maven found")

    external_parent = Path(r"D:\wn\mavenEvo")
    if external_parent.exists():
        for home in sorted(external_parent.glob("apache-maven-*"), reverse=True):
            for executable in executable_names:
                candidate = home / "bin" / executable
                checked.append(str(candidate))
                if candidate.exists():
                    return BuildToolResolution(str(candidate), True, "default_external_maven_scan", checked, "default external Maven scan found")

    for env_name in ("MAVEN_HOME", "M2_HOME"):
        home = os.environ.get(env_name)
        if not home:
            continue
        for executable in executable_names:
            candidate = Path(home) / "bin" / executable
            checked.append(str(candidate))
            if candidate.exists():
                return BuildToolResolution(str(candidate), True, env_name, checked, f"{env_name}/bin Maven found")

    repo_root = Path(__file__).resolve().parents[1]
    for base in [repo_root / "tools", repo_root / "vendor", repo_root]:
        for candidate in base.glob("apache-maven*/bin/mvn*"):
            checked.append(str(candidate))
            if candidate.is_file() and candidate.name.lower() in set(executable_names):
                return BuildToolResolution(str(candidate), True, "project_local_maven", checked, "project-local Maven found")

    for executable in executable_names:
        checked.append(f"PATH:{executable}")
        found = shutil.which(executable)
        if found:
            return BuildToolResolution(found, True, "PATH", checked, "Maven found on PATH")
    return BuildToolResolution(
        None,
        False,
        "not_found",
        checked,
        "Maven executable was not found. Java evaluation cannot run JUnit/JaCoCo/PIT until Maven is available at D:\\wn\\mavenEvo\\apache-maven-3.9.1\\bin\\mvn.cmd or through mvnw, MAVEN_CMD/MVN_CMD, MAVEN_HOME/M2_HOME, project-local apache-maven, or PATH.",
    )


def maven_local_repo_arg(project_root: str | Path) -> str:
    repo = _maven_local_repo(Path(project_root).resolve())
    repo.mkdir(parents=True, exist_ok=True)
    return f"-Dmaven.repo.local={repo}"


def maven_project_settings_args(project_root: str | Path) -> list[str]:
    root = Path(project_root).resolve()
    settings_dir = root / ".m2"
    local_repo = _maven_local_repo(root)
    settings_path = settings_dir / "settings.xml"
    settings_dir.mkdir(parents=True, exist_ok=True)
    local_repo.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        """<settings xmlns="http://maven.apache.org/SETTINGS/1.2.0"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
          xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.2.0 https://maven.apache.org/xsd/settings-1.2.0.xsd">
  <localRepository>{local_repo}</localRepository>
  <profiles>
    <profile>
      <id>stateflow-central</id>
      <repositories>
        <repository>
          <id>central</id>
          <url>https://repo.maven.apache.org/maven2</url>
          <releases><enabled>true</enabled></releases>
          <snapshots><enabled>false</enabled></snapshots>
        </repository>
      </repositories>
      <pluginRepositories>
        <pluginRepository>
          <id>central</id>
          <url>https://repo.maven.apache.org/maven2</url>
          <releases><enabled>true</enabled></releases>
          <snapshots><enabled>false</enabled></snapshots>
        </pluginRepository>
      </pluginRepositories>
    </profile>
  </profiles>
  <activeProfiles>
    <activeProfile>stateflow-central</activeProfile>
  </activeProfiles>
</settings>
""".format(local_repo=str(local_repo).replace("\\", "/")),
        encoding="utf-8",
    )
    return ["-s", str(settings_path), "-gs", str(settings_path)]


def _maven_local_repo(project_root: Path) -> Path:
    if (project_root / ".stateflow_humanevaljava_workspace").is_file():
        configured = os.environ.get("STATEFLOW_HUMANEVALJAVA_MAVEN_REPO")
        if configured:
            return Path(configured).expanduser().resolve()
        return Path(__file__).resolve().parents[1] / "java_workspace" / ".m2" / "repository"
    return project_root / ".m2" / "repository"


def maven_command(project_root: str | Path | None = None) -> str:
    resolution = resolve_maven_command(project_root)
    if not resolution.available or not resolution.command:
        raise FileNotFoundError(resolution.message)
    return resolution.command
