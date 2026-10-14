from __future__ import annotations

import asyncio
import html as html_lib
import os
import re
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree
from typing import Mapping, Sequence


class AndroidPageError(RuntimeError):
    """Base exception raised by androidpage."""


class AmctlNotFoundError(AndroidPageError):
    """Raised when the amctl executable cannot be found."""


class AmctlCommandError(AndroidPageError):
    """Raised when amctl returns a non-zero exit code."""

    def __init__(self, args: Sequence[str], returncode: int, stderr: str = "") -> None:
        self.args = tuple(args)
        self.returncode = returncode
        self.stderr = stderr
        detail = stderr.strip() or f"exit code {returncode}"
        super().__init__(f"amctl {' '.join(self.args)} failed: {detail}")


@dataclass(frozen=True)
class AmctlResult:
    """The result of one amctl invocation."""

    args: tuple[str, ...]
    stdout: str
    stderr: str
    returncode: int


class AmctlTransport:
    """Low-level sync and async process transport for the amctl executable."""

    def __init__(
        self,
        executable: str | os.PathLike[str] = "amctl",
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.executable = os.fspath(executable)
        self.cwd = os.fspath(cwd) if cwd is not None else None
        self.env = dict(env) if env is not None else None
        self.timeout = timeout

    def _command(self, args: Sequence[object]) -> list[str]:
        return [self.executable, *(str(arg) for arg in args)]

    def run(self, *args: object) -> AmctlResult:
        command = self._command(args)
        try:
            completed = subprocess.run(
                command,
                cwd=self.cwd,
                env=self.env,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=False,
            )
        except FileNotFoundError as exc:
            raise AmctlNotFoundError(
                f"Could not find amctl executable: {self.executable!r}"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise AndroidPageError(
                f"amctl timed out after {self.timeout:g}s: {' '.join(command)}"
            ) from exc
        result = AmctlResult(tuple(command[1:]), completed.stdout, completed.stderr, completed.returncode)
        if completed.returncode:
            raise AmctlCommandError(result.args, result.returncode, result.stderr)
        return result

    async def run_async(self, *args: object) -> AmctlResult:
        command = self._command(args)
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=self.cwd,
                env=self.env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise AmctlNotFoundError(
                f"Could not find amctl executable: {self.executable!r}"
            ) from exc
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), self.timeout)
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.wait()
            raise AndroidPageError(
                f"amctl timed out after {self.timeout:g}s: {' '.join(command)}"
            ) from exc
        result = AmctlResult(
            tuple(command[1:]),
            stdout.decode(errors="replace"),
            stderr.decode(errors="replace"),
            process.returncode or 0,
        )
        if result.returncode:
            raise AmctlCommandError(result.args, result.returncode, result.stderr)
        return result


def _log_tree_to_xml(output: str) -> str | None:
    """Convert AccessDroid v1.0.4 Logcat tree lines into XML."""
    pattern = re.compile(
        r"^(?P<indent>\s*)\[(?P<class>[^\]]*)\] id=(?P<id>\S+) "
        r"bounds=(?P<bounds>.*?) txt=(?P<text>.*?) desc=(?P<desc>.*)$"
    )
    nodes: list[tuple[int, str]] = []
    for raw in output.splitlines():
        raw = re.sub(r"^.*D/AccessDroid(?:\([^)]*\))?:\s*", "", raw)
        match = pattern.match(raw)
        if not match:
            continue
        attrs = {
            "class": match.group("class"),
            "id": match.group("id"),
            "bounds": match.group("bounds"),
            "text": match.group("text"),
            "content-desc": match.group("desc"),
        }
        attr_text = " ".join(
            f'{key}="{html_lib.escape(value.strip(), quote=True)}"'
            for key, value in attrs.items()
            if value.strip() and value.strip() != "null"
        )
        depth = len(match.group("indent")) // 2
        nodes.append((depth, f"<node {attr_text}>"))

    if not nodes:
        return None
    result: list[str] = ["<hierarchy>"]
    stack: list[int] = []
    for depth, opening in nodes:
        while stack and stack[-1] >= depth:
            result.append("</node>")
            stack.pop()
        result.append(opening)
        stack.append(depth)
    while stack:
        result.append("</node>")
        stack.pop()
    result.append("</hierarchy>")
    return "".join(result)


def _clean_tree_output(output: str) -> str:
    """Remove Android broadcast logs and return the XML tree payload."""
    value = output.strip()
    if not value.startswith("<"):
        converted = _log_tree_to_xml(value)
        if converted is not None:
            return converted
        start = value.find("<")
        end = value.rfind(">")
        if start < 0 or end < start:
            preview = " ".join(value.split())[:160]
            raise AndroidPageError(
                "amctl tree returned no XML or AccessDroid tree lines; output was: " + preview
            )
        value = value[start : end + 1]
    return value


def _tree_text(xml: str) -> str:
    """Extract visible text and content descriptions from an amctl tree."""
    xml = _clean_tree_output(xml)
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise AndroidPageError("amctl tree did not return valid XML") from exc
    values: list[str] = []
    for node in root.iter():
        for key in ("text", "content-desc"):
            value = node.attrib.get(key, "").strip()
            if value and value not in values:
                values.append(value)
    return "\n".join(values)


def _tree_resolution(xml: str) -> tuple[int, int]:
    """Extract screen width and height from the root accessibility bounds."""
    xml = _clean_tree_output(xml)
    # Resolution only needs the root bounds. Read it before XML parsing so a
    # malformed text/content-desc attribute cannot prevent this lightweight
    # operation from succeeding.
    match = re.search(
        r'bounds=["\']Rect\(\s*0\s*,\s*0\s*-\s*(\d+)\s*,\s*(\d+)\s*\)',
        xml,
    )
    if not match:
        # Some AccessDroid builds omit quotes around attributes or add
        # whitespace differently. Resolution only needs the first root Rect.
        match = re.search(
            r'Rect\s*\(\s*0\s*,\s*0\s*-\s*(\d+)\s*,\s*(\d+)\s*\)',
            xml,
        )
    if match:
        return int(match.group(1)), int(match.group(2))
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise AndroidPageError("amctl tree did not return valid XML") from exc

    for node in root.iter("node"):
        bounds = node.attrib.get("bounds", "")
        match = re.search(
            r"Rect\(\s*0\s*,\s*0\s*-\s*(\d+)\s*,\s*(\d+)\s*\)",
            bounds,
        )
        if match:
            return int(match.group(1)), int(match.group(2))
    raise AndroidPageError("Could not find screen resolution in accessibility tree")


def _tree_text_at(xml: str, x: int, y: int) -> str | None:
    """Return the nearest visible text/content description covering a point."""
    xml = _clean_tree_output(xml)
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise AndroidPageError("amctl tree did not return valid XML") from exc

    candidates: list[tuple[int, str]] = []
    for node in root.iter("node"):
        match = re.search(
            r"Rect\(\s*(-?\d+)\s*,\s*(-?\d+)\s*-\s*(-?\d+)\s*,\s*(-?\d+)\s*\)",
            node.attrib.get("bounds", ""),
        )
        if not match:
            continue
        left, top, right, bottom = (int(value) for value in match.groups())
        if left <= x <= right and top <= y <= bottom:
            value = (node.attrib.get("text") or node.attrib.get("content-desc") or "").strip()
            if value:
                area = max(0, right - left) * max(0, bottom - top)
                candidates.append((area, value))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def _tree_html(xml: str) -> str:
    """Render an amctl accessibility tree as simple, escaped HTML."""
    xml = _clean_tree_output(xml)
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        raise AndroidPageError("amctl tree did not return valid XML") from exc

    def render(node: ElementTree.Element) -> str:
        label = node.attrib.get("text") or node.attrib.get("content-desc") or ""
        node_class = html_lib.escape(node.attrib.get("class", ""), quote=True)
        body = html_lib.escape(label)
        body += "".join(render(child) for child in node)
        return f'<div class="{node_class}">{body}</div>'

    return "<html><body>" + render(root) + "</body></html>"


class AndroidPage:
    """Synchronous AccessDroid controller.

    The command names follow the conventional amctl interface. If an APK build
    uses different names, pass a custom ``commands`` mapping, for example
    ``{"tap": "touch"}``.
    """

    DEFAULT_COMMANDS = {
        "tap": "tap",
        "swipe": "swipe",
        "text": "type",
        "click": "click",
        "click_id": "clickid",
        "find_id": "find",
        "copy": "copy",
        "cut": "cut",
        "paste": "paste",
        "keyevent": "press",
        "screenshot": "screenshot",
        "dump": "tree",
        "screen": "tree",
        "launch": "activity",
        "back": "back",
        "home": "home",
    }

    def __init__(
        self,
        executable: str | os.PathLike[str] = "amctl",
        *,
        transport: AmctlTransport | None = None,
        commands: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.transport = transport or AmctlTransport(executable, timeout=timeout)
        self.commands = {**self.DEFAULT_COMMANDS, **(commands or {})}

    def _cmd(self, name: str, *args: object) -> AmctlResult:
        return self.transport.run(self.commands[name], *args)

    def tap(self, x: int, y: int) -> AmctlResult:
        return self._cmd("tap", x, y)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int | None = None) -> AmctlResult:
        args: tuple[object, ...] = (x1, y1, x2, y2)
        if duration_ms is not None:
            args += (duration_ms,)
        return self._cmd("swipe", *args)

    def scroll(self, direction: str = "down", x: int = 0, y: int = 0) -> AmctlResult:
        """Scroll up or down using AccessDroid."""
        direction = direction.lower()
        if direction not in {"up", "down"}:
            raise ValueError("direction must be 'up' or 'down'")
        return self.transport.run("scroll", direction, x, y)

    def text(self, value: str) -> AmctlResult:
        return self._cmd("text", value)

    def click(self, text: str) -> AmctlResult:
        """Click the visible node whose text matches ``text``."""
        return self._cmd("click", text)

    def click_id(self, resource_id: str) -> AmctlResult:
        """Click a node by its Android resource id."""
        return self._cmd("click_id", resource_id)

    def find_id(self, resource_id: str) -> AmctlResult:
        """Ask AccessDroid to find a node by resource id."""
        return self._cmd("find_id", resource_id)

    def copy(self, text: str | None = None) -> AmctlResult:
        """Copy selected text, or put ``text`` into Android clipboard.

        When ``text`` is provided, Termux:API's ``termux-clipboard-set``
        command is used. Without it, the focused Accessibility node's
        current selection is copied through AccessDroid.
        """
        if text is None:
            return self._cmd("copy")
        executable = shutil.which("termux-clipboard-set")
        if not executable:
            raise AndroidPageError(
                "copy(text) requires Termux:API: install the Termux:API app "
                "and run `pkg install termux-api`"
            )
        try:
            completed = subprocess.run(
                [executable], input=text, capture_output=True, text=True,
                timeout=self.transport.timeout, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise AndroidPageError("termux-clipboard-set timed out") from exc
        result = AmctlResult(("copy", text), completed.stdout, completed.stderr, completed.returncode)
        if result.returncode:
            raise AndroidPageError(
                "termux-clipboard-set failed: " + (result.stderr.strip() or str(result.returncode))
            )
        return result

    def cut(self) -> AmctlResult:
        """Cut the selected text from the currently focused node."""
        return self._cmd("cut")

    def paste(self) -> AmctlResult:
        """Paste clipboard contents into the currently focused node."""
        return self._cmd("paste")

    def keyevent(self, key: str | int) -> AmctlResult:
        return self._cmd("keyevent", key)

    def screenshot(self, output: str | os.PathLike[str]) -> AmctlResult:
        return self._cmd("screenshot", Path(output))

    def dump(self) -> str:
        return self._cmd("dump").stdout

    def screen(self, format: str = "xml") -> str:
        """Return the current screen as XML, visible text, or HTML.

        AccessDroid exposes the accessibility tree through ``amctl tree``.
        Text and HTML are derived locally from that XML tree.
        """
        format = format.lower()
        result = self._cmd("screen")
        tree = result.stdout + result.stderr
        if format == "xml":
            return _clean_tree_output(tree)
        if format == "text":
            return _tree_text(tree)
        if format == "html":
            return _tree_html(tree)
        raise ValueError("format must be one of: xml, text, html")

    def xml(self) -> str:
        """Return the current accessibility tree as XML."""
        return self.screen("xml")

    def screen_text(self) -> str:
        """Return visible screen text."""
        return self.screen("text")

    def html(self) -> str:
        """Return the current screen rendered as HTML."""
        return self.screen("html")

    def resolution(self) -> tuple[int, int]:
        """Return the Android screen size as ``(width, height)``."""
        result = self._cmd("screen")
        return _tree_resolution(result.stdout + result.stderr)

    def text_at(self, x: int, y: int) -> str | None:
        """Return the text/content description at screen coordinate ``(x, y)``."""
        result = self._cmd("screen")
        return _tree_text_at(result.stdout + result.stderr, x, y)

    def launch(self, package: str) -> AmctlResult:
        return self._cmd("launch", package)

    def back(self) -> AmctlResult:
        return self._cmd("back")

    def home(self) -> AmctlResult:
        return self._cmd("home")

    def raw(self, *args: object) -> AmctlResult:
        """Execute an amctl command not covered by the convenience methods."""
        return self.transport.run(*args)


class AsyncAndroidPage:
    """Asynchronous AccessDroid controller with the same API as ``AndroidPage``."""

    def __init__(
        self,
        executable: str | os.PathLike[str] = "amctl",
        *,
        transport: AmctlTransport | None = None,
        commands: Mapping[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.transport = transport or AmctlTransport(executable, timeout=timeout)
        self.commands = {**AndroidPage.DEFAULT_COMMANDS, **(commands or {})}

    async def _cmd(self, name: str, *args: object) -> AmctlResult:
        return await self.transport.run_async(self.commands[name], *args)

    async def tap(self, x: int, y: int) -> AmctlResult:
        return await self._cmd("tap", x, y)

    async def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int | None = None) -> AmctlResult:
        args: tuple[object, ...] = (x1, y1, x2, y2)
        if duration_ms is not None:
            args += (duration_ms,)
        return await self._cmd("swipe", *args)

    async def scroll(self, direction: str = "down", x: int = 0, y: int = 0) -> AmctlResult:
        """Scroll up or down using AccessDroid."""
        direction = direction.lower()
        if direction not in {"up", "down"}:
            raise ValueError("direction must be 'up' or 'down'")
        return await self.transport.run_async("scroll", direction, x, y)

    async def text(self, value: str) -> AmctlResult:
        return await self._cmd("text", value)

    async def click(self, text: str) -> AmctlResult:
        """Click the visible node whose text matches ``text``."""
        return await self._cmd("click", text)

    async def click_id(self, resource_id: str) -> AmctlResult:
        """Click a node by its Android resource id."""
        return await self._cmd("click_id", resource_id)

    async def find_id(self, resource_id: str) -> AmctlResult:
        """Ask AccessDroid to find a node by resource id."""
        return await self._cmd("find_id", resource_id)

    async def copy(self, text: str | None = None) -> AmctlResult:
        """Copy selected text, or put ``text`` into Android clipboard."""
        if text is None:
            return await self._cmd("copy")
        executable = shutil.which("termux-clipboard-set")
        if not executable:
            raise AndroidPageError(
                "copy(text) requires Termux:API: install the Termux:API app "
                "and run `pkg install termux-api`"
            )
        try:
            process = await asyncio.create_subprocess_exec(
                executable, stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(text.encode()), self.transport.timeout
            )
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.wait()
            raise AndroidPageError("termux-clipboard-set timed out") from exc
        result = AmctlResult(
            ("copy", text), stdout.decode(errors="replace"),
            stderr.decode(errors="replace"), process.returncode or 0,
        )
        if result.returncode:
            raise AndroidPageError(
                "termux-clipboard-set failed: " + (result.stderr.strip() or str(result.returncode))
            )
        return result

    async def cut(self) -> AmctlResult:
        """Cut the selected text from the currently focused node."""
        return await self._cmd("cut")

    async def paste(self) -> AmctlResult:
        """Paste clipboard contents into the currently focused node."""
        return await self._cmd("paste")

    async def keyevent(self, key: str | int) -> AmctlResult:
        return await self._cmd("keyevent", key)

    async def screenshot(self, output: str | os.PathLike[str]) -> AmctlResult:
        return await self._cmd("screenshot", Path(output))

    async def dump(self) -> str:
        return (await self._cmd("dump")).stdout

    async def screen(self, format: str = "xml") -> str:
        """Return the current screen as XML, visible text, or HTML."""
        format = format.lower()
        result = await self._cmd("screen")
        tree = result.stdout + result.stderr
        if format == "xml":
            return _clean_tree_output(tree)
        if format == "text":
            return _tree_text(tree)
        if format == "html":
            return _tree_html(tree)
        raise ValueError("format must be one of: xml, text, html")

    async def xml(self) -> str:
        """Return the current accessibility tree as XML."""
        return await self.screen("xml")

    async def screen_text(self) -> str:
        """Return visible screen text."""
        return await self.screen("text")

    async def html(self) -> str:
        """Return the current screen rendered as HTML."""
        return await self.screen("html")

    async def resolution(self) -> tuple[int, int]:
        """Return the Android screen size as ``(width, height)``."""
        result = await self._cmd("screen")
        return _tree_resolution(result.stdout + result.stderr)

    async def text_at(self, x: int, y: int) -> str | None:
        """Return the text/content description at screen coordinate ``(x, y)``."""
        result = await self._cmd("screen")
        return _tree_text_at(result.stdout + result.stderr, x, y)

    async def launch(self, package: str) -> AmctlResult:
        return await self._cmd("launch", package)

    async def back(self) -> AmctlResult:
        return await self._cmd("back")

    async def home(self) -> AmctlResult:
        return await self._cmd("home")

    async def raw(self, *args: object) -> AmctlResult:
        return await self.transport.run_async(*args)
