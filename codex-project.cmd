@echo off
set "CODEX_HOME=%~dp0.codex-home"
cmd /c codex -C "%~dp0." %*
