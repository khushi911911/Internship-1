@echo off
cd /d %~dp0
cd frontend
call npm install
call npm run dev
