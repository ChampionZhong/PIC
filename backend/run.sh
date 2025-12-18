#!/bin/bash
# Run uvicorn with limited watch directories to avoid file watch limit

uvicorn main:app --reload --reload-dir . --reload-include "*.py"

