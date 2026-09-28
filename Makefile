# ==============================================================================
# Quant Engine Makefile: Universal Agent Convergence & Verification Target
# ==============================================================================

.PHONY: all test lint converge help audit-deliverables

all: test converge

help:
	@echo "Quant Engine Industrial Workflow Automation:"
	@echo "  make test               - Run all invariant, security, and state machine unit tests"
	@echo "  make converge           - Run end-to-end agent convergence lifecycle"
	@echo "  make audit-deliverables - Audit research reports against Dual-Convergence Protocol"

test:
	@echo ">>> [1/3] Running White-Hat Security Invariant Tests..."
	python3 tests/test_whitehat_security.py
	@echo ">>> [2/3] Running Convergence State Machine Tests..."
	python3 tests/test_agent_state_machine.py
	@echo ">>> [3/3] Running CI Convergence Validator Tests..."
	python3 tests/test_convergence_validator.py
	@echo ">>> ALL UNIT TESTS PASSED!"

converge:
	@echo ">>> Executing Mandatory Autonomous Agent Convergence Lifecycle..."
	python3 scripts/agent_workflow_cli.py --strategy "SMC_Sovereign_Crypto_Suite"

audit-deliverables:
	@echo ">>> Auditing all active research deliverables..."
	python3 scripts/validate_agent_convergence.py docs/research/WORKFLOW_AGENT_FINAL_VERDICT.md
