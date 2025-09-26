Prime Directive:

Ship production code. Core utility only. Zero tolerance for patches or broken code.



MANDATORY Protocol (Non-negotiable):

1. Map all existing system flows, dependencies, side effects BEFORE coding

2. Diagnose root cause with code evidence - fix must be systemic

3. Every change MUST advance project's core objective



Implementation Rules:

- Code: Robust, generalizable, executable. NO hardcoding/duplication/placeholders

- Quality: Static typing, descriptive names, validate ALL I/O, eliminate unsafe calls

- Testing: Cover symptom AND root cause. Full suite passes clean



Issue Resolution Protocol:

- Use error logs + user feedback to isolate exact failure point

- Trace execution path from failure backwards to root cause

- Fix ONLY the identified issue - no speculative changes

- Verify fix resolves original problem without side effects



Operational Simplicity:

- DEFAULT to simple solutions - complexity requires justification

- New features MUST prove real-world utility for THIS project

- Reject abstractions that don't directly serve avatar streaming

- If implementation > 50 lines, question if simpler path exists



Workflow (ENFORCE):

1. Analyze → 2. Diagnose → 3. Plan → 4. Implement → 5. Test → 6. Document



Project: 

AI avatar: Stream local A/V → HuggingFace Spacce withh a10 GPU → Realtime face-swap + voice conversion → Stream back as virtual camera/mic for Zoom/WhatsApp.



Operations:

- Target: https://huggingface.co/spaces/Islamckennon/mirage

- Push GitHub/HuggingFace after EVERY change

- Await user feedback before proceeding

- Production functionality ONLY - no demos/experiments