Prime Directive:
Deliver production-ready, systemic solutions to root causes. Prioritize core utility and absolute system integrity. There is zero tolerance for surface patches, brittle fixes, or non-functional code.
Mandatory Protocol:
Map the System: Before acting, map all relevant logic flows, data transformations, and dependencies. Identify all side effects.
Isolate Root Cause: Diagnose the fundamental issue with code-based evidence. Ensure the fix is systemic and permanent.
Align with Utility: Every change must advance the project's core objective. Reject low-impact optimizations.
Implementation Mandates:
Code Integrity: All code must be robust, generalizable, and directly executable. Prohibit all hardcoding, duplicated functionality, and placeholder logic.
Quality & Security: Enforce static typing, descriptive naming, and strict linting. Validate all I/O, eliminate unsafe calls, and add regression guards.
Testing: Test coverage must target both the symptom and its root cause. The full test suite must pass without warnings.
Execution Workflow:
Analyze system flow.
Confirm root cause.
Plan solution.
Implement the robust fix.
Validate with all tests.
Document systemic insights.

Project: Implements an AI avatar by streaming a user's local audio and video to a Hugging Face GPU server for immediate processing. In the cloud, the system performs simultaneous generative face swapping—animating a source image's identity with the user's live motion—and real-time voice conversion, which morphs the user's speech to a target profile while preserving the original prosody. The fully synchronized audio-visual output is then streamed back to the local machine, functioning as an integrated virtual camera and microphone for seamless use in communication platforms like Zoom and WhatsApp.

Operational instructions:
- All implementations must be architected for the huggingface space located at https://huggingface.co/spaces/Islamckennon/mirage
- After every change, push to github and huggingface, then await user feedback for next steps.
- All code must be archhitected towards project real-world functionality only.