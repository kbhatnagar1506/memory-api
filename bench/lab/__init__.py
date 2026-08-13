"""The lab: experiments that find what to improve, on our own corpora.

`tests/` proves nothing broke. `bench/` scores public benchmarks. This is the
third thing: instruments for finding the NEXT improvement, built after the format
experiment taught us the shape such an instrument has to have.

What that experiment found, and why it dictates the design:

  * Eleven context formats scored 10-11/15 -- FLAT. Format was the wrong
    variable. The signal was in the four questions that failed in every format.
  * Of those four, THREE were scorer bugs, not system failures. "Yes" was
    rejected because the expected string demanded "yes, 180 vs 210"; "aisle"
    never matched because the expected token carried a comma. A broken scorer
    silently converts correct answers into fake findings, so `scoring.py` exists
    and is tested.
  * The fourth was real, and the stage trace located it in one step: the
    migration memory was NOT in the evidence, so the model answered correctly
    from what it saw. Retrieval failure, not synthesis -- and the fix (the
    `asked_at` wiring) was verified causally by re-running retrieval with the
    flag on: 0 March memories in top-6 became 2, including the answer.

So the loop this package implements:

    corpus (ours, not LME)  ->  trace (which STAGE failed)  ->  fix
        ->  experiment (prompt / format / policy, evidence held fixed)
        ->  score (a scorer that cannot invent failures)

Everything retrieval-side runs with no credentials. Only the answer arms need a
model, and they reuse cached evidence so iterating on prompts costs answer calls
and nothing else.
"""
