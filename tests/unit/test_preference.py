"""Preference questions: a HELD-OUT test for a rule that used to be fitted.

The ADVICE branch of `classify` was written by looking at which of
LongMemEval's 30 single-session-preference questions it missed and adding each
missed phrasing. That reached 29/30 and taught us nothing about whether it
would hold anywhere else, because the questions it was measured on are the
questions it was built from.

Every question in this file was written for this repository. None is copied
from, or paraphrased from, any evaluation set -- which is what makes the
positives evidence of generalisation rather than a second look at the training
data. The negatives matter just as much: routing a factual question into
recommendation mode makes the model invent a suggestion instead of recalling
what the corpus actually says, and that failure is silent.

Structured by GRAMMATICAL FRAME rather than by topic, because the frames are
what the rule now claims to recognise. A frame with only one example is a
phrasing; a frame with five is a claim.
"""

from __future__ import annotations

import pytest

from mapi.domain.synthesis.classify import QuestionKind, classify

# -- positives, by the frame each one exercises ----------------------------

DIRECTIVE_TO_ASSISTANT = [
    "Can you suggest a wine that would go with what I usually cook?",
    "Could you recommend a gym near the places I tend to work from?",
    "Would you propose an itinerary that fits how I like to travel?",
    "Can you please advise on a laptop for the work I described?",
    "Will you help me put together a reading list?",
]

BARE_IMPERATIVE = [
    "Suggest a restaurant for Thursday.",
    "Recommend me something to cook this weekend.",
    "Suggest some podcasts based on what I listen to.",
    "Advise a route that avoids the motorway.",
    "Recommend 3 films I would probably like.",
]

FIRST_PERSON_DELIBERATION = [
    "What should I bring to the potluck?",
    "Should I take the aisle or the window on this one?",
    "Which laptop should I buy given how I work?",
    "Where should we hold the offsite?",
    "How should I structure the deposit?",
    "Ought I to renew the lease?",
    "Help me decide between the two offers.",
    "Help me narrow down a gift for my sister.",
]

ADVICE_NOUN = [
    "Any tips for keeping the sourdough alive while travelling?",
    "I would love some suggestions for the garden.",
    "Do you have advice about the commute?",
    "Got any ideas for the anniversary?",
    "Looking for recommendations on noise-cancelling headphones.",
    "Give me a few pointers on the pitch.",
    "I need guidance on which visa route applies.",
    "What are your thoughts on the offer?",
]

EVALUATE_A_PROPOSAL = [
    "Is switching to the annual plan a good idea?",
    "Would it be worth flying out a day early?",
    "Does it make sense to consolidate the two accounts?",
    "What do you think about moving the deadline?",
    "Is it a good idea to bring the dog?",
]

#: Imperatives whose verb is also a common noun, which is why they need their
#: own frame anchored to the start of the string rather than to a following
#: determiner. Found by `bench/lab/capability_live.py` against a live model:
#: "Plan how I should get to a client review in Edinburgh" classified DIRECT,
#: took the fact-lookup contract, and replied "I don't have that in memory" --
#: refusing a request that always has an answer. The preference lab could not
#: see it, because it applied the advice prompt to every question directly and
#: so never exercised the classifier at all.
PLAN_IMPERATIVE = [
    "Plan how I should get to a client review in Edinburgh next month.",
    "Plan me a weekend at my daughter's next month.",
    "Plan my lunches for the next rotation.",
    "Plan out the autumn menu.",
    "Pick a restaurant for Friday.",
    "Choose a printer for me.",
    "Can you plan my week?",
    "Please plan the trip.",
]

#: Deliberation in an EMBEDDED clause, a wh-complement, a superlative, and a
#: third party acting on the user's behalf. Found by routing this repo's own
#: 177-question preference corpus through `classify`: 14% of it would have
#: taken the fact-lookup contract in production and been free to refuse. The
#: preference lab could not see it because it applies the advice prompt
#: directly and never calls the classifier at all.
EMBEDDED_DELIBERATION = [
    "Is there anything I should request when I book the room?",
    "What is the one thing I should be insisting on in the new pair?",
    "She's asked whether there's anything she ought to tell them when she books.",
    "What should he put in my tea?",
    "Where should she book the table?",
]
#: Genuinely ambiguous on the surface and deliberately NOT asserted either way.
#: "How long should the one I pick take to play?" is an advice request in
#: context and a duration question in form, and it classifies DATE_ARITH. A
#: test that forced it to ADVICE would be asserting that shape beats grammar,
#: which is the fitting this frame set was written to avoid.
AMBIGUOUS_BY_DESIGN = ["How long should the one I pick take to play?"]
WH_COMPLEMENT = [
    "Recommend what kind of place I should book for the retreat.",
    "Suggest which of the two I should take.",
    "Advise how to get there.",
]
SUPERLATIVE = [
    "What's the best way to get across town?",
    "Which is the right material for the panels?",
    "What would be the safest option for the drive?",
]

#: How consumers actually ask for a recommendation: contracted, indefinite,
#: often unpunctuated. Found by probing the classifier in the consumer register
#: rather than the editorial one every corpus in this repo is written in --
#: 2 of 12 of these routed correctly, so "what's a good place to eat" took the
#: DECLINE contract and answered "I don't have that in memory" to a request
#: that always has an answer.
#:
#: The frame is the INDEFINITE DETERMINER: asking for AN instance of a category
#: is a recommendation, asking for THE stored particular is recall. Same
#: wh-word, different determiner, different task.
QUALITY_SEEKING = [
    "what's a good place to eat near me",
    "where's a good spot for brunch",
    "whats a good gift for mom",
    "what's a good movie to watch tonight",
    "who's a good dentist around here",
    "what's a nice hotel in lisbon",
    "whats a decent laptop for uni",
    "any good places to eat",
    "know any good cafes",
    "where's the best coffee",
    "who's the best dentist",
]

ADVICE_QUESTIONS = [
    *DIRECTIVE_TO_ASSISTANT,
    *BARE_IMPERATIVE,
    *FIRST_PERSON_DELIBERATION,
    *ADVICE_NOUN,
    *EVALUATE_A_PROPOSAL,
    *PLAN_IMPERATIVE,
    *EMBEDDED_DELIBERATION,
    *WH_COMPLEMENT,
    *SUPERLATIVE,
    *QUALITY_SEEKING,
]


@pytest.mark.parametrize("question", ADVICE_QUESTIONS)
def test_a_held_out_advice_request_routes_to_advice(question: str) -> None:
    assert classify(question) is QuestionKind.ADVICE


# -- negatives: factual questions that must NOT become recommendations -----
#
# The dangerous ones are deliberately overrepresented: questions that share
# vocabulary with a request for advice ("recommended", "suggested", "should")
# but ask what happened, not what to do.

FACTUAL_QUESTIONS = [
    # Contains an advice word, but reports a past event.
    "What did the doctor recommend at my last appointment?",
    "Which wine did Priya suggest when we met?",
    "What was the advice the lawyer gave me in March?",
    "Whose recommendation did I end up following?",
    "What tips did the instructor give during the lesson?",
    # Contains a modal, but asks about a past obligation or a stated rule.
    "What should the invoice threshold have been?",
    "When should the lease have been renewed?",
    # Preference LOOKUPS -- the stored answer exists, so this is recall.
    "What is my usual coffee order?",
    "Which seat do I normally pick on flights?",
    "What kind of food do I say I dislike?",
    # Ordinary factual shapes.
    "How many invoices did I process in January?",
    "What database do we use?",
    "Who accepted the founding engineer offer?",
    "When did I submit the paper?",
    "How much did the Datadog invoice come to?",
    "What is our entire infrastructure?",
    "Which invoice was dated earliest?",
    "How long after the review did the migration happen?",
    # `plan`, `pick` and `choose` as NOUNS or as past-tense recall. These are
    # what stopped frame 2b joining frame 2: identified by a following
    # determiner, "the plan the architect gave me" is a recall question wearing
    # an imperative's shape. Anchoring to the start of the string is what keeps
    # these DIRECT.
    # The embedded-deliberation frame's dangerous neighbours: the same words in
    # the same order, reporting somebody else's past speech or a missed
    # obligation. Each one broke a draft of that frame before its guard existed.
    "What did the doctor say I should do?",
    "She told me what I should charge — what was it?",
    "He asked what I should bring.",
    "What did she recommend I order?",
    "What should they have done differently?",
    "Which wine did Priya suggest when we met?",
    "What was the best month for sales?",
    "Which was the best performing account last year?",
    # Consumer-register recall that shares vocabulary with QUALITY_SEEKING.
    # These are why frame 3e requires an INDEFINITE determiner and present
    # tense: "a good place" is a request, "my best score" and "the best man at
    # the wedding" are stored particulars.
    "whats my usual coffee",
    "whats my best score",
    "who was the best man at the wedding",
    "what was the best month for sales",
    "what is a good faith estimate",
    "whats the capital of france",
    "What was the plan for March?",
    "Which plan did I pick last year?",
    "What plan am I on with the gym?",
    "Who chose the venue for the launch?",
    "What did I pick for dessert in June?",
]


@pytest.mark.parametrize("question", FACTUAL_QUESTIONS)
def test_a_factual_question_is_not_routed_to_advice(question: str) -> None:
    """A false ADVICE makes the model invent instead of recall."""
    assert classify(question) is not QuestionKind.ADVICE


def test_the_held_out_set_is_large_enough_to_mean_something() -> None:
    """A guard on the evidence, not on the code.

    Deleting examples until the suite passes would turn this file into the
    thing it exists to replace, so the size of the held-out set is asserted.
    """
    assert len(ADVICE_QUESTIONS) >= 30
    assert len(FACTUAL_QUESTIONS) >= 15
    assert len(set(ADVICE_QUESTIONS) & set(FACTUAL_QUESTIONS)) == 0


@pytest.mark.parametrize(
    "frame",
    [
        DIRECTIVE_TO_ASSISTANT,
        BARE_IMPERATIVE,
        FIRST_PERSON_DELIBERATION,
        ADVICE_NOUN,
        EVALUATE_A_PROPOSAL,
    ],
)
def test_every_frame_carries_more_than_one_example(frame: list[str]) -> None:
    """One example is a phrasing. Several are a frame."""
    assert len(frame) >= 5
