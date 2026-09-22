"""Follow-up questions: the rewrite step, and how the pipeline and CLI use it."""

from text2sql import cli
from text2sql.conversation import FollowUpRewriter, Turn
from text2sql.llm import FakeLLM
from text2sql.pipeline import PipelineResult

HISTORY = [
    Turn("How many invoices were issued in 2010?", "SELECT COUNT(*) FROM Invoice WHERE ...", "83")
]


def test_first_question_is_not_rewritten_and_costs_no_call():
    llm = FakeLLM(['{"standalone": "should not be used"}'])
    assert FollowUpRewriter(llm).rewrite([], "How many customers?") == "How many customers?"
    assert llm.calls == []


def test_follow_up_is_rewritten_with_the_previous_turn_as_context():
    llm = FakeLLM(['{"standalone": "How many invoices were issued in 2012?"}'])
    standalone = FollowUpRewriter(llm).rewrite(HISTORY, "and in 2012?")
    assert standalone == "How many invoices were issued in 2012?"
    prompt = llm.calls[0][1]
    assert "How many invoices were issued in 2010?" in prompt
    assert "SELECT COUNT(*) FROM Invoice" in prompt  # the SQL pins down the meaning
    assert prompt.rstrip().endswith('{"standalone": "the question"}')


def test_only_recent_turns_are_sent():
    history = [Turn(f"question {i}") for i in range(6)]
    llm = FakeLLM(['{"standalone": "x"}'])
    FollowUpRewriter(llm).rewrite(history, "and then?")
    prompt = llm.calls[0][1]
    assert "question 2" not in prompt and "question 3" in prompt and "question 5" in prompt


def test_unusable_reply_keeps_the_original_question():
    llm = FakeLLM(["Sure, I think you mean 2012."])
    assert FollowUpRewriter(llm).rewrite(HISTORY, "and in 2012?") == "and in 2012?"


def test_a_reply_that_is_not_one_question_keeps_the_original():
    for reply in ['{"standalone": ["In 2012?", "In 2013?"]}', '{"standalone": 2012}']:
        llm = FakeLLM([reply])
        assert FollowUpRewriter(llm).rewrite(HISTORY, "and in 2012 and 2013?") == (
            "and in 2012 and 2013?"
        )


def test_pipeline_answers_the_standalone_question(make_pipeline):
    pipe, sql_llm = make_pipeline(["SELECT COUNT(*) AS n FROM orders"])
    pipe.rewriter = FollowUpRewriter(FakeLLM(['{"standalone": "How many orders in 2025?"}']))
    out = pipe.ask("and in 2025?", history=[Turn("How many orders in 2024?")])
    assert out.ok, out.message
    assert out.question == "and in 2025?"
    assert out.interpreted_as == "How many orders in 2025?"
    assert "How many orders in 2025?" in sql_llm.calls[0][1]
    assert out.as_turn().question == "How many orders in 2025?"


def test_without_history_nothing_is_interpreted(make_pipeline):
    pipe, _ = make_pipeline(["SELECT COUNT(*) AS n FROM orders"])
    pipe.rewriter = FollowUpRewriter(FakeLLM(['{"standalone": "unused"}']))
    out = pipe.ask("How many orders?")
    assert out.interpreted_as is None


class _StubPipeline:
    def __init__(self):
        self.histories = []

    def ask(self, question, history=()):
        self.histories.append(list(history))
        return PipelineResult(question, status="unanswerable", message="No such data.")


def test_cli_chat_passes_earlier_turns_to_follow_ups(monkeypatch):
    stub = _StubPipeline()
    monkeypatch.setattr(cli.Pipeline, "from_settings", classmethod(lambda cls, s: stub))
    replies = iter(["first question", "and a follow-up", "exit"])
    monkeypatch.setattr(cli.console, "input", lambda _prompt: next(replies))
    assert cli.main(["chat", "--provider", "fake"]) == 0
    assert [len(h) for h in stub.histories] == [0, 1]
    assert stub.histories[1][0].question == "first question"
